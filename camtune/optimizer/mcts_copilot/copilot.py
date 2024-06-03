import os
import math
import torch
import botorch
from dataclasses import dataclass
from torch.distributions import Normal
from torch.quasirandom import SobolEngine
from typing import List, Dict, Any, Callable, Optional, Tuple, Union

import botorch
botorch.settings.debug(True)
from botorch.models import SingleTaskGP

from .node import Node
from ..optim_utils import RAASP
from camtune.utils import DEVICE, DTYPE, print_log, get_result_dir, get_expr_name


COPILOT_ATTRS = {
    'save_path': True,
    'gen_cands': False,
    'gen_cands_raasp': True,
    'gen_batch': True,
    'bounding_box_mode': True,

    'acqf': 'ts',

    'temperature': 1.0,
    'enable_guide_restart': False,
    'guide_restart_num_init': None,
    'fit_feat_vals': True,
    'extra_by_tr': True,
    'record_hyper_volume': False,
    'reject_sampling_times': 5,
    'cluter_score_threshold': 0.,

    'base_Cp': 0.05,
    'tree_depth': 2,
    'tree_depth_min': 1,
    'tree_depth_max': 5,
    'enable_tree_restart': True,

    'init_bounding_box_length': 0.005,
    'model_within_partition': False,

    'node_selection_type': 'UCB',
    'adapt_temperature': False,
    'adapt_Cp': False,


    'classifier_cls': 'base',
    'cluster_type': 'kmeans', # or 'dbscan'
    'cluster_params': {},
    'classifier_type': 'svm', # or 'rf
    'classifier_params': {
        'kernel': 'rbf',
        'gamma': 'auto',
    },

    'max_cholesky_size': float("inf"),
    'num_candidates': 2000,
    'init_bounding_box_length': 0.001,
}
SINGLE_SAMPLING_THRE = 3
@dataclass
class TreeState:
    tree_depth: int = 2
    tree_depth_min: int = 1
    tree_depth_max: int = 5

    failure_counter: int = 0
    failure_tolerance: int = 5

    success_counter: int = 0
    success_tolerance: int = 3
    
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        attrs = ['tree_depth', 'tree_depth_min', 'tree_depth_max', 'failure_tolerance', 'success_tolerance']
        print_log("=" * 50, print_msg=True)
        print_log("[TreeState] Initialized with the following attributes:", print_msg=True)
        for attr in attrs:
            print_log(f"[TreeState]\t{attr}:\t{getattr(self, attr)}", print_msg=True)

def expected_improvement(
        X_cands: torch.Tensor, 
        model: botorch.models.SingleTaskGP, 
        best_f: float,
    ) -> torch.Tensor:
    with torch.no_grad():
        posterior = model.posterior(X_cands) # shape (num_cands,)
        mu = posterior.mean
        sigma = posterior.variance.sqrt()
        
        # Calculate Z
        Z = (mu - best_f) / sigma
        normal = Normal(0, 1)
        
        # Calculate EI
        ei = (mu - best_f) * normal.cdf(Z) + sigma * normal.log_prob(Z).exp()
        
        # Handle case when sigma is zero
        ei[sigma == 0] = 0.0
        
    return ei


class MCTSCopilot:
    def __init__(
        self,
        seed: int,
        num_evals: int,
        bounds: torch.Tensor,
        max_func_value: float,
        min_max_factors: Tuple = (6, 2),
        increment_step: int = 1, # 2,
        failure_tolerance: int = 5,
        success_tolerance: int = 3,
        tree_params: Dict[str, Any] = None,
        **kwargs,
    ):
        self.seed = seed
        self.dimension = bounds.shape[1]
        self.bounds = bounds
        
        for k, v in COPILOT_ATTRS.items():
            if tree_params is None or k not in tree_params:
                setattr(self, k, v)
            elif isinstance(v, dict):
                setattr(self, k, v)
                if tree_params[k] is not None:
                    for kk in tree_params[k]:
                        getattr(self, k)[kk] = tree_params[k][kk]
            else:
                setattr(self, k, tree_params[k])

        self.increment_step = increment_step

        self.state_params = {
            'tree_depth': self.tree_depth,
            'tree_depth_min': max(self.tree_depth_min, 1),
            'tree_depth_max': min(self.tree_depth_max, 6),

            'failure_tolerance': failure_tolerance,
            'success_tolerance': success_tolerance,
        }
        self.state = TreeState(**self.state_params)

        if self.save_path:
            self.path_save_dir = os.path.join(get_result_dir(), f'{get_expr_name()}_path')
            os.makedirs(self.path_save_dir, exist_ok=True)

        print_log('=' * 80, print_msg=True)
        print_log(f"[MCTSCopilotIV] Initialized with the following configurations:", print_msg=True)
        for attr, value in self.__dict__.items():
            if attr in COPILOT_ATTRS:
                print_log(f"[MCTSCopilotIV]\t{attr}:\t{value}", print_msg=True)

    @property
    def state_msg(self) -> str:
        msg = (
                f"[MCTSCopilotIV] "
                f"tree depth: {self.state.tree_depth} / {self.state.tree_depth_min} | "
                f"num. succ: {self.state.success_counter}/{self.state.success_tolerance} | "
                f"num. fail: {self.state.failure_counter}/{self.state.failure_tolerance}"
            )
        return msg
    
    def guide_restart(
            self, 
            num_init: int, 
            X: torch.Tensor, 
            Y: torch.Tensor,
            init_bounding_box_length: float=None,
    ) -> torch.Tensor:
        if init_bounding_box_length is None:
            init_bounding_box_length = self.init_bounding_box_length
        if self.guide_restart_num_init is not None:
            num_init = self.guide_restart_num_init

        Node.clear_tree()
        root_node: Node = self.init_node(None, 0, X, Y)
        curr_node: Node = root_node
        path: List[Node] = [root_node]
        leaf_size = min(X.shape[0] / 5, 5)
        while True:
            label_to_data, split_succ = curr_node.fit()
            if not split_succ:
                break

            splittable_nodes: List[Node] = []
            for label, (X, Y) in label_to_data.items():
                if len(X) > 0:
                    node = self.init_node(curr_node, label, X, Y)
                    if node.num_samples > leaf_size:
                        splittable_nodes.append(node)
                else:
                    print_log(f'[MCTSCopilotIV] Node {curr_node.id} fitting warning: empty data for label {label}')

            if len(splittable_nodes) == 0:
                break

            sorted_children = sorted(splittable_nodes, key=lambda node: node.get_UCB(Node.Cp))
            curr_node = sorted_children[-1]
            path.append(curr_node)
        
        sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
        return Node.generate_samples_in_region(
            num_samples=num_init, path=path, sobol=sobol, seed=self.seed,
            init_bounding_box_length=init_bounding_box_length,
        )


    def generate_candidates(
            self, 
            num_candidates: int, 
            X: torch.Tensor, 
            Y: torch.Tensor,
            weights: torch.Tensor,
            lb: Union[float, torch.Tensor] = 0.0,
            ub: Union[float, torch.Tensor] = 1.0,
            negate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[SingleTaskGP]]:
        if negate: Y = -Y
        self.build_tree(X, Y)
        x_center = X[Y.argmax().item()]

        num_cands_map = {}
        for node_id, score in self.leaf_score_map.items():
            num_cands = int(num_candidates * score)
            if num_cands > 0:
                num_cands_map[node_id] = num_cands
        
        seed = self.seed + len(X)
        sobol: SobolEngine = SobolEngine(self.dimension, scramble=True, seed=seed)
        X_cands = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
        for node_id, num_cands in num_cands_map.items():
            node = Node.node_map[node_id]
            node_path: List[Node] = node.get_path()
            X_cands_node: torch.Tensor = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
            
            sample_trial_cnt = 0
            
            while X_cands_node.shape[0] < num_cands and sample_trial_cnt < SINGLE_SAMPLING_THRE:
                X_cands_node_iter = Node.bounding_box_sampling(
                    num_samples=num_cands - X_cands_node.shape[0], path=node_path,
                    sobol=sobol, lb=lb, ub=ub, weights=weights,
                    num_candidates=self.num_candidates,
                    init_bounding_box_length=self.init_bounding_box_length,
                )
                X_cands_node = torch.cat([X_cands_node, X_cands_node_iter], dim=0)
                sample_trial_cnt += 1
            valid_sample_cnt = X_cands_node.shape[0]

            if X_cands_node.shape[0] < num_cands:
                num_extra_cands = num_cands - X_cands_node.shape[0]
                X_cands_extra = sobol.draw(num_extra_cands).to(dtype=DTYPE, device=DEVICE)
                X_cands_extra = lb + X_cands_extra * (ub - lb)
                X_cands_node = torch.cat([X_cands_node, X_cands_extra], dim=0)

            X_cands = torch.cat([X_cands, X_cands_node[:num_cands]], dim=0)
            print_log(f'[MCTSCopilotIV] Node {node_id} generated {valid_sample_cnt}/{num_cands} samples', print_msg=True)

        if self.gen_cands_raasp:
            prob_perturb = min(20.0 /self.dimension, 1.0)
            mask = torch.rand(len(X_cands), self.dimension, dtype=DTYPE, device=DEVICE) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, self.dimension - 1, size=(len(ind),), device=DEVICE)] = 1

            # Create candidate points from the perturbations and the mask
            X_cands_final = x_center.expand(len(X_cands), self.dimension).clone()
            X_cands_final[mask] = X_cands[mask]
        else:
            X_cands_final = X_cands

        return X_cands_final, None

    def generate_batch(
            self, 
            num_samples: int, 
            X_cands: torch.Tensor,
            X: torch.Tensor, 
            Y: torch.Tensor,
            model: botorch.models.SingleTaskGP,
            negate: bool = False,
    ) -> torch.Tensor:
        if negate: Y = -Y

        if self.gen_cands:
            # Assume generate_candidates is called before generate_batch
            X_cands_score = torch.tensor([self.tree_infer(X_cand, self.leaf_score_map) for X_cand in X_cands])
            return self.select_candidates(
                num_samples, X_cands, X_cands_score, model, negate, best_f=max(Y).item()
            )
        
        self.build_tree(X, Y)
        X_cands_scores = torch.tensor([self.tree_infer(X_cand, self.leaf_score_map) for X_cand in X_cands])

        X_next = self.select_candidates(
            num_samples, X_cands, X_cands_scores, model, negate, best_f=max(Y).item()
        )
        return X_next
    
    def select_candidates(
            self,
            num_samples: int,
            X_cands: torch.Tensor,
            X_cands_scores: torch.Tensor,
            model: botorch.models.SingleTaskGP,
            negate: bool = False,
            best_f: float = -float("inf"),
    ):
        if self.acqf == 'ts':
            with torch.no_grad():
                y_cand = model.likelihood(model(X_cands)).sample(torch.Size([num_samples])).detach().cpu() # shape (num_samples, num_cands)
                y_cand = -y_cand if negate else y_cand
            X_next = torch.empty((num_samples, self.dimension), dtype=DTYPE, device=DEVICE)
            for i in range(num_samples):
                # Element-wise multiplication of the scores provided by X_cands_scores and y_cand[i]
                score_dist = X_cands_scores * y_cand[i]
                X_next[i] = X_cands[score_dist.argmax()]
        elif self.acqf == 'ei':
            with torch.no_grad():
                ei_values = expected_improvement(X_cands, model, best_f).detach().cpu().flatten() # shape (num_cands,)
            adjusted_ei = ei_values * X_cands_scores

            _, top_indices = torch.topk(adjusted_ei, num_samples)
            X_next = X_cands[top_indices]
        return X_next



    def tree_infer(self, X: torch.Tensor, leaf_score_map: Dict[int, float]):
        curr_node: Node = Node.node_map[0]
        while len(curr_node.children) > 0:
            label = curr_node.classifier.predict(X.reshape(1, -1))[0]
            curr_node = Node.node_map[curr_node.label_to_children[label]]
        return leaf_score_map[curr_node.id]

    def update_state(self, Y_next: torch.Tensor, step_length: float=0.):
        if max(Y_next) > self.state.best_value + step_length: 
            self.state.success_counter += 1
            self.state.failure_counter = 0
        else:
            self.state.success_counter = 0
            self.state.failure_counter += 1
        
        if self.state.success_counter == self.state.success_tolerance:
            self.state.tree_depth = max(self.state.tree_depth - self.increment_step, self.state.tree_depth_min)
            self.state.success_counter = 0
        elif self.state.failure_counter == self.state.failure_tolerance:
            self.state.tree_depth = min(self.state.tree_depth + self.increment_step, self.state.tree_depth_max)
            self.state.failure_counter = 0

        self.state.best_value = max(Y_next).item()
        if self.state.tree_depth > self.state.tree_depth_max and self.enable_tree_restart:
            self.state.restart_triggered = True

    def init_node(self, parent: Node, label: int, X: torch.Tensor, Y: torch.Tensor) -> Node:
        node = Node(
            parent=parent, label=label, bounds=self.bounds, seed=self.seed, 
            cluster_type=self.cluster_type, cluster_params=self.cluster_params,
            classifier_type=self.classifier_type, classifier_params=self.classifier_params,
            X=X, Y=Y, node_selection_type=self.node_selection_type,
            fit_feat_vals=self.fit_feat_vals,
            classifier_cls=self.classifier_cls,
        )
        if parent is not None:
            parent.add_child(node, label)
        return node

    def build_tree(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[int, float]:
        Node.clear_tree()
        if self.adapt_Cp:
            Cp_factor = self.base_Cp * (1 / 5) ** ((self.state.tree_depth - self.state.tree_depth_min) / (self.state.tree_depth_max - self.state.tree_depth_min))
            Node.set_Cp(torch.abs(Y).max().item() * Cp_factor)

        root_node: Node = self.init_node(None, 0, X, Y)
        split_layer: List[Node] = [root_node]

        depth = 1
        while len(split_layer) > 0 and depth < self.state.tree_depth:
            next_layer: List[Node] = split_layer
            split_layer = []
            depth_incremented = False
            for curr_node in next_layer:
                label_to_data, split_succ = curr_node.fit()
                if not split_succ: continue
                if self.cluter_score_threshold > 0 and curr_node.classifier.cluster_score < self.cluter_score_threshold:
                    print_log(f'[MCTSCopilotIV] Node {curr_node.id} low cluster score ({curr_node.classifier.cluster_score:.4f})')
                    continue

                splittable_nodes: List[Node] = []
                for label, (X, Y) in label_to_data.items():
                    if len(X) > 0:
                        depth_incremented = True
                        node = self.init_node(curr_node, label, X, Y)
                        if node.num_samples > 2: splittable_nodes.append(node)
                    else:
                        print_log(f'[MCTSCopilotIV] Node {curr_node.id} fitting warning: empty data for label {label}')

                if len(splittable_nodes) == 0:
                    break

                # exploits = [torch.mean(node.sample_bag[1]).detach().cpu().item() for node in splittable_nodes]
                # explores = [2. * Node.Cp * math.sqrt(2. * math.log(node.parent.num_samples) / node.num_samples) for node in splittable_nodes]
                # score_msg = ' - '.join([f'Node {node.id} ({exploits[i]:.4f} + {explores[i]:4f};)' for i, node in enumerate(splittable_nodes)])
                # print_log(f'[MCTSCopilotIV] {score_msg}')
 
                split_layer.extend(splittable_nodes)
            if depth_incremented: depth += 1
        
        self.leaf_score_map = self.build_leaf_score_map(depth)

    
    def build_leaf_score_map(self, depth: int):
        Node.leaf_nodes = [node for node in Node.node_map.values() if len(node.children) == 0]

        if not self.adapt_temperature:
            temperature = self.temperature
        else:
            temperature = self.temperature * (0.1 / self.temperature) ** ((self.state.tree_depth - self.state.tree_depth_min) / (self.state.tree_depth_max - self.state.tree_depth_min))

        # Build a score map for each leaf node by softmaxing scores over all leaf nodes
        leaf_score_map = {}
        leaf_scores = torch.tensor([node.score / temperature for node in Node.leaf_nodes]).reshape(-1, 1)
        leaf_scores = torch.softmax(leaf_scores, dim=0)
        for i, leaf_node in enumerate(Node.leaf_nodes):
            leaf_score_map[leaf_node.id] = leaf_scores[i].item()
        print_log(f'[MCTSCopilotIV] Built leaf score map: {leaf_score_map}')

        print_log(f'[MCTSCopilotIV] Tree built with {depth} layers and {len(Node.leaf_nodes)} leaf nodes')
        return leaf_score_map
