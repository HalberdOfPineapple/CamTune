import os 
import torch
import json
from torch.quasirandom import SobolEngine
from typing import Callable, List, Dict, Tuple, Any, Optional
from botorch.utils.transforms import unnormalize, normalize

from .base_optimizer import BaseOptimizer
from .sampler import LHSSampler, SobolSampler
from .turbo_optimizer import TuRBO
from .optim_utils import round_by_bounds, ObjFuncWrapper
from .winter_utils import MCTSLocalControl, OPTIMIZER_MAP, Node
# from .mcts_copilot import Node

from camtune.utils import (print_log, get_tree_dir, get_expr_name, 
                           get_benchmark_name, get_result_dir, get_log_idx,
                           DEVICE, DTYPE)

LOCAL_TURBO_PARAMS = {
    'num_restarts': 10,
    'raw_samples': 512,
    'n_candidates': 5000,
    'max_cholesky_size': float("inf"),
    'acqf': "ts",
    "init_bounding_box_length": 0.0005
}
MCTS_ATTRS = {
    'Cp': 1.,
    'leaf_size': 5,
    'global_num_init': 10,
    'node_selection_type': 'UCB',
    'save_path': False,
    'save_tr': True,

    'local_optimizer_type': 'turbo',
    'local_optimizer_params': LOCAL_TURBO_PARAMS,
    
    'local_control_params': {
        'init_mode': True,
        'sampling_mode': False,
        'real_mode': False,
        'attach_leaf': False,
        'jump_ratio': 0., 
        'jump_tolerance': 3,
    },

    'classifier_type': 'svm',
    'classifier_params': {
        'kernel_type': 'rbf',
        'gamma_type': 'auto',
        'cluster_method': 'kmeans',
    },
}

class WinterMCTS(BaseOptimizer):
    def __init__(
        self,
        bounds: torch.Tensor, # shape: (2, dimension)
        obj_func: Callable,
        batch_size: int = 1,
        seed: int = 0,
        discrete_dims: List[int] = [],
        optimizer_params: Dict[str, Any] = None,
        init_design: str = 'sobol',
        use_default: bool = True,
        default_conf: Optional[torch.Tensor] = None,
    ):
        super().__init__(bounds, obj_func,
                         batch_size=batch_size,
                         seed=seed, 
                         discrete_dims=discrete_dims, 
                         optimizer_params=optimizer_params,
                         init_design=init_design,
                         use_default=use_default,
                         default_conf=default_conf,
            )

        for k, v in MCTS_ATTRS.items():
            if optimizer_params is None or k not in optimizer_params:
                setattr(self, k, v)
            elif isinstance(v, dict):
                setattr(self, k, v)
                if optimizer_params[k] is not None:
                    for kk in optimizer_params[k]:
                        getattr(self, k)[kk] = optimizer_params[k][kk]
            else:
                setattr(self, k, optimizer_params[k])
        self.orig_obj_func = obj_func
        self.obj_func = ObjFuncWrapper(bounds, discrete_dims, obj_func)

        # ----------------------------------------------------------------
        # Local Optimizer Parameter Adjustment
        self.local_optimizer = None
        self.local_optimizer_cls = TuRBO
        self.local_num_init = optimizer_params.get('local_num_init', self.global_num_init)
        self.local_init_design = optimizer_params.get('local_init_design', self.init_design)
        self.local_optimizer_params['batch_size'] = self.batch_size
        self.local_optimizer_params['num_init'] = self.local_num_init

        if self.save_path:
            self.init_svm_save_dir()

        Node.Cp = self.Cp
        self.root: Node = None
        self.rebuild_iters = []
        self._X = torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
        self._Y = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        
        print_log('=' * 80, print_msg=True)
        print_log(f"[WinterMCTS] Initialized WinterMCTS with the following parameters:", print_msg=True)
        for k, v in MCTS_ATTRS.items():
            print_log(f"[WinterMCTS]\t{k}: {getattr(self, k)}", print_msg=True)

    @property
    def num_calls(self):
        return len(self._X)

    @property
    def X(self):
        xs = self._X
        xs = unnormalize(xs, self.bounds)
        xs[:, self.discrete_dims] = round_by_bounds(xs[:, self.discrete_dims], self.bounds[:, self.discrete_dims])

        return xs

    @property
    def Y(self):
        return self._Y
    
    @property
    def best_Y(self):
        return self._Y.max().item() if len(self._Y) > 0 else float('-inf')

    def initial_sampling(self, num_init: int):
        X_init = self.sampler.generate(num_init)
        if self.default_conf is not None:
            default_conf = normalize(self.default_conf, self.bounds)
            X_init = torch.cat([default_conf, X_init], dim=0)
            X_init = X_init[:self.global_num_init]

        Y_init = torch.tensor(
            [self.obj_func(x) for x in X_init], dtype=self.dtype, device=self.device,
        ).unsqueeze(-1)
    
        self._X = torch.cat([self._X, X_init], dim=0)
        self._Y = torch.cat([self._Y, Y_init], dim=0)

    def get_original_data(self):
        return self._X, self._Y
    
    # ----------------------------------------------------------------
    # Main optimization functions
    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.initial_sampling(self.global_num_init)
        self.tr_records = {}

        if not self.local_control_params['init_mode']: 
            num_evals += self.global_num_init

        while self.num_calls < num_evals:
            print_log('-' * 60)
            print_log(f'[WinterMCTS] optimize: Building tree from call: {self.num_calls}')
            self.build_tree()
            print_log('-' * 30)

            path: List[Node] = self.select()
            print_log('Selected Path: ')
            path_msg = [f'Node {node.id} with {node.sample_bag[0].shape[0]} samples' for node in path]
            print_log(' -> '.join(path_msg), print_msg=True)
            print_log('-' * 30, print_msg=True)

            cands, cand_vals, tr_records = self.local_modelling(
                num_evals=num_evals - self.num_calls,
                path=path, 
                bounds=self.bounds
            )
            print_log('-' * 30)

            self._X = torch.cat([self._X, cands], dim=0)
            self._Y = torch.cat([self._Y, cand_vals], dim=0)
            self.tr_records[self.rebuild_iters[-1]] = tr_records

            if self.save_path:
                self.save_tree_info(path, cands, cand_vals)

            print_log(f'[WinterMCTS] optimize: Best value found: {self.best_Y:.3f} after {self.num_calls} calls')
            print_log('-' * 60)
        if self.save_tr: self.save_tr_info()

        if not self.local_control_params['init_mode']: 
            self._X = self._X[self.global_num_init:]
            self._Y = self._Y[self.global_num_init:]

        print_log(f'[WinterMCTS] Iterations where search tree is rebuilt: {self.rebuild_iters}', print_msg=True)
        return self.X, self.Y
  
    def local_modelling(self, num_evals: int, path: List[Node], bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            path: List[Node] - A list of nodes from the root to the current node
            bounds: torch.Tensor - (2, dimension) - The bounds of the search space
        """
        if self.local_optimizer is None:
            self.local_optimizer: BaseOptimizer = self.local_optimizer_cls(
                bounds = self.bounds,
                obj_func = self.orig_obj_func,
                batch_size = self.batch_size,
                seed = self.seed,
                discrete_dims= self.discrete_dims,
                optimizer_params = self.local_optimizer_params,
                init_design=self.init_design,
                use_default=False,
                manual_seed=False, # Local optimizer should avoid resetting torch random seed 
            )

        
        local_control = MCTSLocalControl(
            global_best_y=self.best_Y,
            **self.local_control_params,
        )
        return self.local_optimizer.optimize_local(num_evals, local_control, path)
    
    # ----------------------------------------------------------------
    # MCTS related functions
    def select(self) -> List[Node]:
        curr_node: Node = self.root

        path: List[Node] = [curr_node]
        while len(curr_node.children) > 0:
            curr_node = max(curr_node.children, 
                            key=lambda node: self.get_node_score(node))
            path.append(curr_node)
        
        Node.check_path(path)
        return path

    def clear_tree(self):
        curr_node: Node = self.root
        if curr_node is not None:
            node_queue: List[Node] = [curr_node]
            while len(node_queue) > 0:
                curr_node = node_queue.pop(0)
                node_queue.extend(curr_node.children)

                Node.obj_counter -= 1
                del curr_node
        Node.reset()

    def init_node(self, parent: Node, X: torch.Tensor, Y: torch.Tensor, label: int, is_pos: bool):
        node = Node(
            parent=parent,
            label=label,
            bounds=self.bounds,
            dtype=self.dtype,
            device=self.device,
            classifier_type=self.classifier_type,
            classifier_params=self.classifier_params,
            seed=self.seed,
        )
        node.update_sample_bag(X, Y)

        # Checking label assignment
        if parent is not None:
            child_label: int = node.label
            target_label: int = parent.classifier.pos_label if is_pos else 1 - parent.classifier.pos_label
            if child_label != target_label:
                raise ValueError(f"Node {node.id} with {node.sample_bag[0].shape[0]} got unexpected label {child_label} "
                                 f" (expceted {target_label}, where is_pos={is_pos})")

            target_child_data: torch.Tensor = node.sample_bag[0]
            labels = parent.classifier.predict(target_child_data)
            if not (labels ==  child_label).all():
                raise ValueError(f"Node {node.id} with {node.sample_bag[0].shape[0]} samples is not "
                                 f"classified to label {child_label} by its parent's classifier. "
                                 f"Got labels: {labels}")

        return node

    def build_tree(self):
        self.clear_tree()
        self.rebuild_iters.append(self.num_calls)

        self.root = self.init_node(parent=None, X=self._X, Y=self._Y, label=None, is_pos=None)
        nodes_splittable: List[Node] = [self.root]
        while len(nodes_splittable) > 0:
            nodes_to_split = [node for node in nodes_splittable]
            nodes_splittable = []
            for node in nodes_to_split:
                try:
                    # pos_data - ((num_pos_samples, dimension), (num_pos_samples, 1))
                    # neg_data - ((num_neg_samples, dimension), (num_neg_samples, 1))
                    pos_data, neg_data, pos_label = node.fit()
                except:
                    continue

                if pos_data[0].shape[0] == 0 or neg_data[0].shape[0] == 0:
                    continue

                if pos_data[0].shape[0] > 0:
                    # pos_child's data should be classified to pos_label by its parent's classifier
                    pos_child: Node = self.init_node(node, pos_data[0], pos_data[1], label=pos_label, is_pos=True)
                    if pos_child.num_visits > self.leaf_size:
                        nodes_splittable.append(pos_child)
    
                if neg_data[0].shape[0] > 0:
                    neg_child: Node = self.init_node(node, neg_data[0], neg_data[1], label=1 - pos_label, is_pos=False)
                    if neg_child.num_visits > self.leaf_size:
                        nodes_splittable.append(neg_child)
    
                print_log(f'[WinterMCTS] Build Tree: Split onto {pos_data[0].shape[0]} positive samples (node: {pos_child.id})'
                      f' and {neg_data[0].shape[0]} negative samples (node: {neg_child.id})'
                      f' where pos label is {pos_label}')

        Node.sum_leaf_scores()
        


    
    def get_node_score(self, node: Node) -> float:
        if self.node_selection_type == 'UCB':
            return node.get_UCB(self.Cp)
        else:
            raise ValueError(f"Node selection type {self.node_selection_type} not supported")
    
    # ----------------------------------------------------------------
    # Saving related functions
    def init_svm_save_dir(self):
        log_idx = get_log_idx()
        if log_idx == 0:
            self.svm_save_dir = os.path.join(get_tree_dir(), get_expr_name())
        else:
            self.svm_save_dir = os.path.join(get_tree_dir(), f'{get_expr_name()}_{log_idx}')
    
    def init_tr_save_path(self):
        log_idx = get_log_idx()
        if log_idx == 0:
            self.tr_save_dir = os.path.join(get_tree_dir(), f'{get_expr_name()}_tr.log')
        else:
            self.tr_save_dir = os.path.join(get_tree_dir(), f'{get_expr_name()}_{log_idx}')
    
    def save_tr_info(self):
        if get_log_idx() == 0:
            tr_save_path = os.path.join(get_result_dir(), f'{get_expr_name()}_tr.json')
        else:
            tr_save_path = os.path.join(get_result_dir(), f'{get_expr_name()}_{get_log_idx()}_tr.json')
        print_log(f'[WinterMCTS] save_tr_info: Saving the trust region data into {tr_save_path}', print_msg=True)
        with open(tr_save_path, 'w') as f:
            json.dump(self.tr_records, f)
    
        

    def save_tree_info(self, path: List[Node], phase_x: torch.Tensor, phase_y: torch.Tensor):
        print_log(f'[WinterMCTS] save_tree_info: Saving the whole tree info at call {self.num_calls}', print_msg=True)
        svm_save_dir = os.path.join(self.svm_save_dir, f'{self.rebuild_iters[-1]}')
        if not os.path.exists(svm_save_dir):
            os.makedirs(svm_save_dir)

        nodes = Node.node_map.values()
        leaf_nodes = [node for node in nodes if len(node.children) == 0]
        for node in nodes:
            pkl_path: str = os.path.join(svm_save_dir, f'{node.id}.pkl')
            node.save_classifier(pkl_path)

        path_fp: str = os.path.join(svm_save_dir, 'path.txt')
        with open(path_fp, 'w') as f:
            for node in path:
                f.write(f'{node.id}\n')

        label_path: str = os.path.join(svm_save_dir, f'labels.txt')
        with open(label_path, 'w') as f:
            for node in nodes:
                f.write(f'{node.id}: {node.label}\n')
        
        children_path: str = os.path.join(svm_save_dir, f'children.txt')
        with open(children_path, 'w') as f:
            for node in nodes:
                if len(node.children) == 0:
                    continue
                children_ids = [child.id for child in node.children]
                f.write(f'{node.id}: {children_ids}\n')
        
        for leaf_node in leaf_nodes:
            leaf_save_dir = os.path.join(svm_save_dir, f'{leaf_node.id}_leaf')
            if not os.path.exists(leaf_save_dir):
                os.makedirs(leaf_save_dir)

            data_path = os.path.join(leaf_save_dir, 'initial_data.log')
            with open(data_path, 'w') as f:
                node_x, node_y = leaf_node.sample_bag[0].cpu().numpy(), leaf_node.sample_bag[1].cpu().numpy()
                for x, y in zip(node_x, node_y):
                    x = str(list(x))
                    f.write(f"{y}, {x}\n")
            
            if leaf_node.id == path[-1].id:
                phase_data_path = os.path.join(leaf_save_dir, 'phase_data.log')
                with open(phase_data_path, 'w') as f:
                    phase_x_cpu, phase_y_cpu = phase_x.cpu().numpy(), phase_y.cpu().numpy()
                    for x, y in zip(phase_x_cpu, phase_y_cpu):
                        x = str(list(x))
                        f.write(f"{y}, {x}\n")
