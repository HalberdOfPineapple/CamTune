import torch
import botorch
import numpy as np
from math import ceil
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from torch.quasirandom import SobolEngine
from botorch.generation import MaxPosteriorSampling

from ..mcts_copilot.node import Node
from camtune.utils import DEVICE, DTYPE, print_log



STATE_ATTRS = {
    'max_tree_depth': 10,
    'tree_depth_limit': 50,
    'increment_factor': 2.,
    'success_tolerance': 3,
    'failure_tolerance': 5,

    'cluster_type': 'kmeans', # or 'dbscan'
    'cluster_params': {
        'n_clusters': 2,
        'n_init': 'auto',
    },

    'classifier_type': 'svm', # or 'rf'
    'classifier_params': {
        'kernel': 'rbf',
        'gamma': 'auto',
    },
}

class MCTurboNodeII(Node):
    def __init__(
        self, 
        depth: int,
        parent: 'MCTurboNodeII', 
        label: int,
        bounds: torch.Tensor,
        seed: int,
        X: torch.Tensor, Y: torch.Tensor,
        classifier_type: str, classifier_params: Dict,
        cluster_type: str, cluster_params: Dict,  
    ):
        super().__init__(
            parent=parent, label=label, bounds=bounds, seed=seed,
            classifier_type=classifier_type, classifier_params=classifier_params,
            cluster_type=cluster_type, cluster_params=cluster_params,
            X=X, Y=Y, 
        )

        self.depth = depth
        self.best_value = Y.max().item()
        self.mean_value = Y.mean().item()
        self.best_x = X[Y.argmax()]
    
    def add_samples(self, X: torch.Tensor, Y: torch.Tensor):
        self.sample_bag[0] = torch.cat([self.sample_bag[0], X])
        self.sample_bag[1] = torch.cat([self.sample_bag[1], Y])
        if Y.max().item() > self.best_value:
            self.best_value = Y.max().item()
            self.best_x = X[Y.argmax()]
        
        # Backpropagate the new samples to the parent node
        if self.parent is not None:
            self.parent.add_samples(X, Y)


    # define an operator for node comparison based on the best value and then number of samples
    def __lt__(self, other: 'MCTurboNodeII') -> bool:
        if self.mean_value != other.mean_value:
            return self.mean_value < other.mean_value
        else:
            return self.num_samples < other.num_samples
    
    def fit(self
    ) -> Tuple[Dict[int, 'Node'], bool]:
        if self.sample_bag[0].shape[0] == 0:
            raise ValueError("Fit is called when the current node's sample bag is empty")

        # Let the trained SVM determine the positive and negative samples instead of KMeans
        # One thing for sure is that pos_data will be classified to pos_label while neg_data will be classified to 1 - pos_label
        max_mean_diff = -float('inf')
        for _ in range(5):
            fit_success = self.classifier.fit(self.sample_bag[0], self.sample_bag[1])
            if not fit_success:
                print_log(f"[Node.fit] Node {self.id} has only one label when clustering")
                return None, False
            
            labels = self.classifier.predict(self.sample_bag[0])
            label_set = set(labels)
            if len(label_set) == 1:
                continue
            
            mean_diff = np.abs(self.sample_bag[1][labels == 1].mean().cpu() - self.sample_bag[1][labels == 0].mean().cpu())
            if mean_diff > max_mean_diff:
                max_mean_diff = mean_diff
                best_labels = labels
        if max_mean_diff < 0: 
            print_log(f"[Node.fit] Node {self.id} has only one label when clustering")
            return None, False

        label_to_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for label in set(best_labels):
            label_to_data[label] = (
                self.sample_bag[0][labels == label],
                self.sample_bag[1][labels == label]
            )
        
        return label_to_data, True


class MCTurboStateII():
    def __init__(
        self,
        seed: int,
        bounds: torch.Tensor,
        X: torch.Tensor, Y: torch.Tensor,
        state_params: Dict[str, Any] = None,
    ):
        self.seed = seed
        self.dimension = bounds.shape[1]
        self.bounds = bounds

        for attr, default_val in STATE_ATTRS.items():
            if state_params is None or attr not in state_params:
                setattr(self, attr, default_val)
            elif isinstance(default_val, dict):
                setattr(self, attr, default_val)
                if state_params[attr] is not None:
                    for kk in state_params[attr]:
                        getattr(self, attr)[kk] = state_params[attr][kk]
            else:
                setattr(self, attr, state_params[attr])

        self.root_node: MCTurboNodeII = self.init_node(0, None, 0, X, Y)
        self.rebuild_path()

        self.success_counter = 0
        self.failure_counter = 0
        self.sec_order_fail_counter = 0
        self.restart_triggered = False

        print_log(f"[MCTurboStateII] ")

    @property
    def best_value(self) -> float:
        return self.root_node.best_value
    
    def rebuild_path(self):
        X_collected, Y_collected = self.root_node.sample_bag

        # The tree is built and path is selected when the state is reinitialized
        self.root_node: MCTurboNodeII = self.build_tree(X_collected, Y_collected)
        self.selected_path: List[MCTurboNodeII] = self.select_path()
    
    def check_region(self, X: torch.Tensor) -> np.array:
        return MCTurboNodeII.path_filter(self.selected_path, X)
    
    def update_state(self, X_next: torch.Tensor, Y_next: torch.Tensor):
        # Update counter variables
        if max(Y_next) > self.root_node.best_value:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        # Update sampling threshold corresponding to the success/failure counters
        if self.success_counter == self.success_tolerance:
            self.max_tree_depth = max(self.max_tree_depth / self.increment_factor, 1)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.max_tree_depth = self.increment_factor * self.max_tree_depth
            self.failure_counter = 0

        # Update the tree with the new data, through which the new data are recorded in the nodes (necessart for reinitialization)
        self.selected_path[-1].add_samples(X_next, Y_next)

        if self.max_tree_depth > self.tree_depth_limit:
            self.restart_triggered = True
    
    def generate_candidates(self, num_candidates: int, weights: torch.tensor, tr_length: float) -> torch.Tensor:
        selected_path: List[MCTurboNodeII] = self.selected_path
        x_center: torch.Tensor = selected_path[-1].best_x
        
        # Clamps all elements in input into the range [min, max]
        # tr_lbs, tr_ubs - (1, dim)
        tr_lbs = torch.clamp(x_center - tr_length / 2 * weights, 0.0, 1.0)
        tr_ubs = torch.clamp(x_center + tr_length / 2 * weights, 0.0, 1.0)

        # RASSP
        sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
        pert = sobol.draw(num_candidates).to(dtype=DTYPE, device=DEVICE)
        pert = tr_lbs + (tr_ubs - tr_lbs) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dimension, 1.0)
        mask = torch.rand(len(pert), self.dimension, dtype=DTYPE, device=DEVICE) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, self.dimension - 1, size=(len(ind),), device=DEVICE)] = 1

        # Create candidate points from the perturbations and the mask
        X_cands = x_center.expand(len(pert), self.dimension).clone()
        X_cands[mask] = pert[mask]

        return X_cands
    

    def select_path(self) -> List[MCTurboNodeII]:
        curr_node: MCTurboNodeII = self.root_node
        path: List[MCTurboNodeII] = [curr_node]
        while len(curr_node.children) > 0:
            curr_node = max(curr_node.children)
            path.append(curr_node)
        return path

    def build_tree(self, X: torch.Tensor, Y: torch.Tensor) -> MCTurboNodeII:
        MCTurboNodeII.clear_tree()
        root_node: MCTurboNodeII = self.init_node(0, None, 0, X, Y)

        splittable_nodes: List[MCTurboNodeII] = [root_node]
        depth, num_nodes = 0, 1
        while len(splittable_nodes) > 0:
            split_layer: List[MCTurboNodeII] = splittable_nodes
            splittable_nodes = []
            depth += 1

            for node in split_layer:
                label_to_data, split_succ = node.fit()
                if not split_succ:
                    break

                for label, (X, Y) in label_to_data.items():
                    if len(X) > 0:
                        child_node: MCTurboNodeII = self.init_node(depth, node, label, X, Y)
                        num_nodes += 1

                        if depth < self.max_tree_depth and len(X) > 2:
                            splittable_nodes.append(child_node)
                    else:
                        print_log(f'[MCTurboStateII] MCTurboNodeII {node.id} fitting warning: empty data for label {label}')
            

        print_log(f'[MCTurboStateII] Generated a tree with {num_nodes} nodes and depth {depth}')
        return root_node

    def init_node(self, depth: int, parent: MCTurboNodeII, label: int, X: torch.Tensor, Y: torch.Tensor) -> MCTurboNodeII:
        node = MCTurboNodeII(
            depth=depth,
            parent=parent, label=label, bounds=self.bounds, seed=self.seed, 
            cluster_type=self.cluster_type, cluster_params=self.cluster_params,
            classifier_type=self.classifier_type, classifier_params=self.classifier_params,
            X=X, Y=Y,
        )
        if parent is not None:
            parent.add_child(node, label)
        return node


    
