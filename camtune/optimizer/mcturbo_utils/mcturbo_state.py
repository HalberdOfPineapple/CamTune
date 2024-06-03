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
    'n_candidates': 5000,
    'leaf_size': 10,
    'increment_factor': 2.0,
    'success_tolerance': 3,
    'failure_tolerance': 5,
    'init_bounding_box_length': 0.0001,

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

class MCTurboNode(Node):
    def __init__(
        self, 
        depth: int,
        parent: 'MCTurboNode', 
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
    def __lt__(self, other: 'MCTurboNode') -> bool:
        if self.best_value != other.best_value:
            return self.best_value < other.best_value
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


class MCTurboState():
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

        self.root_node: MCTurboNode = self.init_node(0, None, 0, X, Y)
        self.reinit_state()

        print_log(f"[MCTuRBOState] ")

    @property
    def best_value(self) -> float:
        return self.root_node.best_value
    
    def generate_candidates(self, num_candidates: int, weights: torch.tensor) -> torch.Tensor:
        selected_path: List[MCTurboNode] = self.select_path()
        idx = 0
        while idx < len(selected_path) and selected_path[idx].num_samples > self.sampling_threshold:
            idx += 1
        self.last_leaf_node = selected_path[-1]
        selected_path = selected_path[:idx+1]

        # Generate candidates from the selected node
        pert, x_center = MCTurboNode.gen_samples_in_region_around_center(
            num_samples=num_candidates, 
            path=selected_path, 
            seed=self.seed,
            weights=weights,
            init_bounding_box_length=self.init_bounding_box_length,
        )
        if len(pert) < num_candidates:
            print_log(f"[MCTuRBOState] Warning: Only generated {len(pert)}/{num_candidates} candidates", print_msg=True)
            num_rand_cands = num_candidates - len(pert)
            sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
            pert_rand = sobol.draw(num_rand_cands).to(dtype=DTYPE, device=DEVICE)
            pert = torch.cat([pert, pert_rand], dim=0)


        # RAASP
        prob_perturb = min(20.0 / self.dimension, 1.0)
        mask = torch.rand(len(pert), self.dimension, dtype=DTYPE, device=DEVICE) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, self.dimension - 1, size=(len(ind),), device=DEVICE)] = 1

        # Create candidate points from the perturbations and the mask
        X_cands = x_center.expand(len(pert), self.dimension).clone()
        X_cands[mask] = pert[mask]

        return X_cands
    
    def reinit_state(self):
        X_collected, Y_collected = self.root_node.sample_bag
        self.root_node: MCTurboNode = self.build_tree(X_collected, Y_collected)

        selected_path: List[MCTurboNode] = self.select_path()
        self.sampling_threshold = selected_path[ceil(len(selected_path) / 2)].num_samples

        self.failure_counter = 0
        self.success_counter = 0
        self.restart_triggered = False
    
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
            self.sampling_threshold = min(self.increment_factor * self.sampling_threshold, self.leaf_size)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.sampling_threshold = self.sampling_threshold / self.increment_factor
            self.failure_counter = 0

        # Update the tree with the new data, through which the new data are recorded in the nodes (necessart for reinitialization)
        self.last_leaf_node.add_samples(X_next, Y_next)

        if self.sampling_threshold < self.leaf_size:
            self.restart_triggered = True


    def select_path(self) -> List[MCTurboNode]:
        curr_node: MCTurboNode = self.root_node
        path: List[MCTurboNode] = [curr_node]
        while len(curr_node.children) > 0:
            curr_node = max(curr_node.children)
            path.append(curr_node)
        return path

    def build_tree(self, X: torch.Tensor, Y: torch.Tensor) -> MCTurboNode:
        MCTurboNode.clear_tree()
        root_node: MCTurboNode = self.init_node(0, None, 0, X, Y)

        splittable_nodes: List[MCTurboNode] = [root_node]
        depth, num_nodes = 0, 1
        while len(splittable_nodes) > 0:
            split_layer: List[MCTurboNode] = splittable_nodes
            splittable_nodes = []

            for node in split_layer:
                label_to_data, split_succ = node.fit()
                if not split_succ:
                    break

                for label, (X, Y) in label_to_data.items():
                    if len(X) > 0:
                        child_node: MCTurboNode = self.init_node(depth+1, node, label, X, Y)
                        num_nodes += 1

                        if child_node.num_samples > self.leaf_size:
                            splittable_nodes.append(child_node)
                    else:
                        print_log(f'[MCTurboState] MCTurboNode {node.id} fitting warning: empty data for label {label}')
            depth += 1

        print_log(f'[MCTurboState] Generated a tree with {num_nodes} nodes and depth {depth}')
        return root_node

    def init_node(self, depth: int, parent: MCTurboNode, label: int, X: torch.Tensor, Y: torch.Tensor) -> MCTurboNode:
        node = MCTurboNode(
            depth=depth,
            parent=parent, label=label, bounds=self.bounds, seed=self.seed, 
            cluster_type=self.cluster_type, cluster_params=self.cluster_params,
            classifier_type=self.classifier_type, classifier_params=self.classifier_params,
            X=X, Y=Y,
        )
        if parent is not None:
            parent.add_child(node, label)
        return node


    
