import torch
import math
import numpy as np
from torch.quasirandom import SobolEngine

from typing import Dict, Tuple, List, Callable, Union
from .classifier import CLASSIFIER_MAP, BaseClassifier

from camtune.utils import DEVICE, DTYPE, print_log

class Node:
    Cp: float = 1.

    obj_counter: int = 0
    node_map: Dict[int, 'Node'] = {}

    leaf_nodes: List['Node'] = []
    leaf_total_score: float = 0.

    def __init__(
        self, 
        parent: 'Node', 
        label: int,
        bounds: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        classifier_type: str, 
        classifier_params: Dict,
        seed: int,
    ):
        self.id = Node.obj_counter
        Node.obj_counter += 1
        Node.node_map[self.id] = self

        self.parent = parent
        self.label = label
        if parent is not None:
            self.parent.children.append(self)

        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.dtype = dtype
        self.device = device

        self.seed = seed
        self.classifier: BaseClassifier = CLASSIFIER_MAP[classifier_type.lower()](
            bounds=bounds,
            seed=self.seed, 
            classifier_params=classifier_params)

        self.sample_bag: List[torch.Tensor, torch.Tensor] = [
            torch.empty((0, self.dimension), dtype=dtype, device=device),
            torch.empty((0, 1), dtype=dtype, device=device),
        ]

        self.children: List[Node] = []

    @classmethod
    def reset(cls):
        cls.obj_counter = 0
        cls.node_map = {}
        cls.leaf_nodes = []
        cls.leaf_total_score = 0.
    
    @property
    def num_visits(self) -> int:
        return len(self.sample_bag[0])
    
    def get_UCB(self, Cp: float) -> float:
        if self.parent is None:
            return 0.

        exploit_value = torch.mean(self.sample_bag[1]).detach().cpu().item()
        exploration_value = 2. * Cp * \
            math.sqrt(2. * math.log(self.parent.num_visits) / self.num_visits)
        
        return exploit_value + exploration_value


    def update_sample_bag(self, X: torch.Tensor, Y: torch.Tensor):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows but got {} and {}".format(X.shape[0], Y.shape[0]))
        
        self.sample_bag[0] = torch.cat((self.sample_bag[0], X), dim=0)
        self.sample_bag[1] = torch.cat((self.sample_bag[1], Y), dim=0)
    
    
    def fit(self
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int]:
        if self.sample_bag[0].shape[0] == 0:
            raise ValueError("Fit is called when the current node's sample bag is empty")

        # labels = self.classifier.fit(self.sample_bag[0], self.sample_bag[1])
        self.classifier.fit(self.sample_bag[0], self.sample_bag[1])

        # Let the trained SVM determine the positive and negative samples instead of KMeans
        # One thing for sure is that pos_data will be classified to pos_label while neg_data will be classified to 1 - pos_label
        labels = self.classifier.predict(self.sample_bag[0])
        pos_data = (
            self.sample_bag[0][labels == self.classifier.pos_label],
            self.sample_bag[1][labels == self.classifier.pos_label])
        neg_data = (
            self.sample_bag[0][labels != self.classifier.pos_label],
            self.sample_bag[1][labels != self.classifier.pos_label])

        return pos_data, neg_data, self.classifier.pos_label

    def predict_label(self, X: torch.Tensor) -> np.array: 
        # labels - (num_samples, )
        labels = self.classifier.predict(X)
        return labels

    def save_classifier(self, classifier_save_path: str):
        self.classifier.save(classifier_save_path)

    def plot_node_region(self, plot_save_path: str):
        labels = self.classifier.predict(self.sample_bag[0])
        pos_X, pos_Y = (
            self.sample_bag[0][labels == self.classifier.pos_label].detach().cpu().numpy(),
            self.sample_bag[1][labels == self.classifier.pos_label].detach().cpu().numpy())
        pos_mean = np.mean(pos_Y)

        neg_X, neg_Y = (
            self.sample_bag[0][labels != self.classifier.pos_label].detach().cpu().numpy(),
            self.sample_bag[1][labels != self.classifier.pos_label].detach().cpu().numpy())
        neg_mean = np.mean(neg_Y)
    
        lbs = self.bounds.cpu().numpy()[0]
        ubs = self.bounds.cpu().numpy()[1]
        xx = np.linspace(lbs[0], ubs[0], 300)
        yy = np.linspace(lbs[1], ubs[1], 300)

        xv, yv = np.meshgrid(xx, yy)
        pred_labels = self.classifier.predict(
            torch.from_numpy(np.c_[xv.ravel(), yv.ravel()]).to(dtype=self.dtype, device=self.device))
        pred_labels = pred_labels.reshape( xv.shape )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.contourf(xv, yv, pred_labels, alpha=0.4)

        ax.scatter(pos_X[:, 0], pos_X[:, 1], marker='o', 
                   label=f"pos-{pos_mean:.2f}-{len(pos_X)}")
        ax.scatter(neg_X[:, 0], neg_X[:, 1], marker='x',
                   label=f"neg-{neg_mean:.2f}-{len(neg_X)}")
        ax.legend(loc="best")

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xlim([lbs[0], ubs[0]])
        ax.set_ylim([lbs[1], ubs[1]])
    
        plt.savefig(plot_save_path)
        plt.close()

    @classmethod
    def sum_leaf_scores(cls):
        cls.leaf_nodes = [node for node in cls.node_map.values() if len(node.children) == 0]
        cls.leaf_total_score = sum([np.exp(node.get_UCB(Node.Cp)) for node in cls.leaf_nodes])
    
    @staticmethod
    def get_partition_score(cand: torch.Tensor) -> float:
        node = Node.node_map[0]
        while len(node.children) > 0:
            label = node.predict_label(cand.unsqueeze(0))[0]
            for child in node.children:
                if child.label == label:
                    node = child
                    break
        
        return np.exp(node.get_UCB(Node.Cp)) / Node.leaf_total_score
            
        
    @staticmethod
    def path_filter(path: List['Node'], candidates: torch.Tensor) -> np.array:
        """
        Args:
            path: List[Node] - A list of nodes from the root to the current node
            candidates: torch.Tensor - (num_candidates, dimension)

        Returns:
            choices: np.array - (num_candidates, ) - A boolean array indicating whether each candidate is accepted
        """
        choices: np.array = np.full((candidates.shape[0], ), True)
        for i in range(len(path) - 1):
            curr_node: Node = path[i]
            target_label: int = path[i + 1].label

            labels = curr_node.predict_label(candidates[choices])
            choices[choices] = labels == target_label

            if choices.sum() == 0:
                break

        return choices

    @staticmethod
    def check_path(path: List['Node']):
        leaf: Node = path[-1]
        leaf_X: torch.Tensor = leaf.sample_bag[0]

        in_regions = Node.path_filter(path, leaf_X)
        num_in_regions = in_regions.sum()

        print_log('-' * 20, print_msg=True)
        print_log(f"[Node.check_path] {num_in_regions} / {leaf_X.shape[0]} samples of the leaf node {leaf.id}"
            f" are in the region of the leaf node {leaf.id} with label {leaf.label}", print_msg=True)

    @staticmethod
    def bounding_box_sampling(
        num_samples: int,
        sobol: SobolEngine,
        path: List['Node'],
        init_bounding_box_length: float,
        weights: torch.Tensor = None, # (dimension, )
        lb: Union[float, torch.Tensor] = 0.0,
        ub: Union[float, torch.Tensor] = 1.0,
        num_candidates: int = 5000,
    ) -> torch.Tensor:
        X_in_region: torch.Tensor = path[-1].sample_bag[0]
        dimension = X_in_region.shape[-1]

        bounding_box_length = init_bounding_box_length
        weights = weights if weights is not None else \
                    torch.ones(dimension, dtype=DTYPE, device=DEVICE)
    
        sobol_num_samples = num_candidates if num_candidates > 0 else 2 * num_samples
        X_init: torch.Tensor = torch.empty((0, dimension), dtype=DTYPE, device=DEVICE)
        sign = False
        for _ in range(X_in_region.shape[0]):
            ratio: float  = 1.

            # randomly select a point in the region as the region_center
            region_center = X_in_region[torch.randint(X_in_region.shape[0], (1, ))]

            # sobol_samples - (sobol_num_samples, dim)
            sobol_samples = sobol.draw(sobol_num_samples).to(dtype=DTYPE, device=DEVICE)
            while bounding_box_length < 1.0:
                # bounding_box_lbs, bounding_box_ubs - (1, dim)
                bouning_box_lbs = torch.clamp(region_center - bounding_box_length / 2 * weights, lb, ub)
                bouning_box_ubs = torch.clamp(region_center + bounding_box_length / 2 * weights, lb, ub)

                sobol_cands = sobol_samples * (bouning_box_ubs - bouning_box_lbs) + bouning_box_lbs
                in_region = Node.path_filter(path, sobol_cands) # (num_in_region_samples, )
                
                ratio = in_region.sum().item() / sobol_num_samples
                if ratio < 1.:
                    sign = True
                    X_init = torch.cat((X_init, sobol_cands[in_region]), dim=0)
                    break

                bounding_box_length *= 2
        if not sign:
            X_init = torch.cat((X_init, sobol_cands[in_region]), dim=0)
        
        X_init = X_init[np.random.permutation(len(X_init))][:num_candidates]
        return X_init

    @staticmethod
    def generate_samples_in_region(
        num_samples: int,
        path: List['Node'],
        init_bounding_box_length: float,
        seed: int,
        weights: torch.Tensor = None, # (dimension, )
        lb: Union[float, torch.Tensor] = 0.0,
        ub: Union[float, torch.Tensor] = 1.0,
        num_candidates: int = 5000,
    ) -> torch.Tensor:    
        sobol = SobolEngine(path[-1].sample_bag[0].shape[-1], scramble=True, seed=seed)
        X_init = Node.bounding_box_sampling(
            num_samples, sobol, path, 
            init_bounding_box_length, 
            weights, lb, ub, num_candidates
        )
    
        if X_init.shape[0] > num_samples:
            # randomly select num_samples samples from X_init
            rand_indices = torch.randperm(X_init.shape[0])[:num_samples]
            X_init = X_init[rand_indices]
        elif X_init.shape[0] > 0:
            print_log(f"[Node.generate_samples_in_region] Not enough samples generated in the region. ({X_init.shape[0]} / {num_samples}) ")
        else:
            print_log(f"[Node.generate_samples_in_region] No legal samples generated in the region.")
            X_init = sobol.draw(num_samples).to(dtype=DTYPE, device=DEVICE)

        return X_init