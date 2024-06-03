import torch
import math
import numpy as np
import pygmo as pg
from joblib import dump, load
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize
from torch.quasirandom import SobolEngine
from typing import Dict, Tuple, List, Callable, Union

from .classifier import BaseClassifier, CLASSIFIER_MAP
from ..optim_utils import RAASP, train_gp

from camtune.utils import DEVICE, DTYPE, print_log
SINGLE_SAMPLING_THRE = 5
BASE_CP = 0.5

class Node:
    Cp: float = BASE_CP

    obj_counter: int = 0
    node_map: Dict[int, 'Node'] = {}

    leaf_nodes: List['Node'] = []
    leaf_total_score: float = 0.

    def __init__(
        self, 
        parent: 'Node', 
        label: int,
        bounds: torch.Tensor,
        seed: int,
        classifier_type: str, classifier_params: Dict,
        cluster_type: str, cluster_params: Dict,
        X: torch.Tensor, Y: torch.Tensor,
        classifier_cls: str='base',
        node_selection_type: str='UCB',
        num_cands_for_vol_esti: int=0,
        fit_feat_vals: bool=True,
    ):
        
        self.id = Node.obj_counter
        Node.obj_counter += 1
        Node.node_map[self.id] = self
        self.node_selection_type = node_selection_type

        self.parent = parent
        self.label = label
        self.sample_bag: List[torch.Tensor, torch.Tensor] = [X, Y]
        self.candidate_bag: torch.Tensor = torch.empty((0, bounds.shape[1]), dtype=DTYPE, device=DEVICE)

        self.children: List[Node] = []
        self.children_labels: Dict[int, Node] = {} # child node id -> label
        self.label_to_children: Dict[int, int] = {} # label -> child node id

        self.seed = seed
        self.bounds = bounds
        self.dimension = bounds.shape[1]

        self.classifier_cls: str = classifier_cls.lower()
        self.cluster_type: str = cluster_type
        self.classifier_type: str = classifier_type
        self.cluster_params: Dict = cluster_params
        self.classifier_params: Dict = classifier_params
        self.classifier: BaseClassifier = CLASSIFIER_MAP[classifier_cls](
            bounds=bounds,
            seed=self.seed, 
            fit_feat_vals=fit_feat_vals,
            classifier_type=classifier_type, classifier_params=classifier_params,
            cluster_type=cluster_type, cluster_params=cluster_params
        )

        self.hyper_volume = 0 if num_cands_for_vol_esti == 0 else self.estimate_hypervolume(num_cands_for_vol_esti)

    def add_child(self, child: 'Node', label: int):
        self.children.append(child)
        self.children_labels[child.id] = label
        self.label_to_children[label] = child.id
        child.parent_check()

    def parent_check(self):
        if self.parent is not None:
            if self.id not in self.parent.children_labels:
                raise ValueError(f"[CopilotNode] Node {self.id} is not in its parent's (Node {self.parent.id}) children_labels")

            target_label: int = self.parent.children_labels[self.id]
            if self.label != target_label:
                raise ValueError(f"[CopilotNode] Node {self.id} with {self.sample_bag[0].shape[0]} got unexpected label {self.label} "
                                 f" (expected label: {target_label}")

            target_child_data: torch.Tensor = self.sample_bag[0]
            labels = self.parent.classifier.predict(target_child_data)
            if not (labels ==  self.label).all():
                raise ValueError(f"[CopilotNode] Node {self.id} with {self.sample_bag[0].shape[0]} samples is not "
                                 f"classified to label {self.label} by its parent's classifier. "
                                 f"Got labels: {labels}")

    @classmethod
    def clear_tree(cls):
        node_ids = list(cls.node_map.keys())
        for node_id in node_ids:
            del cls.node_map[node_id]
        
        cls.Cp = BASE_CP
        cls.obj_counter = 0
        cls.node_map = {}
        cls.leaf_nodes = []
        cls.leaf_total_score = 0.
    
    @property
    def num_samples(self) -> int:
        return len(self.sample_bag[0])
    
    @property
    def score(self) -> float:
        if self.node_selection_type.lower() == 'ucb':
            return self.get_UCB(Node.Cp)
        elif self.node_selection_type.lower() == 'best':
            return self.sample_bag[1].cpu().max().item()
        elif self.node_selection_type.lower() == 'mean':
            return self.sample_bag[1].cpu().mean().item()
        else:
            raise ValueError(f"Invalid node selection type: {self.node_selection_type}")
    
    def get_UCB(self, Cp: float) -> float:
        if self.parent is None:
            return 0.

        exploit_value = torch.mean(self.sample_bag[1]).detach().cpu().item()
        exploration_value = 2. * Cp * \
            math.sqrt(2. * math.log(self.parent.num_samples) / self.num_samples)
        
        return exploit_value + exploration_value

    def get_path(self) -> List['Node']:
        path = [self]
        parent_node: Node = self.parent
        while parent_node is not None:
            path.append(parent_node)
            parent_node = parent_node.parent
        return path[::-1]

    
    def fit(self
    ) -> Tuple[Dict[int, 'Node'], bool]:
        if self.sample_bag[0].shape[0] == 0:
            raise ValueError("Fit is called when the current node's sample bag is empty")

        # labels = self.classifier.fit(self.sample_bag[0], self.sample_bag[1])
        fit_success = self.classifier.fit(self.sample_bag[0], self.sample_bag[1])
        cluter_sore = self.classifier.cluster_score
        if not fit_success:
            print_log(f"[Node.fit] Node {self.id} has only one label when clustering")
            return None, False

        # Let the trained SVM determine the positive and negative samples instead of KMeans
        # One thing for sure is that pos_data will be classified to pos_label while neg_data will be classified to 1 - pos_label
        labels = self.classifier.predict(self.sample_bag[0])
        label_set = set(labels)
        if len(label_set) == 1:
            print_log(f"[Node.fit] Node {self.id} has only one label {label_set.pop()}")
            return None, False

        label_to_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for label in set(labels):
            label_to_data[label] = (
                self.sample_bag[0][labels == label],
                self.sample_bag[1][labels == label]
            )
        
        return label_to_data, True

    def predict_label(self, X: torch.Tensor) -> np.array: 
        # labels - (num_samples, )
        labels = self.classifier.predict(X)
        return labels

    def save_classifier(self, classifier_save_path: str):
        self.classifier.save(classifier_save_path)

    @classmethod
    def set_Cp(cls, Cp: float):
        cls.Cp = Cp

    @classmethod
    def sum_leaf_scores(cls):
        cls.leaf_nodes = [node for node in cls.node_map.values() if len(node.children) == 0]
        cls.leaf_total_score = sum([np.exp(node.score) for node in cls.leaf_nodes])
    
    @staticmethod
    def get_partition_score(cand: torch.Tensor) -> float:
        node = Node.node_map[0]
        while len(node.children) > 0:
            label = node.predict_label(cand.unsqueeze(0))[0]
            for child in node.children:
                if child.label == label:
                    node = child
                    break
        
        return np.exp(node.score) / Node.leaf_total_score
            
        
    @staticmethod
    def path_filter(path: List['Node'], candidates: torch.Tensor) -> List[int]:
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

            if candidates[choices].shape[0] == 0:
                print_log(f"[Node.path_filter] No candidates left after the {i}-th node")

            labels = curr_node.predict_label(candidates[choices])
            choices[choices] = labels == target_label

            if choices.sum() == 0:
                break

        return choices
    
    @staticmethod
    def save_path(
        path: List['Node'], 
        save_dir: str, 
        iteration: int,
    ):
        classifier_seq = []
        for i in range(len(path) - 1):
            curr_classifier = path[i].classifier
            target_label = path[i + 1].label
            classifier_seq.append((curr_classifier, target_label))
        dump(classifier_seq, f"{save_dir}/path_{iteration}.joblib")

    @staticmethod
    def check_path(path: List['Node']):
        leaf: Node = path[-1]
        leaf_X: torch.Tensor = leaf.sample_bag[0]

        in_regions = Node.path_filter(path, leaf_X)
        num_in_regions = in_regions.sum()

        print_log('-' * 20)
        print_log(f"[Node.check_path] {num_in_regions} / {leaf_X.shape[0]} samples of the leaf node {leaf.id}"
            f" are in the region of the leaf node {leaf.id} with label {leaf.label}")

    def generate_samples_in_region_random(
        self,
        num_candidates: int,
        seed: int,
    ):
        path: List['Node'] = [self]
        parent_node: Node = self.parent
        while parent_node is not None:
            path.append(parent_node)
            parent_node = parent_node.parent
        path = path[::-1]

        sobol = SobolEngine(self.dimension, scramble=True, seed=seed)
        sobol_samples = sobol.draw(num_candidates).to(dtype=DTYPE, device=DEVICE)
        in_region = Node.path_filter(path, sobol_samples)
        X_cands = sobol_samples[in_region]

        return X_cands[:num_candidates]
    
    def estimate_hypervolume(self, num_candidates: int):
        points_in_region = self.generate_samples_in_region_random(num_candidates, self.seed)
        
        in_region_ratio = (points_in_region.shape[0] + self.sample_bag[0].shape[0]) / (num_candidates + self.sample_bag[0].shape[0])
        return in_region_ratio
    
    def get_hypervolume(self):
        points_in_region = self.generate_samples_in_region_random(2000, self.seed)
        points_in_region = torch.cat((self.sample_bag[0], points_in_region), dim=0)
        print_log(f"[Node.get_hypervolume] Number of samples in the region: {points_in_region.shape[0]}")

        # points_in_region = torch.cat((self.sample_bag[0], self.candidate_bag), dim=0)
        # print_log(f"[Node.get_hypervolume] Number of samples in the region: {self.sample_bag[0].shape[0]} + {self.candidate_bag.shape[0]} = {points_in_region.shape[0]}")

        points_in_region = points_in_region.cpu().numpy()
        hv = pg.hypervolume(points_in_region)
        return hv.compute([1.0] * self.dimension)
    
    
            
        
    @staticmethod
    def local_modelling(
        leaf_node: 'Node',
        tr_length: float,
        seed: int,
        num_candidates: int,
        max_cholesky_size: float = float("inf"),
        training_steps: int = 100,
    ) -> Tuple[torch.Tensor, SingleTaskGP]:
        path: List['Node'] = [leaf_node]
        parent_node: Node = leaf_node.parent
        while parent_node is not None:
            path.append(parent_node)
            parent_node = parent_node.parent
        path = path[::-1]

        if len(path) == 1:
            pass

        X_in_region: torch.Tensor = leaf_node.sample_bag[0]
        Y_in_region: torch.Tensor = leaf_node.sample_bag[1]
        region_center = X_in_region[Y_in_region.argmax()]
        dimension = X_in_region.shape[-1]

        model = train_gp(dimension, X_in_region, standardize(Y_in_region), 
                         max_cholesky_size=max_cholesky_size, training_steps=training_steps)
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # Clamps all elements in input into the range [min, max]
        # tr_lbs, tr_ubs - (1, dim)
        tr_lbs = torch.clamp(region_center - tr_length / 2 * weights, 0.0, 1.0)
        tr_ubs = torch.clamp(region_center + tr_length / 2 * weights, 0.0, 1.0)

        sobol = SobolEngine(dimension, scramble=True, seed=seed)
        sample_trial_cnt = 0
        X_cands = torch.empty((0, dimension), dtype=DTYPE, device=DEVICE)
        while X_cands.shape[0] < num_candidates and sample_trial_cnt < SINGLE_SAMPLING_THRE:
            X_cands_iter = RAASP(sobol, region_center, tr_lbs, tr_ubs, num_candidates=num_candidates)
            X_cands_iter = X_cands_iter[Node.path_filter(path, X_cands_iter)]
            X_cands = torch.cat([X_cands, X_cands_iter], dim=0)
            sample_trial_cnt += 1
        if X_cands.shape[0] < num_candidates:
            print_log(f"[Node] local_modelling: Not enough samples generated in the region: {X_cands.shape[0]} < {num_candidates}", print_msg=True)
            X_cands_extra = RAASP(sobol, region_center, tr_lbs, tr_ubs, num_candidates=num_candidates - X_cands.shape[0])
            X_cands = torch.cat([X_cands, X_cands_extra], dim=0)
        return X_cands, model



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
    
        # sobol_num_samples = num_candidates if num_candidates > 0 else 2 * num_samples
        sobol_num_samples = 2 * num_samples
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
        
        X_init = X_init[np.random.permutation(len(X_init))][:num_samples]
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
        sobol: SobolEngine = None,
    ) -> torch.Tensor:    
        sobol = SobolEngine(path[-1].sample_bag[0].shape[-1], scramble=True, seed=seed) if sobol is None else sobol
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

    @staticmethod
    def gen_samples_in_region_around_center(
        num_samples: int,
        path: List['Node'],
        init_bounding_box_length: float,
        seed: int,
        weights: torch.Tensor = None, # (dimension, )
        lb: Union[float, torch.Tensor] = 0.0,
        ub: Union[float, torch.Tensor] = 1.0,
        sobol: SobolEngine = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_in_region: torch.Tensor = path[-1].sample_bag[0]
        Y_in_region: torch.Tensor = path[-1].sample_bag[1]
        region_center = X_in_region[Y_in_region.argmax()]

        dimension = X_in_region.shape[-1]
        bounding_box_length = init_bounding_box_length
        weights = weights if weights is not None else \
                    torch.ones(dimension, dtype=DTYPE, device=DEVICE)
        
        sobol = SobolEngine(dimension, scramble=True, seed=seed) if sobol is None else sobol
        sobol_samples = sobol.draw(num_samples).to(dtype=DTYPE, device=DEVICE)

        X_init: torch.Tensor = torch.empty((0, dimension), dtype=DTYPE, device=DEVICE)
        ratio: float  = 1.

        while bounding_box_length < 1.0:
            # bounding_box_lbs, bounding_box_ubs - (1, dim)
            bouning_box_lbs = torch.clamp(region_center - bounding_box_length / 2 * weights, lb, ub)
            bouning_box_ubs = torch.clamp(region_center + bounding_box_length / 2 * weights, lb, ub)

            sobol_cands = sobol_samples * (bouning_box_ubs - bouning_box_lbs) + bouning_box_lbs
            in_region = Node.path_filter(path, sobol_cands) # (num_in_region_samples, )
                
            ratio = in_region.sum().item() / num_samples
            if ratio < 1.:
                X_init = torch.cat((X_init, sobol_cands[in_region]), dim=0)
                break

            bounding_box_length *= 2
        if X_init.shape[0] == 0: X_init = torch.cat((X_init, sobol_cands[in_region]), dim=0)

        return X_init, region_center

    @staticmethod
    def gen_samples_around_center(
        num_samples: int,
        path: List['Node'],
        tr_length: float,
        tr_length_min: float,
        seed: int,
        weights: torch.Tensor = None, # (dimension, )
        sobol: SobolEngine = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_in_region: torch.Tensor = path[-1].sample_bag[0]
        Y_in_region: torch.Tensor = path[-1].sample_bag[1]
        region_center = X_in_region[Y_in_region.argmax()]

        dimension = X_in_region.shape[-1]
        bounding_box_length = tr_length
        weights = weights if weights is not None else \
                    torch.ones(dimension, dtype=DTYPE, device=DEVICE)
        
        sobol = SobolEngine(dimension, scramble=True, seed=seed) if sobol is None else sobol
        X_init: torch.Tensor = torch.empty((0, dimension), dtype=DTYPE, device=DEVICE)
        ratio: float  = 1.

        while bounding_box_length > tr_length_min:
            # bounding_box_lbs, bounding_box_ubs - (1, dim)
            bouning_box_lbs = torch.clamp(region_center - bounding_box_length / 2 * weights, 0., 1.)
            bouning_box_ubs = torch.clamp(region_center + bounding_box_length / 2 * weights, 0., 1.)

            cands = RAASP(sobol, region_center, bouning_box_lbs, bouning_box_ubs, num_candidates=num_samples)

            in_region = Node.path_filter(path, cands) # (num_in_region_samples, )    
            ratio = in_region.sum().item() / num_samples
            if ratio > 0.:
                X_init = torch.cat((X_init, cands[in_region]), dim=0)
                if X_init.shape[0] >= num_samples:
                    break

            bounding_box_length /= 2
        return X_init, region_center