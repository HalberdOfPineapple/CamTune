import os 
import torch
from torch.quasirandom import SobolEngine
from typing import Callable, List, Dict, Tuple 
from botorch.utils.transforms import unnormalize

from .node import Node
from .turbo_component import OPTIMIZER_MAP, BaseOptimizer
from .local_control import MCTSLocalControl

from camtune.utils import print_log
from camtune.utils.paths import get_tree_dir, get_expr_name, get_benchmark_name


class MCTS:
    def __init__(
        self,
        bounds: torch.Tensor, # shape: (2, dimension)
        num_init: int,
        local_num_init: int,
        obj_func: Callable,

        optimizer_type: str,
        optimizer_params: Dict,
        classifier_type: str,
        classifier_params: Dict,
        local_control_params: Dict,

        seed: int = 0,
        Cp: float = 1., 
        leaf_size: int = 20,
        node_selection_type: str = 'UCB',
        initial_sampling_method: str = 'Sobol',
        save_path: bool=False,
        default_conf: torch.Tensor = None,
    ):

        self.seed = seed

        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.obj_func = obj_func
        self.default_conf = default_conf
        

        self.num_init = num_init
        self.local_num_init = local_num_init
        self.initial_sampling_method = initial_sampling_method

        self.local_control_params = local_control_params

        self.optimizer_cls = OPTIMIZER_MAP[optimizer_type.lower()]
        self.optimizer_params = optimizer_params

        self.classifier_type = classifier_type
        self.classifier_params = classifier_params

        self.Cp = Cp
        self.leaf_size = leaf_size
        self.node_selection_type = node_selection_type

        self.save_path: bool = save_path
        if self.save_path:
            self.init_svm_save_dir()
        

        self.dtype = bounds.dtype
        self.device = bounds.device

        self.rebuild_iters = []
        self.X = torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
        self.Y = torch.empty((0, 1), dtype=self.dtype, device=self.device)

        self.best_X: torch.Tensor = None
        self.best_Y: torch.Tensor = None

    
    @property
    def num_calls(self):
        return len(self.X)
    
    def init_svm_save_dir(self):
        log_idx = 0
        svm_save_dir = os.path.join(get_tree_dir(), get_benchmark_name(), get_expr_name())
        while os.path.exists(svm_save_dir):
            log_idx += 1
            svm_save_dir = os.path.join(get_tree_dir(), get_benchmark_name(), f'{get_expr_name()}_{log_idx}')
        self.svm_save_dir = svm_save_dir

    def initial_sampling(self):
        if self.initial_sampling_method.lower() == 'sobol':
            sobol = SobolEngine(dimension=self.dimension, scramble=True, seed=self.seed)
            X_init = sobol.draw(n=self.num_init).to(dtype=self.dtype, device=self.device)
            if self.default_conf is not None:
                X_init = torch.cat([self.default_conf, X_init], dim=0)
                X_init = X_init[:self.num_init]

            Y_init = torch.tensor(
                [self.obj_func(x) for x in X_init], dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)
        else:
            raise ValueError(f"Initial sampling method {self.initial_sampling_method} not supported")
    
        self.X = torch.cat([self.X, X_init], dim=0)
        self.Y = torch.cat([self.Y, Y_init], dim=0)

    def clear_tree(self):
        curr_node: 'Node' = self.root
        node_queue: List['Node'] = [curr_node]
        while len(node_queue) > 0:
            curr_node = node_queue.pop(0)
            node_queue.extend(curr_node.children)

            Node.obj_counter -= 1
            del curr_node

        self.root = self.init_node(parent=None, X=self.X, Y=self.Y, label=None, is_pos=None)

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

        nodes_splittable: List['Node'] = [self.root]
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
                    if pos_child.num_visits >= self.leaf_size:
                        nodes_splittable.append(pos_child)
    
                if neg_data[0].shape[0] > 0:
                    neg_child: Node = self.init_node(node, neg_data[0], neg_data[1], label=1 - pos_label, is_pos=False)
                    if neg_child.num_visits >= self.leaf_size:
                        nodes_splittable.append(neg_child)
    
                print_log(f'[WinterMCTS] Build Tree: Split onto {pos_data[0].shape[0]} positive samples (node: {pos_child.id})'
                      f' and {neg_data[0].shape[0]} negative samples (node: {neg_child.id})'
                      f' where pos label is {pos_label}')


    def get_node_score(self, node: Node) -> float:
        if self.node_selection_type == 'UCB':
            return node.get_UCB(self.Cp)
        else:
            raise ValueError(f"Node selection type {self.node_selection_type} not supported")

    def save_path_info(self, path: List[Node], phase_x: torch.Tensor, phase_y: torch.Tensor):
        if len(path) < 2:
            return

        print_log(f'[WinterMCTS] save_path_info: Saving path info at call {self.num_calls}', print_msg=True)
        svm_save_dir = os.path.join(self.svm_save_dir, f'{self.rebuild_iters[-1]}')
        if not os.path.exists(svm_save_dir):
            os.makedirs(svm_save_dir)

        for i in range(len(path) - 1):
            node, child = path[i], path[i+1]
            
            pkl_path: str = os.path.join(svm_save_dir, f'{node.id}.pkl')
            node.save_classifier(pkl_path)

            label_path: str = os.path.join(svm_save_dir, f'{node.id}.txt')
            with open(label_path, 'w') as f:
                f.write(f'{child.label}')
        
        leaf_node = path[-1]
        if leaf_node.sample_bag[0].shape[0] > 0:
            leaf_save_dir = os.path.join(svm_save_dir, f'{leaf_node.id}_leaf')
            if not os.path.exists(leaf_save_dir):
                os.makedirs(leaf_save_dir)

            data_path = os.path.join(leaf_save_dir, 'initial_data.log')
            with open(data_path, 'w') as f:
                node_x, node_y = leaf_node.sample_bag[0].cpu().numpy(), leaf_node.sample_bag[1].cpu().numpy()
                for x, y in zip(node_x, node_y):
                    x = str(list(x))
                    f.write(f"{y}, {x}\n")
            
            phase_data_path = os.path.join(leaf_save_dir, 'phase_data.log')
            with open(phase_data_path, 'w') as f:
                phase_x_cpu, phase_y_cpu = phase_x.cpu().numpy(), phase_y.cpu().numpy()
                for x, y in zip(phase_x_cpu, phase_y_cpu):
                    x = str(list(x))
                    f.write(f"{y}, {x}\n")
            

    def select(self) -> List['Node']:
        curr_node: Node = self.root

        path: List[Node] = [curr_node]
        while len(curr_node.children) > 0:
            curr_node = max(curr_node.children, 
                            key=lambda node: self.get_node_score(node))
            path.append(curr_node)
        
        Node.check_path(path)
        return path

    def local_modelling(self, num_evals: int, path: List['Node'], bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            path: List[Node] - A list of nodes from the root to the current node
            bounds: torch.Tensor - (2, dimension) - The bounds of the search space
        """
        optimizer: BaseOptimizer = self.optimizer_cls(
            obj_func=self.obj_func,
            bounds=bounds,
            num_init=self.local_num_init, # Currently the local optimizer's num_init is the same as the global one
            seed=self.seed,
            optimizer_params=self.optimizer_params,
        )
        leaf_node = path[-1]

        local_control = MCTSLocalControl(
            global_best_y=self.best_Y,
            **self.local_control_params,
        )

        return optimizer.optimize(
            X_in_region=leaf_node.sample_bag[0], 
            Y_in_region=leaf_node.sample_bag[1], 
            num_evals=num_evals, 
            path=path,
            local_control=local_control,
        )

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.initial_sampling()
        self.root = self.init_node(parent=None, X=self.X, Y=self.Y, label=None, is_pos=None)

        while self.num_calls < num_evals:
            print_log('-' * 60)
            print_log(f'[WinterMCTS] optimize: Building tree from call: {self.num_calls}')
            self.build_tree()
            print_log('-' * 30)

            path: List['Node'] = self.select()
            print_log('Selected Path: ')
            for node in path:
                print_log(f'Node {node.id} with {node.sample_bag[0].shape[0]} samples ->', end=' ')
            print_log('')
            print_log('-' * 30)

            cands, cand_vals = self.local_modelling(
                num_evals=num_evals - self.num_calls,
                path=path, 
                bounds=self.bounds
            )
            print_log('-' * 30)

            self.X = torch.cat([self.X, cands], dim=0)
            self.Y = torch.cat([self.Y, cand_vals], dim=0)
            if self.save_path:
                self.save_path_info(path, cands, cand_vals)

            # self.num_calls += cands.shape[0]            
            if self.best_Y is None or cand_vals.max() > self.best_Y:
                self.best_X = cands[cand_vals.argmax()]
                self.best_Y = cand_vals.max()

            print_log(f'[WinterMCTS] optimize: Best value found: {self.best_Y:.3f} after {self.num_calls} calls')
            print_log('-' * 60)

        print_log(f'[WinterMCTS] Iterations where search tree is rebuilt: {self.rebuild_iters}', print_msg=True)
        return self.X, self.Y, self.best_X, self.best_Y
