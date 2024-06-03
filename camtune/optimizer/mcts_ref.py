import os 
import torch
import numpy as np
from torch.quasirandom import SobolEngine
from typing import Callable, List, Dict, Tuple, Any, Optional
from botorch.utils.transforms import unnormalize, normalize

from .base_optimizer import BaseOptimizer
from .optim_utils import round_by_bounds
from .mcstref_utils import MCTS

from camtune.utils.logger import print_log
from camtune.utils.vars import DEVICE, DTYPE

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

# This optimizer assumes input to the objective function has already been converted to target range
# input is np.array of shape (batch_size, dimension)
class ObjFuncWrapper:
    def __init__(self, bounds: torch.Tensor, discrete_dims: List[int], obj_func: Callable):
        self.bounds = bounds
        self.discrete_dims = discrete_dims
        self.lb, self.ub = bounds[0, :].detach().cpu().numpy(), bounds[1, :].detach().cpu().numpy()
        self.obj_func = obj_func

    def __call__(self, input_x: np.array) -> float:
        x = torch.Tensor(input_x.reshape(1, -1)).to(dtype=DTYPE, device=DEVICE)
        x[:, self.discrete_dims] = round_by_bounds(x[:, self.discrete_dims], self.bounds[:, self.discrete_dims])

        eval_val = -self.obj_func(x)
        if isinstance(eval_val, torch.Tensor):
            eval_val = eval_val.cpu().item()
        return eval_val

class MCTSRefOptimizer(BaseOptimizer):
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
        self.mcts = MCTS(
            lb=self.bounds[0,:].detach().cpu().numpy(),
            ub=self.bounds[1,:].detach().cpu().numpy(),
            dims=self.dimension,
            ninits=self.global_num_init,
            func=self.obj_func,
            Cp=self.Cp,
            leaf_size=self.leaf_size,
            kernel_type=self.classifier_params['kernel_type'],
            gamma_type=self.classifier_params['gamma_type'],
            solver_type=self.local_optimizer_type,
            batch_size=self.batch_size,
            local_num_init=optimizer_params.get('local_num_init', self.global_num_init),
        )
        
        print_log('=' * 80, print_msg=True)
        print_log(f"[MCTSRef] Initialized MCTSRef with the following parameters:", print_msg=True)
        for k, v in MCTS_ATTRS.items():
            print_log(f"[MCTSRef]\t{k}: {getattr(self, k)}", print_msg=True)

    @property
    def num_calls(self):
        return len(self._X)

    @property
    def X(self):
        xs = torch.tensor([s[0] for s in self.mcts.samples], dtype=self.dtype, device=self.device)
        xs[:, self.discrete_dims] = round_by_bounds(xs[:, self.discrete_dims], self.bounds[:, self.discrete_dims])

        return xs

    @property
    def Y(self):
        return torch.tensor([s[1] for s in self.mcts.samples], dtype=self.dtype, device=self.device)
    
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
        xs = torch.tensor([s[0] for s in self.mcts.samples], dtype=self.dtype, device=self.device)
        ys = torch.tensor([s[1] for s in self.mcts.samples], dtype=self.dtype, device=self.device).unsqueeze(-1)
        norm_xs = normalize(xs, self.bounds)

        return norm_xs, ys

    def optimize(self, num_steps: int):
        self.mcts.search(num_steps)
        return self.X, self.Y

