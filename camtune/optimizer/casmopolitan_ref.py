import os
import math
import json
import torch
import numpy as np
from typing import Callable, Optional, Tuple, Union, List, Dict, Any

import botorch
botorch.settings.debug(True)

from .base_optimizer import BaseOptimizer
from .optim_utils import round_by_bounds
from .casmopolitan_utils import MixedOptimizer

from camtune.utils import (print_log, get_expr_name, get_result_dir, get_log_idx, DEVICE, DTYPE)


CASMO_ATTRS = {
    'success_tolerance': 3,
    'failure_tolerance': 5,

    'length_init': 0.8,
    'length_min': 0.5 ** 7,
    'length_max': 1.6,

    'length_discrete_init': 20,
    'length_discrete_min': 5,
    'length_discrete_max': 30,

    # Which ones of the continuous dimensions additionally are constrained to have integer values only?
    'int_constrained_dims': None, 
    'wrap_discrete': True,

    # whether to fit an auxiliary GP over the best points encountered in all previous restarts, and
    # sample the points with maximum variance for the next restart.
    'guided_restart': True,

    'save_tr': False,
    "use_ard": True, # Automatic Relevance Determination (ARD)

    'num_restarts': 10,
    'raw_samples': 512,
    'n_candidates': 5000,
    'max_cholesky_size': float("inf"),
    'acqf': "ts",
    "init_bounding_box_length": 0.0001,
    "training_steps": 100,
}
SINGLE_SAMPLING_THRE = 5


    
class NpObjFuncWrapper:
    def __init__(self, bounds: torch.Tensor, discrete_dims: List[int], obj_func: Callable):
        self.discrete_dims = discrete_dims
        self.obj_func = obj_func

        self.bounds = bounds

    def __call__(self, x_batch: np.ndarray):
        x_batch_torch = torch.tensor(x_batch, dtype=self.bounds.dtype, device=self.bounds.device)
        eval_vals = []
        for x in x_batch_torch:
            if len(self.discrete_dims) > 0:
                x[self.discrete_dims] = round_by_bounds(x[self.discrete_dims], self.bounds[:, self.discrete_dims])

            eval_val = self.obj_func(x)
            if isinstance(eval_val, torch.Tensor):
                eval_val = eval_val.cpu().item()
            eval_vals.append(-eval_val)
        return np.array(eval_vals)

class Casmopolitan(BaseOptimizer):
    def __init__(
        self, 
        bounds: torch.Tensor,
        obj_func: Callable, 
        batch_size: int = 1,
        seed: int=0, 
        discrete_dims: List[int] = [],
        optimizer_params: Dict[str, Any] = None,
        init_design: str= "lhs",
        use_default: bool = True,
        default_conf: Optional[torch.Tensor] = None,
        manual_seed: bool = True,
    ):
        super().__init__(bounds, obj_func, 
                        batch_size=batch_size,
                        seed=seed, 
                        discrete_dims=discrete_dims, 
                        optimizer_params=optimizer_params,
                        init_design=init_design,
                        use_default=use_default,
                        default_conf=default_conf,
                        manual_seed=manual_seed,
                    )
        self.obj_func = NpObjFuncWrapper(bounds, discrete_dims, obj_func)
        self.num_init: int = optimizer_params.get('num_init', 10)

        for k, v in CASMO_ATTRS.items():
            if optimizer_params is None or k not in optimizer_params:
                setattr(self, k, v)
            elif isinstance(v, dict):
                setattr(self, k, v)
                if optimizer_params[k] is not None:
                    for kk in optimizer_params[k]:
                        getattr(self, k)[kk] = optimizer_params[k][kk]
            else:
                setattr(self, k, optimizer_params[k])

        self.search_space = optimizer_params.get('search_space', None)
        self.cat_dims: List[int] = sorted(optimizer_params.get('cat_dims', []))
        self.integer_dims: List[int] = sorted(optimizer_params.get('integer_dims', []))
        self.casmo_continuous_dims = sorted(self.continuous_dims + self.integer_dims)

        # List. e.g. [2, 3, 4, 5] -- denotes there are 4 categorical variables, with numbers of categories
        # being 2, 3, 4, and 5 respectively.
        self.categories = []
        for i in self.cat_dims:
            num_categories = (self.bounds[1, i] - self.bounds[0, i]).detach().cpu().item() + 1
            self.categories.append(int(num_categories))
        
        self.casmo_attrs = {k: getattr(self, k) for k in CASMO_ATTRS}
        self.casmo = MixedOptimizer(
            config=self.categories, 
            lb=self.bounds[0, self.casmo_continuous_dims].detach().cpu().numpy(),
            ub=self.bounds[1, self.casmo_continuous_dims].detach().cpu().numpy(),
            # cont_dims=[i for i in range(len(self.casmo_continuous_dims))],
            # cat_dims=[i + len(self.casmo_continuous_dims) for i in range(len(self.cat_dims))],
            cont_dims=self.casmo_continuous_dims,
            cat_dims=self.cat_dims,
            seed=self.seed,
            
            n_init=self.num_init,
            verbose=True,
            min_cuda=1024,
            device=DEVICE,
            dtype=DTYPE,
            acq='thompson',
            kernel_type='mixed',

            **self.casmo_attrs
        )

        
        print_log('=' * 80, print_msg=True)
        print_log(f"[Casmopolitan] Initialized with following configurations", print_msg=True)
        for k, v in CASMO_ATTRS.items():
            print_log(f"\t{k}: {getattr(self, k)}", print_msg=True)

    @property
    def num_calls(self) -> int:
        return len(self.casmo.casmopolitan.X)
    
    @property
    def X(self) -> torch.Tensor:
        X_data = torch.Tensor(self.casmo.casmopolitan.X).to(DEVICE, DTYPE)
        X_data[:, self.discrete_dims] = round_by_bounds(X_data[:, self.discrete_dims], self.bounds[:, self.discrete_dims])
        return X_data
    
    @property
    def Y(self) -> torch.Tensor:
        Y_data = torch.Tensor(self.casmo.casmopolitan.fX).to(DEVICE, DTYPE)
        return -Y_data
    
    def get_original_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X, self.Y
    
    def save_tr_info(self):
        if get_log_idx() == 0:
            tr_save_path = os.path.join(get_result_dir(), f'{get_expr_name()}_tr.json')
        else:
            tr_save_path = os.path.join(get_result_dir(), f'{get_expr_name()}_{get_log_idx()}_tr.json')
        print_log(f'[Casmopolitan] save_tr_info: Saving the trust region data into {tr_save_path}', print_msg=True)
        with open(tr_save_path, 'w') as f:
            json.dump(self.casmo.tr_records, f, indent=4)

    # ----------------------------------------------------
    # Optimization Related
    def initial_sampling(self, num_init: int):
        pass

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor]:
        while self.num_calls < num_evals:
            # x_next_orig = self.casmo.suggest(self.batch_size)
            # x_next = x_next_orig.copy()
            # for i, idx in enumerate(self.casmo_continuous_dims):
            #     x_next[:, idx] = x_next_orig[:, i]
            # for i, idx in enumerate(self.cat_dims):
            #     x_next[:, idx] = x_next_orig[:, i + len(self.casmo_continuous_dims)]
            x_next = self.casmo.suggest(self.batch_size)

            y_next = self.obj_func(x_next)
            self.casmo.observe(x_next, y_next)

            print_log((
                    f"[Casmopolitan] {self.num_calls}) "
                    f"Best value: {self.casmo.casmopolitan.fX.min():.4f} | "
                    f"Cont TR Length: {self.casmo.casmopolitan.length:.3f} | "
                    f"Disc TR Length: {self.casmo.casmopolitan.length_discrete:3f} | "
                    f"num. failures: {self.casmo.casmopolitan.failcount}/{self.casmo.casmopolitan.failtol} | "
                    f"num. successes: {self.casmo.casmopolitan.succcount}/{self.casmo.casmopolitan.succtol}"
                ), print_msg=True
            )

        print_log(f'[Casmopolitan] Iterations where Casmopolitan restarts: {self.casmo.restart_iters}', print_msg=True)
        if self.save_tr: self.save_tr_info()
        return self.X, self.Y

    