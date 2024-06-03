import os
import math
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List, Dict, Any

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration, ConfigurationSpace

import botorch
botorch.settings.debug(True)
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, standardize, normalize
from torch.quasirandom import SobolEngine

from .base_optimizer import BaseOptimizer
from .optim_utils import round_by_bounds
from .llama_utils import get_smac_optimizer, ConfigSpaceGenerator, LHDesignWithBiasedSampling
from .llama_utils.spaces.common import finalize_conf, unfinalize_conf

from camtune.utils import (print_log, get_expr_name, get_result_dir, get_log_idx, DEVICE, DTYPE)


# SMAC optimizer requires:
# llama_config:
#    seed num_evals 
#    optimizer: 
#        init_rand_samples model_type rand_percentage n_estimators
# input_space: ConfigSpace
# obj_func: Configuration -> float

LLAMA_CONFIG = {
    'adapter_alias': 'hesbo',
    'le_low_dim': 16,

    'optimizer': {
        'model_type': 'rf',
        'init_rand_samples': 10,
        'rand_percentage': 0.1,
        'n_estimators': 100,
    }
}

def config_to_tensor(config: Configuration, search_space: ConfigSpaceGenerator) -> torch.Tensor:
    var_to_dim: Dict[str, int] = search_space.knob_to_idx
    config_tensor = torch.empty(1, len(var_to_dim), dtype=DTYPE, device=DEVICE)
    for k, v in dict(config).items():
        var_idx = var_to_dim[k]
        if search_space.knobs[var_idx]['type'] == 'enum':
            v = search_space.enum_val2idx[var_idx][v]
        config_tensor[0, var_to_dim[k]] = v
    return config_tensor

class LlamaFunctionWrapper:
    def __init__(self, obj_func: Callable, search_space: ConfigSpaceGenerator):
        self.eval_counter = 0
        self.best_val = -math.inf

        self.obj_func = obj_func
        self.search_space = search_space

    def __call__(self, config: Configuration) -> float:
        config = self.search_space.unproject_input_point(config)
        config = self.search_space.finalize_conf(config)

        config_tensor = config_to_tensor(config, self.search_space)
        eval_val = self.obj_func(config_tensor)
        if isinstance(eval_val, torch.Tensor):
            eval_val = eval_val.detach().cpu().item()

        if eval_val > self.best_val: self.best_val = eval_val
        self.eval_counter += 1

        print_log(f'[LlamaOptimizer] [Iteration {self.eval_counter}] function_val: {eval_val:.3f} | best value: {self.best_val:.3f}', print_msg=True)
        return -eval_val

class NegFuncWrapper:
    def __init__(self, obj_func: Callable, search_space: ConfigSpaceGenerator):
        self.eval_counter = 0
        self.best_val = -math.inf
        
        self.obj_func = obj_func
        self.search_space = search_space

    def __call__(self, config: Configuration) -> float:
        config = self.search_space.unproject_input_point(config)
        config = self.search_space.finalize_conf(config)

        eval_val = self.obj_func(config)
        if isinstance(eval_val, torch.Tensor):
            eval_val = eval_val.detach().cpu().item()

        if eval_val > self.best_val: self.best_val = eval_val
        self.eval_counter += 1
        
        print_log(f'[LlamaOptimizer] [Iteration {self.eval_counter}] function_val: {eval_val:.3f} | best value: {self.best_val:.3f}', print_msg=True)
        return -eval_val


class LlamaOptimizer(BaseOptimizer):
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

        for k, v in LLAMA_CONFIG.items():
            if optimizer_params is None or k not in optimizer_params:
                setattr(self, k, v)
            elif isinstance(v, dict):
                setattr(self, k, v)
                if optimizer_params[k] is not None:
                    for kk in optimizer_params[k]:
                        getattr(self, k)[kk] = optimizer_params[k][kk]
            else:
                setattr(self, k, optimizer_params[k])

        if optimizer_params.get('definition_path', None) is not None:
            space_dict = {
                'definition': optimizer_params['definition_path'],
                'ignore': 'postgres-none',
                'adapter_alias': self.adapter_alias,
                'le_low_dim': self.le_low_dim,
                'target_metric': optimizer_params.get('perf_name', 'function_val'),
                'bias_prob_sv': optimizer_params.get('bias_prob_sv', None),
                'quantization_factor': optimizer_params.get('quantization_factor', None),
            }
            self.config_space_gen: ConfigSpaceGenerator = ConfigSpaceGenerator.from_config(space_dict, seed=self.seed)
        else:
            self.config_space_gen: ConfigSpaceGenerator = self.build_config_space_generator()

        self.input_space: ConfigurationSpace = self.config_space_gen.generate_input_space(
                                                self.seed, ignore_extra_knobs=None)
        self.var_to_dim: dict = self.config_space_gen.knob_to_idx

        if hasattr(self.obj_func, 'forward_config'):
            self.obj_func = NegFuncWrapper(self.obj_func.forward_config, self.config_space_gen)
        else:
            self.obj_func = LlamaFunctionWrapper(self.obj_func, self.config_space_gen, self.var_to_dim)

        self.llama_config = {key: getattr(self, key) for key in LLAMA_CONFIG}
        self.llama_config['seed'] = self.seed

        print_log('=' * 80, print_msg=True)
        print_log(f"[LlamaOptimizer] Initialized with following configurations", print_msg=True)
        for k in self.llama_config.keys():
            print_log(f"\t{k}: {getattr(self, k)}", print_msg=True)

    def build_config_space_generator(self):
        definition = {}
        for dim in range(0, self.dimension):
            min_val, max_val = self.bounds[0, dim].item(), self.bounds[1, dim].item()
            default_val = np.random.uniform(min_val, max_val)
            info = {
                'name': str(dim),
                'type': 'integer' if dim in self.discrete_dims else 'real',
                'min': min_val,
                'max': max_val,
                'default': default_val,
            }
            definition[str(dim)] = info

        config_space_gen = ConfigSpaceGenerator(
            definition, 
            target_metric='function_val',
            adapter_alias=self.adapter_alias,
            le_low_dim=self.le_low_dim,
            seed=self.seed,
            finalize_conf_func=finalize_conf,
            unfinalize_conf_func=unfinalize_conf,
        )
        return config_space_gen

    @property
    def num_calls(self) -> int:
        return len(self._X)
    
    @property
    def X(self) -> torch.Tensor:
        return self._X
    
    @property
    def Y(self) -> torch.Tensor:
        return self._Y
    
    def get_original_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._X, self._Y
    
    def save_tr_info(self):
        if get_log_idx() == 0:
            tr_save_path = os.path.join(get_result_dir(), f'{get_expr_name()}_tr.json')
        else:
            tr_save_path = os.path.join(get_result_dir(), f'{get_expr_name()}_{get_log_idx()}_tr.json')
        print_log(f'[LlamaOptimizer] save_tr_info: Saving the trust region data into {tr_save_path}', print_msg=True)
        with open(tr_save_path, 'w') as f:
            json.dump(self.tr_records, f)

    # ----------------------------------------------------
    # Optimization Related
    def initial_sampling(self, num_init: int):
        pass
    
    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._X = torch.empty(0, self.dimension, dtype=DTYPE, device=DEVICE)
        self._Y = torch.empty(0, 1, dtype=DTYPE, device=DEVICE)

        self.llama_config['num_evals'] = num_evals
        self.optimizer = get_smac_optimizer(
            llama_config=self.llama_config,
            spaces=self.config_space_gen,
            obj_func=self.obj_func,
        )

        self.optimizer.optimize()

        run_history = self.optimizer.runhistory
        for run_key, run_value in run_history.data.items():
            config_id = run_key.config_id

            config = dict(run_history.ids_config[config_id])
            config = self.config_space_gen.unproject_input_point(config)
            config = self.config_space_gen.finalize_conf(config)
            config_tensor = config_to_tensor(config, self.config_space_gen)
            self._X = torch.cat([self._X, config_tensor], dim=0)

            metric_val = -float(run_value.cost)
            self._Y = torch.cat([self._Y, torch.tensor([[metric_val]], dtype=DTYPE, device=DEVICE)], dim=0)
        
        return self.X, self.Y