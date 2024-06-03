import torch
import numpy as np
from typing import Callable
from ConfigSpace import ConfigurationSpace, Configuration
from botorch.utils.transforms import unnormalize
from typing import List, Callable, Optional, Union, Tuple

from camtune.utils import print_log
from camtune.optimizer import build_optimizer, BaseOptimizer
from camtune.search_space import SearchSpace

class Tuner:
    def __init__(self, 
      expr_name: str, 
      args: dict, 
      obj_func: Callable,
      bounds: torch.Tensor, # (2, D)
      discrete_dims: Optional[List[int]] = [],
      cat_dims: Optional[List[int]] = [],
      integer_dims: Optional[List[int]] = [],
      default_conf: Optional[torch.Tensor] = None,
      definition_path: Optional[str] = None,
      search_space: Optional[SearchSpace] = None,
      tr_bounds: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            expr_name: experiment name
            args: dict of tuner parameters
            obj_func: objective function to be optimized. Note that the function takes tensor as the input 
            bounds: (2, D)
            discrete_dims: list of indices of discrete dimensions
        """
        self.expr_name = expr_name
        self.args = args

        # Directly assume obj_func is to be maximized. 
        # If for minimization, negate it before passing it in.
        self.obj_func = obj_func

        self.seed = self.args['seed']
        self.bounds = bounds
        self.init_design = self.args.get('init_design', 'LHS')
        self.tr_bounds = tr_bounds

        self.dimension = self.bounds.shape[1]
        self.discrete_dims = discrete_dims
        self.continuous_dims = [i for i in range(self.dimension) if i not in discrete_dims]
        self.cat_dims = cat_dims
        self.integer_dims = integer_dims

        self.num_evals = self.args['num_evals']
        self.batch_size = self.args.get('batch_size', 1)

        self.use_default = self.args.get('use_default', True)
        self.default_conf_tensor = default_conf

        self.perf_name = self.args.get('perf_name', 'func_val'), 
        self.perf_unit = self.args.get('perf_unit', 'null')
        self.definition_path = definition_path

        optimizer_params = self.args.get('optimizer_params', {})
        if self.args['optimizer'] == 'llama':
            optimizer_params['perf_name'] = f'{self.perf_name}_{self.perf_unit}'
            optimizer_params['definition_path'] = self.definition_path
        elif 'casmo' in self.args['optimizer']:
            if self.cat_dims == []:
                self.cat_dims = self.discrete_dims
            optimizer_params['cat_dims'] = self.cat_dims
            optimizer_params['integer_dims'] = self.integer_dims
            optimizer_params['search_space'] = search_space
        optimizer_params['tr_bounds'] = self.tr_bounds

        # bounds: torch.Tensor,
        # obj_func: Callable,
        # batch_size: int = 1,
        # seed: int = 0,
        # discrete_dims: List[int] = [],
        # optimizer_params: Dict[str, Any] = None,
        # init_design: str = 'LHS',
        # use_default: bool = True,
        # default_conf: Optional[torch.Tensor] = None,
        np.random.seed(self.seed)
        self.optimizer: BaseOptimizer = build_optimizer(
            self.args['optimizer'],
            bounds=self.bounds,
            batch_size=self.batch_size,
            obj_func=self.obj_func,
            seed=self.seed,
            discrete_dims=discrete_dims,
            optimizer_params=optimizer_params,
            init_design=self.init_design,
            use_default=self.use_default,
            default_conf=self.default_conf_tensor,
        )

    def tune(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            result_X: (num_evals, dim)
            result_Y: (num_evals, 1)
        """
        print_log(f'[Tuner] Start optimization using {self.args["optimizer"]} for {self.num_evals} iterations (internal initialization)', print_msg=True)
        result_X, result_Y = self.optimizer.optimize(self.num_evals)
        
        return result_X, result_Y
