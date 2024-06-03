import torch
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Union, Tuple
from botorch.utils.transforms import unnormalize

from .base_optimizer import BaseOptimizer
from camtune.optimizer.optim_utils import generate_random_discrete, round_by_bounds
from camtune.utils import DTYPE, DEVICE

class UniformOptimizer(BaseOptimizer):
    def __init__(
        self,
        bounds: torch.Tensor,
        obj_func: Callable,
        batch_size: int = 1,
        seed: int = 0,
        discrete_dims: List[int] = None,
        optimizer_params: Dict[str, Any] = None,
        init_design: str = 'LHS',
        use_default: bool = True,
        default_conf: Optional[torch.Tensor] = None,
    ):
        init_design = 'LHS'
        super().__init__(bounds, obj_func, 
                         batch_size=batch_size,
                         seed=seed, 
                         discrete_dims=discrete_dims, 
                         optimizer_params=optimizer_params,
                         init_design=init_design,
                         use_default=use_default,
                         default_conf = default_conf,
                        )

        self.X = torch.empty(0, self.dimension, dtype=DTYPE, device=DEVICE)
        self.Y = torch.empty(0, 1, dtype=DTYPE, device=DEVICE)
    
    def initial_sampling(self, num_init: int):
        raise NotImplementedError(f"UniformOptimizer does not support initial_sampling method.")
    
    def get_original_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X, self.Y

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_evals: number of evaluations
        """ 
        X_sampled = np.random.uniform(0, 1, (num_evals, self.dimension))
        X_sampled = torch.tensor(X_sampled, dtype=DTYPE, device=DEVICE)
        
        X_sampled = unnormalize(X_sampled, self.bounds) if self.tr_bounds is None else unnormalize(X_sampled, self.tr_bounds)
        X_sampled[:, self.discrete_dims] = round_by_bounds(X_sampled[:, self.discrete_dims], self.bounds[:, self.discrete_dims])
        if self.use_default:
            X_sampled = torch.cat([self.default_conf, X_sampled], dim=0)
            X_sampled = X_sampled[:num_evals]

        for x in X_sampled:
            y = torch.tensor(self.obj_func(x), dtype=DTYPE, device=DEVICE).reshape(1, 1)
            
            self.X = torch.cat([self.X, x.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, y], dim=0)

        return self.X, self.Y