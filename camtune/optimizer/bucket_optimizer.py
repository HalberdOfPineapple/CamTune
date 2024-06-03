import torch
import math
from typing import Callable, List, Dict, Any, Optional, Union, Tuple
from botorch.utils.transforms import unnormalize

from .base_optimizer import BaseOptimizer
from camtune.optimizer.optim_utils import generate_random_discrete, round_by_bounds
from camtune.utils import DTYPE, DEVICE

class BucketOptimizer(BaseOptimizer):
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
        raise NotImplementedError(f"BucketOptimizer does not support initial_sampling method.")
    
    def get_original_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X, self.Y
    
    def init_cand_values(self):
        x_range = self.bounds[1][0] - self.bounds[0][0]
        y_range = self.bounds[1][1] - self.bounds[0][1]

        x_step = x_range / self.num_buckets
        y_step = y_range / self.num_buckets

        cand_values = []
        for i in range(self.num_buckets):
            for j in range(self.num_buckets):
                cand_values.append([self.bounds[0][0] + x_step * i, self.bounds[0][1] + y_step * j]) # shape (num_buckets^2, 2)
    
        cand_values = torch.tensor(cand_values, dtype=DTYPE, device=DEVICE)
        return cand_values

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_evals: number of evaluations
        """ 
        self.num_buckets = int(math.sqrt(num_evals))
        cand_values = self.init_cand_values()

        for x in cand_values:
            y = torch.tensor(self.obj_func(x), dtype=DTYPE, device=DEVICE).reshape(1, 1)
            
            self.X = torch.cat([self.X, x.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, y], dim=0)

        return self.X, self.Y