import torch
from typing import Callable, List, Dict, Any, Optional, Union, Tuple
from botorch.utils.transforms import unnormalize

from .base_optimizer import BaseOptimizer
from camtune.optimizer.optim_utils import generate_random_discrete, round_by_bounds
from camtune.utils import DTYPE, DEVICE

class ConstantOptimizer(BaseOptimizer):
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
        super().__init__(bounds, obj_func, 
                         batch_size=batch_size,
                         seed=seed, 
                         discrete_dims=discrete_dims, 
                         optimizer_params=optimizer_params,
                         init_design=init_design,
                         use_default=use_default,
                         default_conf=default_conf)
    
    def initial_sampling(self, num_init: int):
        raise NotImplementedError(f"ConstantOptimizer does not support initial_sampling method.")

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_evals: number of evaluations
            X_init: (num_init, dim)
            Y_init: (num_init, 1)
        """ 
        self.X = torch.empty(0, self.dimension, dtype=DTYPE, device=DEVICE)
        self.Y = torch.empty(0, 1, dtype=DTYPE, device=DEVICE)

        for _ in range(num_evals):
            x = torch.clone(self.default_conf).to(DEVICE, DTYPE).resize(self.dimension)
            y = torch.tensor(
                [self.obj_func(x)], dtype=DTYPE, device=DEVICE,
            ).unsqueeze(-1)

            self.X = torch.cat([self.X, x.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, y], dim=0)

        return self.X, self.Y