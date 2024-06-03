import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional, Union, Tuple

from .sampler import build_init_design
from .mcts_copilot import MCTSCopilot
from camtune.utils import DTYPE, DEVICE, print_log

class BaseOptimizer(ABC):
    def __init__(
        self,
        bounds: torch.Tensor,
        obj_func: Callable,
        batch_size: int = 1,
        seed: int = 0,
        discrete_dims: List[int] = [],
        optimizer_params: Dict[str, Any] = None,
        init_design: str = 'LHS',
        use_default: bool = True,
        default_conf: Optional[torch.Tensor] = None,
        manual_seed: bool = True,
    ):
        self.cls_name = str(self.__class__.__name__)
        
        self.seed = seed
        if manual_seed:
            torch.manual_seed(self.seed)

        self.obj_func = obj_func
        self.batch_size = batch_size

        self.bounds = bounds
        self.dtype, self.device = DTYPE, DEVICE
        self.dimension = self.bounds.shape[1]
        
        self.discrete_dims = discrete_dims
        self.continuous_dims = [i for i in range(self.dimension) if i not in discrete_dims]

        self.init_design = init_design  
        self.sampler = build_init_design(init_design, bounds, seed, discrete_dims)

        optimizer_params = {} if optimizer_params is None else optimizer_params
        self.optimizer_params = optimizer_params
        self.use_default = use_default
        self.default_conf = default_conf

        self.use_copilot = optimizer_params.get('use_copilot', False)
        self.copilot_params = optimizer_params.get('copilot_params', {})
        self.tr_bounds = optimizer_params.get('tr_bounds', None)

    @abstractmethod
    def initial_sampling(self, num_init: int):
        raise NotImplementedError

    @abstractmethod
    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

