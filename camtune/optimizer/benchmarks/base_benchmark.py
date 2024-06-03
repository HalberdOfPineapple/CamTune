import torch
from abc import ABC, abstractmethod
from typing import Callable
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin

from camtune.utils import DEVICE, DTYPE, print_log

PROPERTY_SET = {
    'seed', 'Cp', 'leaf_size', 'node_selection_type', 'initial_sampling_method',
    'bounds', 'num_init', 'obj_func', 'optimizer_type', 'optimizer_params',
    'classifier_type', 'classifier_params',
}

BENCH_INFO_FLDS = {'dim', 'negate', 'bounds'}

class BaseBenchmark(ABC):
    def __init__(self, **kwargs):
        self.dim: int = kwargs.get('dim', 20)

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -10.)
        self.ub: float = kwargs.get('ub', 10.)
        self.bounds: torch.Tensor = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim]).to(dtype=DTYPE, device=DEVICE)
        self.discrete_dims: list = kwargs.get('discrete_dims', [])

        self.mcts_params: dict = kwargs.get('mcts_params', None)

    def __post_init__(self):
        cls_name = str(self.__class__.__name__)
        print_log(f"[{cls_name}] Benchmark Initialized with the following configurations:", print_msg=True)
        for field in BENCH_INFO_FLDS:
            print_log(f"\t{field}:\t{getattr(self, field)}", print_msg=True)
        
        if self.mcts_params is not None:
            print_log(f"[{cls_name}] MCTS Params:", print_msg=True)
            for k, v in self.mcts_params.items():
                print_log(f"\t\t{k}:\t{v}", print_msg=True)

    @property
    def obj_func(self) -> Callable:
        return self._obj_func
    

class EffectiveFuncWrapper(Callable):
    def __init__(self, obj_func: Callable, effective_dim: int):
        self._obj_func = obj_func
        self.effective_dim = effective_dim

    def __call__(self, x: torch.Tensor):
        if x.ndim == 1:
            return self._obj_func(x[:self.effective_dim])
        else:
            return self._obj_func(x[:, :self.effective_dim])


class EffectiveBenchmark(BaseBenchmark):
    def __init__(self, effective_dim: int, **kwargs):
        self.dim: int = kwargs.get('dim', 20)
        self.effective_dim: int = effective_dim
        if self.effective_dim > self.dim:
            raise ValueError(f"[EffectiveBenchmark] Effective dimension ({self.effective_dim}) cannot be greater than the original dimension ({self.dim})")

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -10.)
        self.ub: float = kwargs.get('ub', 10.)
        self.bounds: torch.Tensor = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim]).to(dtype=DTYPE, device=DEVICE)
        self.discrete_dims: list = kwargs.get('discrete_dims', [])

        self.mcts_params: dict = kwargs.get('mcts_params', None)

        self._obj_func: Callable
        self.require_wrapper = kwargs.get('require_wrapper', True)

    def __post_init__(self):
        cls_name = str(self.__class__.__name__)
        print_log(f"[{cls_name}] Benchmark Initialized with the following configurations:", print_msg=True)
        print_log(f"\tEffective Dimension:\t{self.effective_dim}", print_msg=True)
        for field in BENCH_INFO_FLDS:
            print_log(f"\t{field}:\t{getattr(self, field)}", print_msg=True)
        
        if self.mcts_params is not None:
            print_log(f"[{cls_name}] MCTS Params:", print_msg=True)
            for k, v in self.mcts_params.items():
                print_log(f"\t\t{k}:\t{v}", print_msg=True)
        if self.require_wrapper:
            self._obj_func = EffectiveFuncWrapper(self._obj_func, self.effective_dim)

    @property
    def obj_func(self) -> Callable:
        return self._obj_func