import numpy as np
import torch
from typing import Optional, Union, List
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin, Branin, Hartmann

from .base_benchmark import BaseBenchmark, EffectiveBenchmark

from camtune.utils.logger import print_log
from camtune.utils.vars import DEVICE, DTYPE

# -----------------------------------------------
class AckleyBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        # Function settings
        kwargs['lb'], kwargs['ub'] = kwargs.get('lb', -5), kwargs.get('ub', 10)
        super().__init__(**kwargs)

        self._obj_func = Ackley(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()

class EffectiveAckleyBenchmark(EffectiveBenchmark):
    def __init__(self, effective_dim: int, **kwargs):
        kwargs['lb'], kwargs['ub'] = kwargs.get('lb', -5), kwargs.get('ub', 10)
        super().__init__(effective_dim, **kwargs)

        self._obj_func = Ackley(
            dim=self.effective_dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.effective_dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()


class ShiftedAckleyBenchmark(EffectiveBenchmark):
    def __init__(self, effective_dim: int, **kwargs):
        kwargs['lb'], kwargs['ub'] = kwargs.get('lb', -5), kwargs.get('ub', 10)
        super().__init__(effective_dim, **kwargs)

        self.offsets = torch.tensor(
            [-14.15468831, -17.35934204, 4.93227439, 30.68108305, 
             -20.94097318, -9.68946759, 11.23919487, 4.93101114,
             2.87604112, -31.0805155]
        ).to(dtype=DTYPE, device=DEVICE)
        self.bounds = self.bounds - self.offsets

        self._obj_func = Ackley(
            dim=self.effective_dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.effective_dim)]).to(dtype=DTYPE, device=DEVICE)
        self._obj_func = lambda x: self._obj_func(x[:, :self.effective_dim])

        self.__post_init__()


# -----------------------------------------------
class LevyBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        # Function settings
        super().__init__(**kwargs)
        self._obj_func = Levy(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()

class EffectiveLevyBenchmark(EffectiveBenchmark):
    def __init__(self, effective_dim: int, **kwargs):
        super().__init__(effective_dim, **kwargs)
        self._obj_func = Levy(
            dim=self.effective_dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.effective_dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()
    
# -----------------------------------------------
class RosenbrockBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        # Function settings
        super().__init__(**kwargs)

        self._obj_func = Rosenbrock(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()

class EffectiveRosenbrockBenchmark(EffectiveBenchmark):
    def __init__(self, effective_dim: int, **kwargs):
        super().__init__(effective_dim, **kwargs)

        self._obj_func = Rosenbrock(
            dim=self.effective_dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.effective_dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()

# -----------------------------------------------
class RastriginBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        kwargs['lb'], kwargs['ub'] = kwargs.get('lb', -5.12), kwargs.get('ub', 5.12)
        super().__init__(**kwargs)
        self._obj_func = Rastrigin(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=DTYPE, device=DEVICE)


        self.__post_init__()

class EffectiveRastriginBenchmark(EffectiveBenchmark):
    def __init__(self, effective_dim: int, **kwargs):
        kwargs['lb'], kwargs['ub'] = kwargs.get('lb', -5.12), kwargs.get('ub', 5.12)
        super().__init__(effective_dim, **kwargs)

        self._obj_func = Rastrigin(
            dim=self.effective_dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.effective_dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()

# ------------------------------------------------
class BraninBenchmark(EffectiveBenchmark):
    def __init__(self, effective_dim: int, **kwargs):
        kwargs['lb'], kwargs['ub'] = kwargs.get('lb', -5), kwargs.get('ub', 15)
        super().__init__(effective_dim, **kwargs)

        self._obj_func = Branin(
            negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.effective_dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()

# ------------------------------------------------
class HartmannBenchmark(EffectiveBenchmark):
    def __init__(self, effective_dim: int, **kwargs):
        super().__init__(effective_dim, **kwargs)

        self._obj_func = Hartmann(
            dim=self.effective_dim,
            negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.effective_dim)]).to(dtype=DTYPE, device=DEVICE)

        self.__post_init__()


# ------------------------------------------------
class LassoHardBenchmark(EffectiveBenchmark):
    """
    1000-D synthetic Lasso hard benchmark from https://github.com/ksehic/LassoBench .
    Effective dimensionality: 5% of input dimensionality.

    Args:
        noise_std: if > 0: noisy version with fixed SNR, noiseless version otherwise
        **kwargs:
    """

    def __init__(self, noise_std: Optional[float] = 0, **kwargs):
        from LassoBench import LassoBench

        if noise_std > 0:
            print_log(
                f"[LassoHardBenchmark] LassoBenchmark with noise_std {noise_std} chosen. Will use noisy version with snr ratio 10. The exact value of noise_std will be ignored."
            )
        self.benchmark: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_hard", noise=noise_std > 0
        )
        self.dim = self.benchmark.n_features

        self.effective_dims = np.arange(self.dim)[self.benchmark.w_true != 0]
        self.effective_dim = len(self.effective_dims)

        self._obj_func = self.compute
        super().__init__(
            dim=self.dim,
            effective_dim=self.effective_dim,
            lb=-1., ub=1.,
            require_wrapper=False,
        )
        
    
    def compute(self, x: torch.Tensor) -> float:
        x_np = x.cpu().numpy()
        result = self.benchmark.evaluate(x_np)
        return -result
