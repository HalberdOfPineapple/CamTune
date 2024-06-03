import torch
from typing import List

from .base_sampler import BaseSampler
from .sobol_sampler import SobolSampler
from .lhs_sampler import LHSSampler

INIT_DESIGN_MAP = {
    "LHS": LHSSampler,
    "SOBOL": SobolSampler,
}

def build_init_design(
    init_design: str, bounds: torch.Tensor, seed:int=0, discrete_dims: List[int]=None
) -> BaseSampler:
    init_design = init_design.upper()
    if init_design in INIT_DESIGN_MAP:
        return INIT_DESIGN_MAP[init_design](bounds=bounds, seed=seed, discrete_dims=discrete_dims)
    else:
        raise ValueError(f"[Sampler] Undefined initial design: {init_design}")