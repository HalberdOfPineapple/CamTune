import os
import math
import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List

import botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine

from camtune.utils import print_log

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

@dataclass 
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 5
    success_counter: int = 0
    success_tolerance: int = 5 # 10  # paper's version: 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    update_factor: float = 2.

    def __post_init__(self):
        attrs = ['length', 'length_min', 'length_max', 'failure_tolerance', 'success_tolerance']
        print_log("[TurboState] Initialized with the following attributes:", print_msg=True)
        for attr in attrs:
            print_log(f"[TurboState]\t{attr}:\t{getattr(self, attr)}", print_msg=True)

    def to_dict(self):
        return {
            "length": self.length,
            "length_min": self.length_min,
            "length_max": self.length_max,
            "failure_counter": self.failure_counter,
            "failure_tolerance": self.failure_tolerance,
            "success_counter": self.success_counter,
            "success_tolerance": self.success_tolerance,
        }

def update_state(state: TurboState, Y_next: torch.Tensor, step_length: float=0):
    # Note that `tensor(bool)`` can directly be used for condition eval
    if max(Y_next) > state.best_value + step_length: 
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    
    if state.success_counter == state.success_tolerance:
        state.length = min(state.update_factor * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= state.update_factor
        state.failure_counter = 0
    
    state.best_value = max(state.best_value, max(Y_next).item())

    # "Whenever L falls below a given minimum threshold L_min, we discard 
    #  the respective TR and initialize a new one with side length L_init"
    if state.length < state.length_min:
        state.restart_triggered = True

    return state
