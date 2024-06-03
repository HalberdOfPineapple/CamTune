import torch
from dataclasses import dataclass
from typing import List

from camtune.utils import print_log


class MCTSLocalControl:
    def __init__(
            self, 
            global_best_y: float, 
            jump_ratio: float, 
            jump_tolerance: int,
            init_mode: bool = True,  
            real_mode: bool = False,
            attach_leaf: bool = False,
            sampling_mode: bool = False,
    ):
        self.global_best_y = global_best_y

        # samples with performance below this ratio of the best value will be considered as low-performant sample
        # the greater this value is, the more samples will be considered as low-performant (more stringent)
        # when it is 0, no sample will be considered as low-performant (mechanism disabled)
        self.jump_ratio = jump_ratio
        self.jump_tolerance = jump_tolerance

        self.real_mode = real_mode
        self.attach_leaf = attach_leaf
        self.init_mode = init_mode
        self.sampling_mode = sampling_mode

        self.low_perf_count = 0

        self.post_init()
    
    def post_init(self):
        print_log("=" * 50, print_msg=True)
        if self.jump_ratio == 0:
            print_log(f"[MCTSLocalControl] Local jump mechanism is disabled", print_msg=True)
        else:
            print_log(f"[MCTSLocalControl] Local jump mechanism is enabled with jump ratio = {self.jump_ratio} and tolerance = {self.jump_tolerance}", print_msg=True)

        print_log(f"[MCTSLocalControl] Real mode (path-confined sampling) is {'enabled' if self.real_mode else 'disabled'}", print_msg=True)
        print_log(f"[MCTSLocalControl] Attach leaf mechanism is {'enabled' if self.attach_leaf else 'disabled'}", print_msg=True)
        print_log(f"[MCTSLocalControl] Init-guidance by path-region is {'enabled' if self.init_mode else 'disabled'}", print_msg=True)
        print_log(f"[MCTSLocalControl] Sampling-guided mode is {'enabled' if self.sampling_mode else 'disabled'}", print_msg=True)
        

    def check_jump(self, Y_next: torch.Tensor, failure_tolerance: int) -> bool:
        if self.jump_ratio == 0: return False

        for y in Y_next:
            # Disable this mechanism if a better sample is found
            if y > self.global_best_y: 
                self.jump_ratio = 0 
                return False

            # Check if the sample is a low-performance partition
            if y > 0 and y < self.global_best_y * self.jump_ratio:
                self.low_perf_count += 1
            elif y < 0 and y < self.global_best_y / self.jump_ratio: 
                # e.g. best = -25, best / 0.2 = -125, y = -150 => low perf
                self.low_perf_count += 1
            else:
                self.low_perf_count = 0
    
            if self.low_perf_count >= self.jump_tolerance * failure_tolerance:
                print_log(f"[MCTSLocalControl] Low-performance partition detected by successive failure, restart", print_msg=True)
                return True
        
        return False