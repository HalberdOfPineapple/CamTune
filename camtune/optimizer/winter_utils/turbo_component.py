import os
import math
import torch
import numpy as np
from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union, List, Dict

import botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from .node import Node
from .local_control import MCTSLocalControl

from camtune.utils import print_log, DEVICE, DTYPE


class BaseOptimizer:
    @abstractmethod
    def optimize(self, X_in_region: torch.Tensor, Y_in_region: torch.Tensor, num_evals: int, path: List[Node]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class PartitionGP(SingleTaskGP):
    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor, cand_to_score: Callable, **kwargs):
        super().__init__(train_X, train_Y, **kwargs)
        self.cand_to_score = cand_to_score

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        cand_score = self.cand_to_score(x)
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultivariateNormal(mean_x, covar_x) * cand_score

# ================================================================
# TuRBO
# ================================================================
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 5 # float("nan")  # to be post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # paper's version: 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

def update_state(state: TurboState, Y_next: torch.Tensor):
    
    # Note that `tensor(bool)`` can directly be used for condition eval
    if max(Y_next) > state.best_value: 
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    
    state.best_value = max(state.best_value, max(Y_next).item())

    # "Whenever L falls below a given minimum threshold L_min, we discard 
    #  the respective TR and initialize a new one with side length L_init"
    if state.length < state.length_min:
        state.restart_triggered = True

    return state

ACQFS = {"ts", "ei"}
DEFAULT_TURBO_PARAMS = {
    'acqf': 'ts',
    'num_restarts': 10,
    'raw_samples': 512,
    'sampling_method': 'normal',
    'init_bounding_box_length': 0.0005,
    'max_cholesky_size': float('inf')
}
class TuRBO(BaseOptimizer):
    def __init__(self, 
        obj_func: Callable,
        bounds: torch.Tensor, # (2, dimension)
        num_init: int,
        seed: int, 
        optimizer_params: Dict
    ):
        self.seed = seed
        self.obj_func = obj_func
    
        self.num_init = num_init
        
        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.batch_size = optimizer_params['batch_size']
        for k, v in DEFAULT_TURBO_PARAMS.items():
            setattr(self, k, optimizer_params.get(k, v))

        if not self.acqf in ACQFS:
            raise ValueError(f"Acquisition function {self.acqf} not supported")
        self.n_candidates = min(5000, max(2000, 200 * self.dimension)) # if not SMOKE_TEST else 4

        # -------------------------------------------------
        # TuRBO State Params
        self.turbo_state_params = {
            'dim': self.dimension,
            'batch_size': self.batch_size,
        }
        if 'turbo_state_params' in optimizer_params:
            for k, v in optimizer_params['turbo_state_params'].items():
                self.turbo_state_params[k] = eval(v) if isinstance(v, str) else v

        self.dtype = bounds.dtype
        self.device = bounds.device

        self.num_calls = 0

        print_log(f'[TuRBOComponents] Intiailized with the following configurations:', print_msg=True)
        for k, v in DEFAULT_TURBO_PARAMS.items():
            print_log(f"[TuRBOComponents]\t{k}: {getattr(self, k)}", print_msg=True)
        for k, v in self.turbo_state_params.items():
            print_log(f"[TuRBOComponents]\tTuRBO State.{k}: {v}", print_msg=True)

    
    def initial_sampling(self, num_init: int):
        sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
        X_init: torch.Tensor = sobol.draw(num_init).to(dtype=self.dtype, device=self.device)
        Y_init = torch.tensor(
            [self.obj_func(x) for x in X_init], dtype=DTYPE, device=DEVICE,
        ).unsqueeze(-1)

        return X_init, Y_init
        
    def gen_candidates(
            self, 
            x_center: torch.Tensor, 
            tr_lbs: torch.Tensor, 
            tr_ubs: torch.Tensor, 
        ) -> torch.Tensor:
        sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
        pert = sobol.draw(self.n_candidates).to(dtype=DTYPE, device=DEVICE)
        pert = tr_lbs + (tr_ubs - tr_lbs) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dimension, 1.0)
        mask = torch.rand(self.n_candidates, self.dimension, dtype=DTYPE, device=DEVICE) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, self.dimension - 1, size=(len(ind),), device=DEVICE)] = 1

        # Create candidate points from the perturbations and the mask
        X_cands = x_center.expand(self.n_candidates, self.dimension).clone()
        X_cands[mask] = pert[mask]

        return X_cands

    def generate_batch(self, 
        state: TurboState,
        model: botorch.models.model.Model, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        path: Optional[List[Node]] = None,
    ) -> torch.Tensor:
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        x_center = X[Y.argmax(), :].clone()

        # Length scales for all dimensions
        # weights - (dim, )
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # Clamps all elements in input into the range [min, max]
        # tr_lbs, tr_ubs - (1, dim)
        tr_lbs = torch.clamp(x_center - state.length / 2 * weights, 0.0, 1.0)
        tr_ubs = torch.clamp(x_center + state.length / 2 * weights, 0.0, 1.0)

        if self.acqf == "ts":
            X_cands = self.gen_candidates(x_center, tr_lbs, tr_ubs)
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = thompson_sampling(X_cands, num_samples=self.batch_size)

                # Apply partition-based weighting: re-weight samples based on their leaf score
                if self.local_control.sampling_mode:
                    weights = torch.tensor([Node.get_partition_score(x) for x in X_next], dtype=DTYPE, device=DEVICE)
                    weighted_samples = torch.multinomial(weights, self.batch_size, replacement=False)

                    X_next = X_next[weighted_samples]
        elif self.acqf == "ei":
            ei = qExpectedImprovement(model=model, best_f=Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lbs, tr_ubs]),
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        else:
            raise ValueError(f"[TuRBOComponent] Acquisition function {self.acqf} not supported")
        
        return X_next
    
    def generate_samples_in_region(
        self,
        num_samples: int,
        path: List[Node],
        X_in_region: torch.Tensor,
        Y_in_region: torch.Tensor,
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        region_center: torch.Tensor = X_in_region[torch.argmax(Y_in_region), :].clone()
        X_init = Node.generate_samples_in_region(
            dimension=self.dimension,
            num_samples=num_samples,
            path=path,
            init_bounding_box_length=self.init_bounding_box_length,
            region_center=region_center,
            seed=self.seed if seed is None else seed,
        )

        return X_init

    def optimize(
            self, 
            X_in_region: torch.Tensor, 
            Y_in_region: torch.Tensor, 
            num_evals: int, 
            path: List[Node],
            local_control: MCTSLocalControl,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.local_control = local_control

        
        # -------------------------------------------------
        num_init = min(self.num_init, num_evals)
        if local_control.init_mode:
            X_init = self.generate_samples_in_region(num_init, path, X_in_region, Y_in_region)
            Y_init = torch.tensor(
                [self.obj_func(x) for x in X_init], dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)
        else:
            X_init, Y_init = self.initial_sampling(num_init)
        self.num_calls += num_init
        print(f"[TuRBOComponent] Start local modeling with {num_init} data points")
        
        # -------------------------------------------------
        state = TurboState(**self.turbo_state_params)
        
        # -------------------------------------------------
        X_sampled = X_init # torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
        Y_sampled = Y_init # torch.empty((0, 1), dtype=self.dtype, device=self.device)
        while not state.restart_triggered and self.num_calls < num_evals:
            train_X = X_sampled if not local_control.attach_leaf else torch.cat((X_sampled, X_in_region), dim=0)
            train_Y = standardize(Y_sampled) if not local_control.attach_leaf else standardize(torch.cat((Y_sampled, Y_in_region), dim=0))

            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5, ard_num_dims=self.dimension, lengthscale_constraint=Interval(0.005, 4.0),
                ),
            )

            model = SingleTaskGP(
                train_X, train_Y, covar_module=covar_module, likelihood=likelihood
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                try:
                    fit_gpytorch_mll(mll)
                except Exception as ex:
                    print_log(f"[TuRBOComponent] {ex} happens when fitting model, use Adam.")
                    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                    for _ in range(100):
                        optimizer.zero_grad()
                        output = model(train_X)
                        loss = -mll(output, train_Y.flatten())
                        loss.backward()
                        optimizer.step()
                X_next = self.generate_batch(
                    state=state, model=model, 
                    X=train_X, Y=train_Y,
                    path=path if local_control.real_mode else None,
                )
            X_next = X_next[:min(self.batch_size, num_evals - self.num_calls)]
            Y_next = torch.tensor(
                [self.obj_func(x) for x in X_next], dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)

            X_sampled = torch.cat((X_sampled, X_next), dim=0)
            Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)

            # Statistics Update
            self.num_calls += len(X_next)
            state = update_state(state, Y_next)

            print_log(
                ("[TuRBOComponent] "
                f"[{len(X_sampled)} in node {path[-1].id}] "
                f"Best value: {state.best_value:.3f} | TR length: {state.length:.3f} | "
                f"num. failures: {state.failure_counter}/{state.failure_tolerance} | "
                f"num. successes: {state.success_counter}/{state.success_tolerance}"),
                print_msg=True,
            )

            if local_control is not None and local_control.check_jump(Y_next, state.failure_tolerance):
                state.restart_triggered = True
                break

        if state.restart_triggered:
            print_log(
                f"[TuRBOComponent] Local Modelling converges (TR length below threshold) after {self.num_calls} evaluations",
                print_msg=True,
            )

        return X_sampled, Y_sampled

OPTIMIZER_MAP = {
    'turbo': TuRBO,
}