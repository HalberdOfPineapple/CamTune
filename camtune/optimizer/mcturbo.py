import os
import math
import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List, Dict, Any

import botorch
botorch.settings.debug(True)
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, standardize, normalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from camtune.utils import print_log, DTYPE, DEVICE
from .base_optimizer import BaseOptimizer
from .optim_utils import round_by_bounds, ObjFuncWrapper
from .turbo_utils import TurboState, update_state
from .winter_utils import MCTSLocalControl, Node
from .mcturbo_utils import MCTurboState

MCTURBO_ATTRS = {
    'num_restarts': 10,
    'raw_samples': 512,
    'n_candidates': 2400, # 5000,
    'max_cholesky_size': float("inf"),
    'acqf': "ts",
    "init_bounding_box_length": 0.0001,
    "training_steps": 100,
}

class MCTuRBO(BaseOptimizer):
    def __init__(
        self, 
        bounds: torch.Tensor,
        obj_func: Callable, 
        batch_size: int = 1,
        seed: int=0, 
        discrete_dims: List[int] = [],
        optimizer_params: Dict[str, Any] = None,
        init_design: str= "lhs",
        use_default: bool = True,
        default_conf: Optional[torch.Tensor] = None,
        manual_seed: bool = True,
    ):
        super().__init__(bounds, obj_func, 
                        batch_size=batch_size,
                        seed=seed, 
                        discrete_dims=discrete_dims, 
                        optimizer_params=optimizer_params,
                        init_design=init_design,
                        use_default=use_default,
                        default_conf=default_conf,
                        manual_seed=manual_seed,
        )

        self.num_init: int = optimizer_params.get('num_init', 10)
        self._X = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
        self._Y = torch.empty((0, 1), dtype=DTYPE, device=DEVICE)
        self.restart_counter = 0
        self.restart_iters = []

        for k, v in MCTURBO_ATTRS.items():
            if k not in optimizer_params:
                setattr(self, k, v)
            else:
                setattr(self, k, optimizer_params[k])
        self.obj_func = ObjFuncWrapper(bounds, discrete_dims, obj_func)

        self.mcturbo_state_params = optimizer_params.get('mcturbo_state_params', {})

        print_log('=' * 80, print_msg=True)
        print_log(f"[MCTuRBO] Initialized with following configurations", print_msg=True)
        for k, v in MCTURBO_ATTRS.items():
            print_log(f"\t{k}: {getattr(self, k)}", print_msg=True)

    @property
    def num_calls(self) -> int:
        return len(self._X)
    
    @property
    def X(self) -> torch.Tensor:
        X_data = self._X
        X_data = unnormalize(X_data, self.bounds)
        X_data[:, self.discrete_dims] = round_by_bounds(X_data[:, self.discrete_dims], self.bounds[:, self.discrete_dims])
        return X_data
    
    @property
    def Y(self) -> torch.Tensor:
        return self._Y
    
    def get_original_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._X, self._Y

    def initial_sampling(self, num_init: int):
        X_init: torch.Tensor = self.sampler.generate(num_init)
        if self.use_default:
            default_conf = normalize(self.default_conf, self.bounds)
            X_init = torch.cat([default_conf, X_init], dim=0)
            X_init = X_init[:num_init]

        Y_init = torch.tensor(
            [self.obj_func(x) for x in X_init], dtype=DTYPE, device=DEVICE,
        ).unsqueeze(-1)

        return X_init, Y_init 
    

    def generate_batch(self, 
        state: MCTurboState,
        model: SingleTaskGP, 
        X: torch.Tensor, # train_X - normalized, unit scale
        Y: torch.Tensor, # train_Y - standardized (unit scale)
        batch_size: int,
    ) -> torch.Tensor:
        assert X[:, self.continuous_dims].min() >= 0.0 and X[:, self.continuous_dims].max() <= 1.0
        assert torch.all(torch.isfinite(Y))

        # Length scales for all dimensions
        # weights - (dim, )
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        X_cands = state.generate_candidates(self.n_candidates, weights)
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            X_next = thompson_sampling(X_cands, num_samples=batch_size)
        return X_next

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.num_evals = num_evals

        while self.num_calls < self.num_evals:
            num_init = min(self.num_init, self.num_evals - self.num_calls)
            X_sampled, Y_sampled = self.initial_sampling(num_init)
            self._X = torch.cat((self._X, X_sampled), dim=0)
            self._Y = torch.cat((self._Y, Y_sampled), dim=0)
            if self.num_calls >= self.num_evals: return self.X, self.Y

            self.state = MCTurboState(
                seed=self.seed,
                bounds=self.bounds,
                X=X_sampled, Y=Y_sampled,
                state_params=self.mcturbo_state_params,
            )
            
            print_log('-' * 80, print_msg=True)
            print_log(f"[MCTuRBO] Start {self.restart_counter+1}-th modeling with {num_init} data points", print_msg=True)

            while not self.state.restart_triggered and self.num_calls < self.num_evals:
                X_sampled, Y_sampled = self.turbo_iter(X_sampled, Y_sampled)
                if self.num_calls >= self.num_evals: break
                if self.state.restart_triggered: 
                    # self.state.reinit_state()
                    self.restart_counter += 1
                    self.restart_iters.append(self.num_calls + 1)

        print_log(f'[MCTuRBO] Iterations where TuRBO restarts: {self.restart_iters}', print_msg=True)
        return self.X, self.Y

    
    
    def turbo_iter(self, X_sampled: torch.Tensor, Y_sampled: torch.Tensor):
        # -----------------------------------       
        # Normalize training data
        train_X = X_sampled
        train_Y = standardize(Y_sampled)

        # -----------------------------------  
        # Define the model (Posterior)
        with torch.enable_grad():
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5, ard_num_dims=self.dimension, lengthscale_constraint=Interval(0.005, 4.0),
                ),
            )
            model = SingleTaskGP(train_X, train_Y, covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # -----------------------------------
            # Generate a new batch
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                try:
                    fit_gpytorch_mll(mll)
                except Exception as ex:
                    print_log(f"[MCTuRBO] {ex} happens when fitting model, restart.")
                    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                    for _ in range(100):
                        optimizer.zero_grad()
                        output = model(X_sampled)
                        loss: torch.Tensor = -mll(output, train_Y.flatten())
                        loss.backward()
                        optimizer.step()

        X_next = self.generate_batch(
            state=self.state, model=model,
            X=train_X, Y=train_Y,
            batch_size=self.batch_size,
        )

        X_next = X_next[:min(self.num_evals - self.num_calls, self.batch_size)]
        Y_next = torch.tensor(
            [self.obj_func(x) for x in X_next], dtype=DTYPE, device=DEVICE,
        ).unsqueeze(-1)

        # -----------------------------------  
        # Update the state
        self.state.update_state(X_next, Y_next)
        X_sampled = torch.cat((X_sampled, X_next), dim=0)
        Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)
        self._X = torch.cat((self._X, X_next), dim=0)
        self._Y = torch.cat((self._Y, Y_next), dim=0)
                
        print_log(
            f"[MCTuRBO] [Restart {self.restart_counter}] {self.num_calls}) "
            f"Best value: {self.state.best_value:.2e} | Sampling Threshold: {self.state.sampling_threshold:.2e} | "
            f"num. failures: {self.state.failure_counter}/{self.state.failure_tolerance} | "
            f"num. successes: {self.state.success_counter}/{self.state.success_tolerance}", 
            print_msg=True
        )
        # print_log('-' * 80, print_msg=True)
        return X_sampled, Y_sampled
