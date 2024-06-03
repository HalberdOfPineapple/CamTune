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
from botorch.exceptions import ModelFittingError
from botorch.utils.transforms import unnormalize, standardize, normalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from camtune.utils import print_log, DTYPE, DEVICE
from .base_optimizer import BaseOptimizer
from .optim_utils import round_by_bounds, ObjFuncWrapper
from .baxus_utils import *

device, dtype = DEVICE, DTYPE

BAXUS_ATTRS = {
    'acqf': 'ts',
    'n_candidates': 5000,
    'raw_samples': 512,
    'num_restarts': 10,
    'max_cholesky_size': float("inf"),
}

class BAXUS(BaseOptimizer):
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
    ):
        super().__init__(bounds, obj_func, 
                        batch_size=batch_size,
                        seed=seed, 
                        discrete_dims=discrete_dims, 
                        optimizer_params=optimizer_params,
                        init_design=init_design,
                        use_default=use_default,
                        default_conf=default_conf,
                    )
        self.seed = seed
        for key, value in BAXUS_ATTRS.items():
            setattr(self, key, optimizer_params.get(key, value))

        self.num_init: int = optimizer_params.get('num_init', 10)
        self.baxus_state_params = optimizer_params.get('baxus_state_params', {})
        self.obj_func = ObjFuncWrapper(bounds, discrete_dims, obj_func)

        self._X = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
        self._Y = torch.empty((0, 1), dtype=DTYPE, device=DEVICE)

        print_log('=' * 80, print_msg=True)
        print_log(f"[BAXUS] Initialized with following configurations", print_msg=True)

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

    def gen_candidates(
            self, 
            x_center: torch.Tensor, 
            tr_lbs: torch.Tensor, 
            tr_ubs: torch.Tensor, 
        ) -> torch.Tensor:
        dim = x_center.shape[-1]
        sobol = SobolEngine(dim, scramble=True, seed=self.seed)
        pert = sobol.draw(self.n_candidates).to(dtype=DTYPE, device=DEVICE)
        pert = tr_lbs + (tr_ubs - tr_lbs) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(self.n_candidates, dim, dtype=DTYPE, device=DEVICE) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=DEVICE)] = 1

        # Create candidate points from the perturbations and the mask
        X_cands = x_center.expand(self.n_candidates, dim).clone()
        X_cands[mask] = pert[mask]

        return X_cands

    def generate_batch(
        self,
        state: BaxusState,
        model: SingleTaskGP,
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.detach().view(-1)
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length, -1.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length, -1.0, 1.0)

        if self.acqf == "ts":
            X_cands = self.gen_candidates(x_center, tr_lb, tr_ub)

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cands, num_samples=batch_size)
        else:
            raise NotImplementedError(f"Acquisition function {self.acqf} not implemented")

        return X_next

    
    
    def get_initial_points(self, dim, num_init):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=self.seed)
        # return sobol.draw(n=num_init).to(dtype=dtype, device=device)
        return 2 * sobol.draw(n=num_init).to(dtype=dtype, device=device) - 1

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.num_evals = num_evals

        self.restart_counter = 0
        self.restart_iters = []
        while self.num_calls < self.num_evals:
            print_log('-' * 80, print_msg=True)
            print_log(f"[BAxUS] Restart {self.restart_counter}:", print_msg=True)

            # ----------------------------------------------
            # Initial sampling at the beginning of each restart
            num_init = min(self.num_init, self.num_evals - self.num_calls)
            state = BaxusState(dim=self.dimension, eval_budget=self.num_evals - num_init)

            # embed_matrix - (d_init, dim), converting embedding vectors to original space
            embed_matrix = embedding_matrix(input_dim=state.dim, target_dim=state.d_init)
            X_sampled_embed = self.get_initial_points(state.d_init, num_init)

            X_sampled = X_sampled_embed @ embed_matrix
            X_sampled = (X_sampled + 1) / 2

            Y_sampled = torch.tensor(
                [self.obj_func(x) for x in X_sampled], dtype=DTYPE, device=DEVICE,
            ).unsqueeze(-1)

            self._X = torch.cat((self._X, X_sampled), dim=0)
            self._Y = torch.cat((self._Y, Y_sampled), dim=0)
            if self.num_calls >= self.num_evals: 
                break

            print_log(f"[BAxUS] Start local modeling with {num_init} data points", print_msg=True)
            

            # ----------------------------------------------
            # BAxUS iterations
            while self.num_calls < self.num_evals: # Run until BAxUS converges
                X_sampled_embed, X_sampled, Y_sampled, state = self.baxus_iter(embed_matrix, X_sampled_embed, X_sampled, Y_sampled, state)
                if state.restart_triggered:
                    state.restart_triggered = False
                    
                    embed_matrix, X_sampled_embed = increase_embedding_and_observations(
                        embed_matrix, X_sampled_embed, state.new_bins_on_split
                    )
                    print(f"[BAxUS] Restart retriggered: increasing target space to {len(embed_matrix)}")

                    state.target_dim = len(embed_matrix)
                    state.length = state.length_init
                    state.failure_counter = 0
                    state.success_counter = 0

                    self.restart_counter += 1
                    self.restart_iters.append(self.num_calls + 1)

        print_log(f'[BAxUS] Iterations where BAxUS restarts: {self.restart_iters}', print_msg=True)
        return self.X, self.Y
    
    def baxus_iter(
            self, 
            embed_matrix: torch.Tensor, # (d_init, dim)
            X_sampled_embed: torch.Tensor,
            X_sampled: torch.Tensor, 
            Y_sampled: torch.Tensor, 
            state: BaxusState
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, BaxusState]:

        with botorch.settings.validate_input_scaling(False):
            # -----------------------------------       
            # Normalize training data
            train_X = X_sampled_embed
            train_Y = standardize(Y_sampled)

            with torch.enable_grad():
                # -----------------------------------  
                # Define the model (Posterior)
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(train_X, train_Y, likelihood=likelihood)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # -----------------------------------  
                # Generate a new batch
                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        # Right after increasing the target dimensionality, the covariance matrix becomes indefinite
                        # In this case, the Cholesky decomposition might fail due to numerical instabilities
                        # In this case, we revert to Adam-based optimization
                        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                        for _ in range(100):
                            optimizer.zero_grad()
                            output = model(X_sampled_embed)
                            loss = -mll(output, train_Y.flatten())
                            loss.backward()
                            optimizer.step()

            # Create a batch
            X_next_embed = self.generate_batch(
                state=state, model=model,
                X=train_X, Y=train_Y,
                batch_size=self.batch_size,
            )
            X_next = X_next_embed @ embed_matrix
            X_next = (X_next + 1) / 2
            
            X_next = X_next[:min(self.num_evals - self.num_calls, self.batch_size)]
            Y_next = torch.tensor(
                [self.obj_func(x) for x in X_next], dtype=DTYPE, device=DEVICE,
            ).unsqueeze(-1)

            # -----------------------------------  
            # Update the state
            state = update_state(state, Y_next)
            X_sampled_embed = torch.cat((X_sampled_embed, X_next_embed), dim=0)
            X_sampled = torch.cat((X_sampled, X_next), dim=0)
            Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)
            self._X = torch.cat((self._X, X_next), dim=0)
            self._Y = torch.cat((self._Y, Y_next), dim=0)
                    
            print_log(
                f"[BAxUS] [Restart {self.restart_counter}] {self.num_calls}) "
                f"Best value: {state.best_value:.2e} | TR length: {state.length:.2e} | "
                f"num. failures: {state.failure_counter}/{state.failure_tolerance} | "
                f"num. successes: {state.success_counter}/{state.success_tolerance}", 
                print_msg=True
            )

        return X_sampled_embed, X_sampled, Y_sampled, state