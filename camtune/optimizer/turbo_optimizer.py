import os
import math
import json
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

from .base_optimizer import BaseOptimizer
from .optim_utils import round_by_bounds, ObjFuncWrapper, RAASP
from .turbo_utils import TurboState, update_state
from .winter_utils import MCTSLocalControl, Node
from .mcts_copilot import MCTSCopilot

from camtune.utils import (print_log, get_expr_name, get_result_dir, get_log_idx, DEVICE, DTYPE)


ACQFS = {"ts", "ei"}
TURBO_ATTRS = {
    'use_copilot': False,
    'copilot_params': {},
    'step_length_factor': 0,

    'var_sobol': False,
    'uniform': False,
    'bounding_box_mode': False,

    'save_tr': True,
    'num_restarts': 10,
    'raw_samples': 512,
    'n_candidates': 5000,
    'max_cholesky_size': float("inf"),
    'acqf': "ts",
    "init_bounding_box_length": 0.0001,
    "training_steps": 100,
}
SINGLE_SAMPLING_THRE = 5


def tr_to_dict(x_center: torch.Tensor, tr_lbs: torch.Tensor, tr_ubs: torch.Tensor, weights: torch.Tensor, state: TurboState) -> Dict:
    return {
        'x_center': torch.clone(x_center).detach().cpu().numpy().tolist(),
        'tr_lbs': torch.clone(tr_lbs).detach().cpu().numpy().tolist(),
        'tr_ubs': torch.clone(tr_ubs).detach().cpu().numpy().tolist(),
        'weights': torch.clone(weights).detach().cpu().numpy().tolist(),
        **state.to_dict(),
    }

class TuRBO(BaseOptimizer):
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

        for k, v in TURBO_ATTRS.items():
            if optimizer_params is None or k not in optimizer_params:
                setattr(self, k, v)
            elif isinstance(v, dict):
                setattr(self, k, v)
                if optimizer_params[k] is not None:
                    for kk in optimizer_params[k]:
                        getattr(self, k)[kk] = optimizer_params[k][kk]
            else:
                setattr(self, k, optimizer_params[k])

        self.obj_func = ObjFuncWrapper(bounds, discrete_dims, obj_func)
        self.enable_step_len = optimizer_params.get('enable_step_len', False)
        self.use_copilot: bool = self.use_copilot
        self.turbo_state_params = optimizer_params.get('turbo_state_params', {})
        for ts_param, ts_val in self.turbo_state_params.items():
            self.turbo_state_params[ts_param] = eval(ts_val) if isinstance(ts_val, str) else ts_val

        if self.acqf not in ACQFS:
            raise ValueError(f"Acquisition function {self.acqf} not supported")

        print_log('=' * 80, print_msg=True)
        print_log(f"[TuRBO] Initialized with following configurations", print_msg=True)
        for k, v in TURBO_ATTRS.items():
            print_log(f"\t{k}: {getattr(self, k)}", print_msg=True)
        for k, v in self.turbo_state_params.items():
            print_log(f"\t{k}: {v}", print_msg=True)

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
    
    def save_tr_info(self):
        if get_log_idx() == 0:
            tr_save_path = os.path.join(get_result_dir(), f'{get_expr_name()}_tr.json')
        else:
            tr_save_path = os.path.join(get_result_dir(), f'{get_expr_name()}_{get_log_idx()}_tr.json')
        print_log(f'[TuRBO] save_tr_info: Saving the trust region data into {tr_save_path}', print_msg=True)
        with open(tr_save_path, 'w') as f:
            json.dump(self.tr_records, f)

    def copilot_gen_cands(self) -> bool:
        if not self.use_copilot: return False
        tree_params = self.copilot_params.get('tree_params', {})
        return tree_params.get('gen_cands', False)

    # ----------------------------------------------------
    # Optimization Related
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
        sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed + (0 if not self.var_sobol else self.num_calls))
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
        model: SingleTaskGP, 
        X: torch.Tensor, # train_X - normalized, unit scale
        Y: torch.Tensor, # train_Y - standardized (unit scale)
        batch_size: int,
    ) -> torch.Tensor:
        assert X[:, self.continuous_dims].min() >= 0.0 and X[:, self.continuous_dims].max() <= 1.0
        assert torch.all(torch.isfinite(Y))

        x_center: torch.Tensor = X[Y.argmax(), :].clone()

        # Length scales for all dimensions
        # weights - (dim, )
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # Clamps all elements in input into the range [min, max]
        # tr_lbs, tr_ubs - (1, dim)
        tr_lbs = torch.clamp(x_center - state.length / 2 * weights, 0.0, 1.0)
        tr_ubs = torch.clamp(x_center + state.length / 2 * weights, 0.0, 1.0)

        self.tr_records[self.restart_counter][self.num_calls] = \
            tr_to_dict(x_center, tr_lbs, tr_ubs, weights, state)
        
        if self.use_copilot:
            X_cands = self.gen_candidates(x_center, tr_lbs, tr_ubs)
            X_next = self.copilot.generate_batch(batch_size, X_cands, X, Y, model)
            return X_next
        else:
            # X_cands - (n_candidates, dim)
            X_cands = self.gen_candidates(x_center, tr_lbs, tr_ubs)
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = thompson_sampling(X_cands, num_samples=batch_size)

        return X_next

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.num_evals = num_evals
        self.tr_records = {}

        self.restart_counter = 0
        restart_iters = []
        while self.num_calls < self.num_evals:
            print_log('-' * 80, print_msg=True)
            print_log(f"[TuRBO] Restart {self.restart_counter}:", print_msg=True)

            # ----------------------------------------------
            # Initial sampling at the beginning of each restart
            num_init = min(self.num_init, self.num_evals - self.num_calls)
            X_sampled, Y_sampled = self.initial_sampling(num_init)
            self._X = torch.cat((self._X, X_sampled), dim=0)
            self._Y = torch.cat((self._Y, Y_sampled), dim=0)
            if self.num_calls >= self.num_evals: 
                break

            if self.use_copilot:
                self.copilot: MCTSCopilot = MCTSCopilot(
                    bounds=self.bounds,
                    seed=self.seed,
                    num_evals=num_evals,
                    max_func_value=self._Y.abs().max().item(),
                    tr_length_min=self.turbo_state_params.get('length_min', 0.5 ** 6),
                    **self.copilot_params,
                )
                if self.copilot.enable_guide_restart:
                    extra_num_init: int = min(min(len(X_sampled) / 4, 5), self.num_evals - self.num_calls)
                    extra_X_init: torch.Tensor = self.copilot.guide_restart(
                        extra_num_init, self._X, self._Y,
                        init_bounding_box_length=self.init_bounding_box_length
                    )
                    if extra_X_init.shape[0] > 0:
                        extra_X_init = extra_X_init[:num_init]
                        extra_Y_init = torch.tensor(
                            [self.obj_func(x) for x in extra_X_init], dtype=DTYPE, device=DEVICE,
                        ).unsqueeze(-1)
                        X_sampled = torch.cat((X_sampled, extra_X_init), dim=0)
                        Y_sampled = torch.cat((Y_sampled, extra_Y_init), dim=0)
                        self._X = torch.cat((self._X, extra_X_init), dim=0)
                        self._Y = torch.cat((self._Y, extra_Y_init), dim=0)
                        if self.num_calls >= self.num_evals: break
                    else:
                        print_log(f"[TuRBO] Copilot guide restart failed (no samples generated within region)")

            
            # ----------------------------------------------
            # TuRBO state initialization
            state = TurboState(
                dim=self.dimension, 
                batch_size=self.batch_size,
                **self.turbo_state_params,
            )

            # ----------------------------------------------
            # TuRBO iterations
            print_log(f"[TuRBO] Start local modeling with {num_init} data points", print_msg=True)
            self.tr_records[self.restart_counter] = {}
            while not state.restart_triggered and self.num_calls < self.num_evals: # Run until TuRBO converges
                X_sampled, Y_sampled, state = self.turbo_iter(X_sampled, Y_sampled, state)
                if self.use_copilot and self.copilot.state.restart_triggered: break
            if self.num_calls >= self.num_evals: break
    
            self.restart_counter += 1
            restart_iters.append(self.num_calls + 1)

        if self.save_tr:
            self.save_tr_info()

        print_log(f'[TuRBO] Iterations where TuRBO restarts: {restart_iters}', print_msg=True)
        return self.X, self.Y

    
    
    def turbo_iter(self, X_sampled: torch.Tensor, Y_sampled: torch.Tensor, state: TurboState):
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
                    print_log(f"[TuRBO] {ex} happens when fitting model, restart.")
                    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                    for _ in range(100):
                        optimizer.zero_grad()
                        output = model(X_sampled)
                        loss = -mll(output, train_Y.flatten())
                        loss.backward()
                        optimizer.step()

            X_next = self.generate_batch(
                state=state, model=model,
                X=train_X, Y=train_Y,
                batch_size=self.batch_size,
            )

        X_next = X_next[:min(self.num_evals - self.num_calls, self.batch_size)]
        Y_next = torch.tensor(
            [self.obj_func(x) for x in X_next], dtype=DTYPE, device=DEVICE,
        ).unsqueeze(-1)

        # -----------------------------------  
        # Update the state
        # state = update_state(state, Y_next)
        X_sampled = torch.cat((X_sampled, X_next), dim=0)
        Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)
        self._X = torch.cat((self._X, X_next), dim=0)
        self._Y = torch.cat((self._Y, Y_next), dim=0)
        if self.step_length_factor > 0:
            step_length = torch.std(torch.topk(Y_sampled.flatten().cpu(), 10).values) * self.step_length_factor
        else:
            step_length = 0
        state = update_state(state, Y_next, step_length=step_length)

        if self.use_copilot: self.copilot.update_state(Y_next, step_length=step_length)
                
        print_log(
            f"[TuRBO] [Restart {self.restart_counter}] {self.num_calls}) "
            f"Best value: {state.best_value:.2e} | TR length: {state.length:.2e} | "
            f"num. failures: {state.failure_counter}/{state.failure_tolerance} | "
            f"num. successes: {state.success_counter}/{state.success_tolerance}", 
            print_msg=True
        )
        if self.use_copilot:
            print_log(self.copilot.state_msg, print_msg=True)
            print_log('-' * 80, print_msg=True)
        return X_sampled, Y_sampled, state



# ----------------------------------------------------------------------------------------------------------------------

    def optimize_local(self, num_evals: int, local_control: MCTSLocalControl, path: List[Node]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        self.num_evals = num_evals
        self.local_control = local_control
        self.tr_records = {}

        # ----------------------------------------------
        # Initial sampling
        num_init = min(self.num_init, num_evals)
        if local_control.init_mode:
            X_init = Node.generate_samples_in_region(
                num_samples=num_init,
                path=path,
                init_bounding_box_length=self.init_bounding_box_length,
                seed=self.seed,
            )
            Y_init = torch.tensor(
                [self.obj_func(x) for x in X_init], dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)
        else:
            X_init, Y_init = self.initial_sampling(num_init)

        X_sampled, Y_sampled = X_init, Y_init
        self._X = X_sampled
        self._Y = Y_sampled

        if self.num_calls >= self.num_evals: 
            return X_sampled, Y_sampled, {}
        
        print_log('-' * 80, print_msg=True)
        print_log(f"[TuRBO-Local] Start local modelling with {X_init.shape[0]} samples:", print_msg=True)

        # ----------------------------------------------
        # TuRBO state initialization
        state = TurboState(
            dim=self.dimension, 
            batch_size=self.batch_size,
            **self.turbo_state_params,
        )

        # ----------------------------------------------
        # TuRBO iterations
        while not state.restart_triggered and self.num_calls < self.num_evals: # Run until TuRBO converges
            X_sampled, Y_sampled, state = self.turbo_iter_local(X_sampled, Y_sampled, state, path)

        return X_sampled, Y_sampled, self.tr_records.copy()

    def turbo_iter_local(self, X_sampled: torch.Tensor, Y_sampled: torch.Tensor, state: TurboState, path: List[Node]):
        # -----------------------------------       
        # Normalize training data
        if not self.local_control.attach_leaf:
            train_X = X_sampled
            train_Y = standardize(Y_sampled)
        else:
            train_X = torch.cat((X_sampled, path[-1].sample_bag[0]), dim=0)
            train_Y = torch.cat((Y_sampled, path[-1].sample_bag[1]), dim=0)
            train_Y = standardize(train_Y)

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
                    print_log(f"[TuRBO-Local] {ex} happens when fitting model, restart.")
                    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                    for _ in range(self.training_steps):
                        optimizer.zero_grad()
                        output = model(train_X)
                        loss = -mll(output, train_Y.flatten())
                        loss.backward()
                        optimizer.step()

        X_next = self.generate_batch_local(
            state=state, model=model,
            X=train_X, Y=train_Y,
            batch_size=self.batch_size,
            path=path if self.local_control.real_mode or self.local_control.sampling_mode else None,
        )

        X_next = X_next[:min(self.num_evals - self.num_calls, self.batch_size)]
        Y_next = torch.tensor(
            [self.obj_func(x) for x in X_next], dtype=DTYPE, device=DEVICE, 
        ).unsqueeze(-1)

        # -----------------------------------  
        # Update the state
        state = update_state(state, Y_next)
        X_sampled = torch.cat((X_sampled, X_next), dim=0)
        Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)
        self._X = torch.cat((self._X, X_next), dim=0)
        self._Y = torch.cat((self._Y, Y_next), dim=0)
        
        print_log(
            f"[TuRBO-Local] {self.num_calls}) "
            f"Best value: {state.best_value:.2e} | TR length: {state.length:.2e} | "
            f"num. failures: {state.failure_counter}/{state.failure_tolerance} | "
            f"num. successes: {state.success_counter}/{state.success_tolerance}", 
            print_msg=True
        )

        return X_sampled, Y_sampled, state

    def gen_candidates_local(
            self, 
            x_center: torch.Tensor, 
            tr_lbs: torch.Tensor, 
            tr_ubs: torch.Tensor, 
            weights: torch.Tensor=None,
            path: List[Node]=None,
        ) -> torch.Tensor:
        
        if path is None:
            return self.gen_candidates(x_center, tr_lbs, tr_ubs)
        elif self.bounding_box_mode:
            sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
            sample_trial_cnt = 0
            X_cands = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
            while X_cands.shape[0] < self.n_candidates and sample_trial_cnt < SINGLE_SAMPLING_THRE:
                X_cands_iter = Node.bounding_box_sampling(
                    num_samples=self.n_candidates,
                    sobol=sobol, path=path, weights=weights,
                    init_bounding_box_length=self.init_bounding_box_length,
                    lb=tr_lbs, ub=tr_ubs,
                )
                X_cands = torch.cat([X_cands, X_cands_iter], dim=0)
                sample_trial_cnt += 1
        else:
            sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
            sample_trial_cnt = 0
            X_cands = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
            while X_cands.shape[0] < self.n_candidates and sample_trial_cnt < SINGLE_SAMPLING_THRE:
                if not self.uniform:
                    X_cands_iter = RAASP(sobol, x_center, tr_lbs, tr_ubs, self.n_candidates)
                else:
                    X_cands_iter = torch.rand((self.n_candidates - X_cands.shape[0], self.dimension), device=DEVICE)
                    X_cands_iter = tr_lbs + (tr_ubs - tr_lbs) * X_cands_iter

                X_cands_iter = X_cands_iter[Node.path_filter(path, X_cands_iter)]
                X_cands = torch.cat([X_cands, X_cands_iter], dim=0)
                sample_trial_cnt += 1
        valid_samples = X_cands.shape[0] 
        total_samples = sample_trial_cnt * self.n_candidates

        if X_cands.shape[0] >= self.n_candidates:
            X_cands = X_cands[:self.n_candidates]
        else:
            print_log(f"[TuRBO-Local] No enough samples generated in the region: {X_cands.shape[0]} < {self.n_candidates}", print_msg=True)
            sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
            if not self.uniform:
                X_cands_extra = RAASP(sobol, x_center, tr_lbs, tr_ubs, self.n_candidates - X_cands.shape[0])
            else:
                X_cands_extra = torch.rand((self.n_candidates - X_cands.shape[0], self.dimension), device=DEVICE)
                X_cands_extra = tr_lbs + (tr_ubs - tr_lbs) * X_cands_extra
            X_cands = torch.cat([X_cands, X_cands_extra], dim=0)
    
        print_log(
            f"[TuRBO-Local] {self.num_calls}) valid sampling rate from region: {valid_samples}/{total_samples}", 
            print_msg=True
        )
        print_log('-' * 80, print_msg=True)
        return X_cands

    def generate_batch_local(self, 
        state: TurboState,
        model: SingleTaskGP, 
        X: torch.Tensor, # train_X - normalized, unit scale
        Y: torch.Tensor, # train_Y - standardized (unit scale)
        batch_size: int,
        path: List[Node] = None,
    ) -> torch.Tensor:
        assert X[:, self.continuous_dims].min() >= 0.0 and X[:, self.continuous_dims].max() <= 1.0
        assert torch.all(torch.isfinite(Y))

        x_center: torch.Tensor = X[Y.argmax(), :].clone()

        # Length scales for all dimensions
        # weights - (dim, )
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # Clamps all elements in input into the range [min, max]
        # tr_lbs, tr_ubs - (1, dim)
        tr_lbs = torch.clamp(x_center - state.length / 2 * weights, 0.0, 1.0)
        tr_ubs = torch.clamp(x_center + state.length / 2 * weights, 0.0, 1.0)

        self.tr_records[self.num_calls] = tr_to_dict(x_center, tr_lbs, tr_ubs, weights, state)

        if self.acqf == "ts":
            # X_cands - (n_candidates, dim)
            X_cands = self.gen_candidates_local(x_center, tr_lbs, tr_ubs, weights, path)
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = thompson_sampling(X_cands, num_samples=batch_size)

        elif self.acqf == "ei":
            ei = qExpectedImprovement(model=model, best_f=Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lbs, tr_ubs]),
                q=batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        else:
            raise ValueError(f"[TuRBO] Acquisition function {self.acqf} not supported")

        return X_next