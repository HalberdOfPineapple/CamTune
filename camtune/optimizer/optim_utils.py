import torch
import numpy as np
from typing import List, Callable
from torch.quasirandom import SobolEngine
from torch.distributions import Normal
from ConfigSpace import ConfigurationSpace, Configuration

import botorch
botorch.settings.debug(True)
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.utils.transforms import unnormalize, standardize, normalize

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from camtune.search_space import SearchSpace
from camtune.utils import DTYPE, DEVICE, print_log

ACQ_FUNC_MAP = {
    'ei': ExpectedImprovement,
    'qei': qExpectedImprovement,
}

def generate_random_discrete(num_evals: int, bounds: torch.Tensor, discrete_dims: List[int]) -> torch.Tensor:
    discrete_dim = len(discrete_dims)
    lower_bounds = bounds[0, discrete_dims]
    upper_bounds = bounds[1, discrete_dims]

    # Generate random samples within the unit hypercube [0,1]^D and then scale them to the bounds
    device, dtype = bounds.device, bounds.dtype
    random_samples = torch.rand(num_evals, discrete_dim, device=device, dtype=dtype)
    scaled_samples = lower_bounds + random_samples * (upper_bounds - lower_bounds)

    # Round the samples to the nearest integer
    rounded_samples = torch.round(scaled_samples)
    return rounded_samples

def assign_after_check(cand_val, key, params):
    return cand_val if key not in params else params[key]

def round_by_bounds(X_cands: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    return torch.max(
        torch.min(torch.round(X_cands), bounds[1, :]), 
        bounds[0, :]
    )

def tensor_to_config(
        sample_tensor: torch.Tensor, 
        search_space: SearchSpace
    ) -> Configuration:
        tensor_vals = sample_tensor.cpu().numpy()

        configspace: ConfigurationSpace = search_space.input_space
        hps: List = search_space.input_variables
        discrete_dims: List[int] = search_space.discrete_dims
        continuous_dims: List[int] = search_space.continuous_dims

        valid_config = {}
        for knob_idx in discrete_dims:
            num_val = int(tensor_vals[knob_idx])
            config_val = search_space.discrete_idx_to_value(knob_idx, num_val)
            valid_config[hps[knob_idx].name] = config_val

        for knob_idx in continuous_dims:
            config_val = float(tensor_vals[knob_idx])
            valid_config[hps[knob_idx].name] = config_val

        return Configuration(configspace, values=valid_config)

def nd_array_to_config(
    sample_array: np.ndarray,
    search_space: SearchSpace,
    data_normalized: bool=True
) -> Configuration:
    configspace: ConfigurationSpace = search_space.input_space
    hps: List = search_space.input_variables
    discrete_dims: List[int] = search_space.discrete_dims
    continuous_dims: List[int] = search_space.continuous_dims

    if data_normalized:
        bounds: torch.Tensor = search_space.bounds
        sample_array = unnormalize(torch.tensor(sample_array).to(dtype=DTYPE, device=DEVICE), bounds).cpu().numpy()

    valid_config = {}
    for knob_idx in discrete_dims:
        num_val = int(sample_array[knob_idx])
        config_val = search_space.discrete_idx_to_value(knob_idx, num_val)
        valid_config[hps[knob_idx].name] = config_val
    
    for knob_idx in continuous_dims:
        config_val = float(sample_array[knob_idx])
        valid_config[hps[knob_idx].name] = config_val

    return Configuration(configspace, values=valid_config)

def get_default(expr_config: dict, search_space: SearchSpace, negate: bool):
    if ('use_default' in expr_config['tune'] and expr_config['tune']['use_default']) or \
            expr_config['tune']['optimizer'] == 'constant':
        default_conf_tensor: torch.Tensor = search_space.get_default_conf_tensor()
    else:
        default_conf_tensor: torch.Tensor = None

    default_perf: float = None if 'default_perf' not in expr_config['database'] else expr_config['database']['default_perf']
    if default_perf:
        # need to consider sign change because it needs to be directly used by optimizer
        default_perf = -default_perf if negate else default_perf 
    return default_conf_tensor, default_perf


def RAASP(
    sobol: SobolEngine,
    x_center: torch.Tensor, 
    tr_lbs: torch.Tensor, 
    tr_ubs: torch.Tensor, 
    num_candidates: int=5000,
) -> torch.Tensor:
        dimension = x_center.shape[-1]
        pert = sobol.draw(num_candidates).to(dtype=DTYPE, device=DEVICE)
        pert = tr_lbs + (tr_ubs - tr_lbs) * pert

        prob_perturb = min(20.0 / dimension, 1.0)
        mask = torch.rand(len(pert), dimension, dtype=DTYPE, device=DEVICE) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dimension - 1, size=(len(ind),), device=DEVICE)] = 1

        # Create candidate points from the perturbations and the mask
        X_cands = x_center.expand(len(pert), dimension).clone()
        X_cands[mask] = pert[mask]

        return X_cands

class ObjFuncWrapper:
    def __init__(self, bounds: torch.Tensor, discrete_dims: List[int], obj_func: Callable, tr_bounds: torch.Tensor=None):
        self.discrete_dims = discrete_dims
        self.obj_func = obj_func

        self.bounds = bounds
        self.tr_bounds = tr_bounds

    def __call__(self, input_x: torch.Tensor):
        x = unnormalize(input_x, self.bounds)
        x[self.discrete_dims] = round_by_bounds(x[self.discrete_dims], self.bounds[:, self.discrete_dims])

        eval_val = self.obj_func(x)
        if isinstance(eval_val, torch.Tensor):
            eval_val = eval_val.cpu().item()
        return eval_val

    def call_with_tr(self, input_x: torch.Tensor):
        x = unnormalize(input_x, self.tr_bounds)
        x[self.discrete_dims] = round_by_bounds(x[self.discrete_dims], self.tr_bounds[:, self.discrete_dims])

        eval_val = self.obj_func(x)
        if isinstance(eval_val, torch.Tensor):
            eval_val = eval_val.cpu().item()
        return eval_val


def train_gp(
    dimension: int,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    max_cholesky_size: float,
    training_steps: int=100,
):
    # -----------------------------------
    # Define the model (Posterior)
    with torch.enable_grad():
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5, ard_num_dims=dimension, lengthscale_constraint=Interval(0.005, 4.0),
            ),
        )
        model = SingleTaskGP(
            train_X, train_Y, covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # -----------------------------------
        # Generate a new batch
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            try:
                fit_gpytorch_mll(mll)
            except Exception as ex:
                print_log(f"[OptimUtils] Train_GP: {ex} happens when fitting model, restart.")
                optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                for _ in range(training_steps):
                    optimizer.zero_grad()
                    output = model(train_X)
                    loss = -mll(output, train_Y.flatten())
                    loss.backward()
                    optimizer.step()

    return model


def expected_improvement(
        X_cands: torch.Tensor, 
        model: botorch.models.SingleTaskGP, 
        best_f: float,
    ) -> torch.Tensor:
    with torch.no_grad():
        posterior = model.posterior(X_cands) # shape (num_cands,)
        mu = posterior.mean
        sigma = posterior.variance.sqrt()
        
        # Calculate Z
        Z = (mu - best_f) / sigma
        normal = Normal(0, 1)
        
        # Calculate EI
        ei = (mu - best_f) * normal.cdf(Z) + sigma * normal.log_prob(Z).exp()
        
        # Handle case when sigma is zero
        ei[sigma == 0] = 0.0
        
    return ei