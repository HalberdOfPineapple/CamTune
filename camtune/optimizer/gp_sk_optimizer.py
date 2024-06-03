import torch 
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple

from scipy.stats import norm
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, standardize, normalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from .base_optimizer import BaseOptimizer
from .optim_utils import ACQ_FUNC_MAP, round_by_bounds, ObjFuncWrapper
from camtune.utils import print_log, DTYPE, DEVICE


def expected_improvement(X: np.array, gp: GaussianProcessRegressor, y_best, xi=0.01):
    # Compute the mean and standard deviation from the Gaussian Process model
    mean, std = gp.predict(X, return_std=True)

    # Compute the improvement
    improvement = mean - y_best - xi
    z = improvement / std

    # Compute the Expected Improvement
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    
    # Return negative EI for minimization purposes (if needed)
    # For maximizing EI, keep as positive; for minimization, negate
    return ei


GP_ATTRS = {
    'num_restarts': 4,
    'raw_samples': 512,
    'n_candidates': 5000,
    'max_cholesky_size': float("inf"),
}
class GPROptimizer(BaseOptimizer):
    def __init__(
        self,
        bounds: torch.Tensor,
        obj_func: Callable,
        batch_size: int = 1,
        seed: int = 0,
        discrete_dims: List[int] = [],
        optimizer_params: Dict[str, Any] = None,
        init_design: str = 'LHS',
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
                         default_conf=default_conf,)
        self.obj_func = ObjFuncWrapper(self.bounds, self.discrete_dims, obj_func, self.tr_bounds)
        self.num_init: int = optimizer_params.get('num_init', 10)

        noise = 0.1
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

        for k, v in GP_ATTRS.items():
            if k not in self.optimizer_params:
                setattr(self, k, v)
            else:
                setattr(self, k, self.optimizer_params[k])

        self._X: torch.Tensor = torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
        self._Y: torch.Tensor = torch.empty((0, 1), dtype=self.dtype, device=self.device)

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
        X_init = self.sampler.generate(num_init)

        if self.use_default:
            default_conf = normalize(self.default_conf, self.bounds)
            X_init = torch.cat([default_conf, X_init], dim=0)
            X_init = X_init[:num_init]

        if self.tr_bounds is None:
            Y_init = torch.tensor(
                [self.obj_func(x) for x in X_init], dtype=DTYPE, device=DEVICE,
            ).unsqueeze(-1)
        else:
            print_log(f'[GP-BO] Using TR bounds to boost initial sampling.', print_msg=True)
            Y_init = torch.tensor(
                [self.obj_func.call_with_tr(x) for x in X_init], dtype=DTYPE, device=DEVICE,
            ).unsqueeze(-1)

        self._X = X_init
        self._Y = Y_init

        return X_init, Y_init
    
    def generate_batch(self, batch_size: int, gp: GaussianProcessRegressor, y_best: float):
        # Create a set of candidate points to evaluate EI
        candidate_X = np.random.rand(self.n_candidates, self.dimension)  # Random candidates
        
        # Calculate Expected Improvement for the candidates
        ei = expected_improvement(candidate_X, gp, y_best)

        # Select the top 'batch_size' candidates with the highest EI
        top_indices = np.argsort(ei)[:batch_size]  # Assuming we're minimizing
        X_next = torch.tensor(candidate_X[top_indices], dtype=self.dtype, device=self.device)

        return X_next

    @ignore_warnings(category=ConvergenceWarning)
    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.initial_sampling(self.num_init)
        while self._X.shape[0] < num_evals:
            batch_size = min(self.batch_size, num_evals - self._X.shape[0])

            train_X: np.array = self._X.cpu().numpy()
            train_Y: np.array = standardize(self._Y).cpu().numpy()
            
            # Fit a Gaussian Process to the training data
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5), 
                n_restarts_optimizer=self.num_restarts
            )
            gp.fit(train_X, train_Y)
            X_next = self.generate_batch(batch_size, gp, np.max(train_Y))
            Y_next = torch.tensor(
                [self.obj_func(x) for x in X_next], dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)

            self._X = torch.cat([self._X, X_next], dim=0)
            self._Y = torch.cat([self._Y, Y_next], dim=0)

            log_msg = (
                f"[GP-BO] Sample {self.num_calls} | "
                f"Best value: {self._Y.max().item():.2f} |"
            )
            print_log(log_msg, print_msg=True)
        
        return self.X, self.Y