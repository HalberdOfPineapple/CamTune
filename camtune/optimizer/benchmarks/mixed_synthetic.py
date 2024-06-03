import numpy as np
import torch

from .base_benchmark import BaseBenchmark, EffectiveBenchmark
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin
from camtune.utils.logger import print_log
from camtune.utils.vars import DEVICE, DTYPE

class Func2C(BaseBenchmark):
    """Func2C is a mixed categorical and continuous function. The first 2 dimensions are categorical,
    with possible 3 and 5 possible values respectively. The last 2 dimensions are continuous"""

    """
    Global minimum of this function is at
    x* = [1, 1, -0.0898/2, 0.7126/2]
    with f(x*) = -0.2063
    """

    def __init__(self, lamda=1e-6, **kwargs):
        # Specifies the indices of the dimensions that are categorical and continuous, respectively
        super(Func2C, self).__init__(**kwargs)

        self.discrete_dims = np.array([0, 1])
        self.continuous_dims = np.array([2, 3])
        self.dim = len(self.continuous_dims) + len(self.discrete_dims)

        self.disc_lbs = torch.tensor([0, 0]).to(dtype=DTYPE, device=DEVICE)
        self.cont_lbs = torch.tensor([self.lb] * len(self.continuous_dims)).to(dtype=DTYPE, device=DEVICE)
        self.lbs = torch.cat([self.disc_lbs, self.cont_lbs], dim=0)

        self.disc_ubs = torch.tensor([3-1, 5-1]).to(dtype=DTYPE, device=DEVICE)
        self.cont_ubs = torch.tensor([self.ub] * len(self.continuous_dims)).to(dtype=DTYPE, device=DEVICE)
        self.ubs = torch.cat([self.disc_ubs, self.cont_ubs], dim=0)

        self.bounds = torch.stack([self.lbs, self.ubs], dim=0) # shape: (2, 4)

        # Specfies the range for the continuous variables
        self.lamda = lamda

    @property
    def obj_func(self):
        return self.compute_single

    def compute(self, X: torch.Tensor):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]

        res = np.zeros((N, ))
        X_cat = X[:, self.discrete_dims]
        X_cont = X[:, self.continuous_dims]
        X_cont = X_cont * 2

        for i, X in enumerate(X):
            if X_cat[i, 0] == 0:
                res[i] = myrosenbrock(X_cont[i, :])
            elif X_cat[i, 0] == 1:
                res[i] = mysixhumpcamp(X_cont[i, :])
            else:
                res[i] = mybeale(X_cont[i, :])

            if X_cat[i, 1] == 0:
                res[i] += myrosenbrock(X_cont[i, :])
            elif X_cat[i, 1] == 1:
                res[i] += mysixhumpcamp(X_cont[i, :])
            else:
                res[i] += mybeale(X_cont[i, :])
        res += self.lamda * np.random.rand(*res.shape)
        return res if not self.negate else -res
    
    def compute_single(self, X: torch.Tensor):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        res = np.zeros((1, ))
        X_cat = X[:, self.discrete_dims]
        X_cont = X[:, self.continuous_dims]
        X_cont = X_cont * 2

        if X_cat[0, 0] == 0:
            res[0] = myrosenbrock(X_cont[0, :])
        elif X_cat[0, 0] == 1:
            res[0] = mysixhumpcamp(X_cont[0, :])
        else:
            res[0] = mybeale(X_cont[0, :])
        
        if X_cat[0, 1] == 0:
            res[0] += myrosenbrock(X_cont[0, :])
        elif X_cat[0, 1] == 1:
            res[0] += mysixhumpcamp(X_cont[0, :])
        else:
            res[0] += mybeale(X_cont[0, :])
        res += self.lamda * np.random.rand(*res.shape)
        return res[0] if not self.negate else -res[0]

    

class ShiftedFunc2C(Func2C):
    def __init__(self, lamda=1e-6, **kwargs):
        # Specifies the indices of the dimensions that are categorical and continuous, respectively
        super(ShiftedFunc2C, self).__init__(**kwargs)

        shifted_amt = torch.tensor([-20, 10]).to(dtype=DTYPE, device=DEVICE)

        self.discrete_dims = np.array([0, 1])
        self.continuous_dims = np.array([2, 3])
        self.dim = len(self.continuous_dims) + len(self.discrete_dims)

        self.disc_lbs = torch.tensor([0, 0]).to(dtype=DTYPE, device=DEVICE)
        self.cont_lbs = torch.tensor([self.lb] * len(self.continuous_dims)).to(dtype=DTYPE, device=DEVICE)
        self.cont_lbs = self.cont_lbs + shifted_amt
        self.lbs = torch.cat([self.disc_lbs, self.cont_lbs], dim=0)

        self.disc_ubs = torch.tensor([3-1, 5-1]).to(dtype=DTYPE, device=DEVICE)
        self.cont_ubs = torch.tensor([self.ub] * len(self.continuous_dims)).to(dtype=DTYPE, device=DEVICE)
        self.cont_ubs = self.cont_ubs + shifted_amt
        self.ubs = torch.cat([self.disc_ubs, self.cont_ubs], dim=0)

        self.bounds = torch.stack([self.lbs, self.ubs], dim=0) # shape: (2, 4)

        # Specfies the range for the continuous variables
        self.lamda = lamda
    

# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X: torch.Tensor):
    X = X.detach().cpu().numpy()
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:  # one observation
        x1 = X[0]
        x2 = X[1]
    else:  # multiple observations
        x1 = X[:, 0]
        x2 = X[:, 1]
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx.reshape(-1, 1) / 300


# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html
# =============================================================================
def mysixhumpcamp(X: torch.Tensor):
    X = X.detach().cpu().numpy()
    X = np.reshape(X, (-1, 2))
    if len(X.shape) == 1:
        x1 = X[0]
        x2 = X[1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval.reshape(-1, 1) / 10


# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X: torch.Tensor):
    X = X.detach().cpu().numpy() / 2
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:
        x1 = X[0] * 2
        x2 = X[1] * 2
    else:
        x1 = X[:, 0] * 2
        x2 = X[:, 1] * 2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval.reshape(-1, 1) / 50