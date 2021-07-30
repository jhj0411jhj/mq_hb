import numpy as np
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.constants import VERY_SMALL_NUMBER
from openbox.utils.util_funcs import get_types


class GaussianProcess_BoTorch(AbstractModel):
    def __init__(self, config_space, standardize_y=False, **kwargs):
        types, bounds = get_types(config_space)
        # resource feature
        types = np.hstack((types, [0])).astype(int)
        bounds = np.vstack((bounds, [[0.0, 1.0]])).astype(float)
        super().__init__(types=types, bounds=bounds, **kwargs)

        self.tkwargs = {
            # use torch.float32 instead of torch.double
            # prevent gpytorch.utils.errors.NotPSDError in gp.posterior(), see 'gpytorch/utils/cholesky.py'
            "dtype": torch.float32,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self.standardize_y = standardize_y
        if self.standardize_y:
            self.outcome_transform = Standardize(m=1)  # single objective
        else:
            self.outcome_transform = None
        self.gp = None

    def _train(self, X: np.ndarray, y: np.ndarray):
        y = y.reshape(-1, 1)
        assert X.shape[0] == y.shape[0]
        train_X = torch.tensor(X, **self.tkwargs)
        train_Y = torch.tensor(y, **self.tkwargs)

        self.gp = SingleTaskGP(train_X, train_Y, outcome_transform=self.outcome_transform)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)

    def _predict(self, X: np.ndarray):
        test_X = torch.tensor(X, **self.tkwargs)
        posterior = self.gp.posterior(test_X)
        mu = posterior.mean.cpu().detach().numpy().astype(np.float64)
        var = posterior.variance.cpu().detach().numpy().astype(np.float64)
        var = np.clip(var, VERY_SMALL_NUMBER, np.inf)
        return mu, var
