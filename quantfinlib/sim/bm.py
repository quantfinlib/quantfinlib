from typing import Union, Optional
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np


def _as_row(x):
    return np.asarray(x).reshape(1, -1)


class BrownianMotion(BaseEstimator):
    """Brownian Motion simulation"""

    def __init__(self, x0=0.0, mean=0.0, std=0.01, cor=None, num_samples=252):
        # Parameters
        self.x0 = x0
        self.mean = mean
        self.std = std
        self.cor = None
        self.num_samples = None

        # Private attributes
        if self.cor is None:
            self.L_ = None
        else:
            self.L_ = np.linalg.cholesky(self.cor)

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series], **kwargs):

        # Make sure that x is a 2d array
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # num samples
        self.num_samples = x.shape[0]

        # Initial value
        self.x0 = x[0, :].reshape(1, -1)

        # changes from one row to the next
        dx = np.diff(x, axis=0)

        # Mean and standard deviation of the changes
        self.mean = np.mean(dx, axis=0, keepdims=True)
        self.std = np.std(dx, axis=0, ddof=1, keepdims=True)

        # Optionally correlations if we have multiple columns
        if x.shape[1] > 1:
            self.cor = np.corrcoef(dx, rowvar=False)
            self.L_ = np.linalg.cholesky(self.cor)
        else:
            self.cor = None
            self.L_ = None

        return self

    def sample(
        self, num_steps: Optional[int], random_state: Optional[int] = None
    ) -> np.array:

        # The number of time steps (rows)
        if num_steps is None:
            num_steps = self.num_steps

        # Answer
        ans = np.zeros(shape=(num_steps, self.mean.shape[1]))

        # set the initial value
        ans[0, :] = self.x0

        # Create a Generator instance with the seed
        rng = np.random.default_rng(random_state)

        # fill in Normal noise
        ans[1:num_steps, :] = rng.normal(size=(num_steps - 1, self.mean.shape[1]))

        # Optionally correlate the noise
        if self.L_ is not None:
            ans[1:num_steps, :] = ans[1:num_steps, :] @ self.L_.T

        # Translate the noise with mean and variance
        ans[1:num_steps, :] = ans[1:num_steps, :] * _as_row(self.std) + _as_row(
            self.mean
        )

        return ans
