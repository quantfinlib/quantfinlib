"""
Brownian Motion simulation

Classes in this module:

BrownianMotion()
"""

__all__ = ["BrownianMotion"]

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BrownianMotion(BaseEstimator):
    """Brownian Motion simulation"""

    def __init__(self, drift=0.0, vol=0.1, cor=None):
        # Parameters
        self.drift = drift
        self.vol = vol
        self.cor = None

        # Private attributes
        self.x0_ = 0
        self.dt_ = 1.0 / 252.0
        self.num_samples_ = 253

        if self.cor is None:
            self.L_ = None
        else:
            self.L_ = np.linalg.cholesky(self.cor)

    def fit(
        self,
        x: Union[np.ndarray, pd.DataFrame, pd.Series],
        dt: Optional[float],
        **kwargs
    ):

        # Make sure that x is a 2d array
        x_ = np.asarray(x)
        if x_.ndim == 1:
            x_ = x_.reshape(-1, 1)

        # Initial value
        self.dt_ = dt
        self.x0_ = x_[0, :].reshape(1, -1)
        self.num_samples_ = x_.shape[0]

        # changes from one row to the next
        dx = np.diff(x_, axis=0)

        # Mean and standard deviation of the changes
        self.drift = np.mean(dx, axis=0, keepdims=True) / self.dt_
        self.vol = np.std(dx, axis=0, ddof=1, keepdims=True) / np.sqr(self.dt_)

        # Optionally correlations if we have multiple columns
        if x.shape[1] > 1:
            self.cor = np.corrcoef(dx, rowvar=False)
            self.L_ = np.linalg.cholesky(self.cor)
        else:
            self.cor = None
            self.L_ = None

        return self

    def sample(
        self,
        x0: Union[float, np.ndarray, pd.DataFrame, pd.Series, None],
        dt: Optional[float],
        num_steps: Optional[int],
        random_state: Optional[int] = None,
        include_x0: bool = True,
    ) -> np.array:

        # Handle default values
        if x0 is None:
            x0 = self.x0_
        if dt is None:
            dt = self.dt_
        if num_steps is None:
            num_steps = self.num_steps_

        # Answer
        ans = np.zeros(shape=(num_steps + 1, self.mean.shape[1]))

        # set the initial value
        ans[0, :] = self.x0

        # Create a Generator instance with the seed
        rng = np.random.default_rng(random_state)

        # fill in Normal noise
        ans[1 : num_steps + 1, :] = rng.normal(size=(num_steps, self.drift.shape[1]))

        # Optionally correlate the noise
        if self.L_ is not None:
            ans[1 : num_steps + 1, :] = ans[1 : num_steps + 1, :] @ self.L_.T

        # Translate the noise with mean and variance
        ans[1 : num_steps + 1, :] = (
            ans[1 : num_steps + 1, :] * self.vol * dt**0.5 + self.drift * dt
        )

        # compound
        ans = np.cumsum(ans, axis=0)

        if include_x0:
            return ans
        else:
            return ans[1:, :]

        return ans
