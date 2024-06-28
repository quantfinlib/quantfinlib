"""
Brownian Motion simulation

Classes in this module:

BrownianMotion()
"""

__all__ = ["BrownianMotion"]


from typing import Union, Optional
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from quantfinlib.sim._base import SimHelperBase


class BrownianMotionBase(BaseEstimator, SimHelperBase):
    """Brownian Motion base class"""

    def __init__(self, drift=0.0, vol=0.1, cor=None):

        # Parameters
        self.drift = np.asarray(drift).reshape(1, -1)
        self.vol = np.asarray(vol).reshape(1, -1)

        # Private attributes
        if cor is None:
            self.cor = None
            self.L_ = None
        else:
            self.cor = np.asarray(cor)
            self.L_ = np.linalg.cholesky(self.cor)

        
    def fit(self, x: np.ndarray, dt: float):
        
        x = SimHelperBase.inspect_and_normalize_fit_args(x, dt)
        
        # changes from one row to the next
        dx = np.diff(x, axis=0)

        # Mean and standard deviation of the changes
        self.drift = np.mean(dx, axis=0, keepdims=True) / dt
        self.vol = np.std(dx, axis=0, ddof=1, keepdims=True) / np.sqrt(dt)

        # Optionally correlations if we have multiple columns
        if x.shape[1] > 1:
            self.cor = np.corrcoef(dx, rowvar=False)
            self.L_ = np.linalg.cholesky(self.cor)
        else:
            self.cor = None
            self.L_ = None

        return self

    def sim_path(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None
    ) -> np.ndarray:
    
        # Allocate storage for the simulation
        num_cols = self.drift.shape[1]
        ans = np.zeros(shape=(num_steps + 1, num_cols * num_paths))

        # set the initial value of the simulation
        SimHelperBase.set_x0(ans, x0)
        
        # Create a Generator instance with the seed
        rng = np.random.default_rng(random_state)

        # fill in Normal noise
        ans[1 : num_steps + 1, :] = rng.normal(size=(num_steps, ans.shape[1]))

        tmp = ans[1 : num_steps + 1, :]
        tmp = tmp.reshape(-1, num_paths, num_cols)

        # Optionally correlate the noise
        if self.L_ is not None:
            tmp = tmp @ self.L_.T

        # Translate the noise with drift and variance
        tmp = tmp * self.vol * dt**0.5 + self.drift * dt

        # reshape back
        ans[1 : num_steps + 1, :] = tmp.reshape(-1, num_paths * num_cols)

        # compound
        ans = np.cumsum(ans, axis=0)
        
        return ans


class BrownianMotion(BrownianMotionBase):
    """Brownian Motion simulation"""

    def __init__(self, drift=0.0, vol=0.1, cor=None):
        super().__init__(drift, vol, cor)

    def fit(
        self,
        x: Union[np.ndarray, pd.DataFrame, pd.Series],
        dt: Optional[float],
        **kwargs
    ):
        values, dt = SimHelperBase.inspect_and_normalize_fit_args(x, dt)
        BrownianMotionBase.fit(values, dt)
        return self

    def sim_path(
        self,
        x0: Optional[Union[float, np.ndarray, pd.DataFrame, pd.Series, str]] = None,
        dt: Optional[float] = None,
        num_steps: Optional[int] = 252,
        num_paths: Optional[int] = 1,
        label_start = None,
        label_freq: Optional[str] = None,
        random_state: Optional[int] = None,
        include_x0: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:

        # handle arg defaults
        x0, dt, label_start, label_freq = SimHelperBase.normalize_sim_path_args(x0, dt, label_start, label_freq)
                
        # do the sims
        ans = BrownianMotionBase.sim_path(x0, dt, num_steps, num_paths, random_state)

        # format the ans
        ans = SimHelperBase.format_ans(ans, label_start, label_freq, include_x0)

        return ans
