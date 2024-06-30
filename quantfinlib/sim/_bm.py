"""
Brownian Motion simulation.

Classes in this module:

BrownianMotion()
"""

__all__ = ["BrownianMotion"]


from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from quantfinlib.sim._base import SimHelperBase


class BrownianMotionBase(BaseEstimator, SimHelperBase):
    """Brownian Motion base class"""

    def __init__(self, drift=0.0, vol=0.1, cor=None):
        super().__init__()

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

        x, dt = self.inspect_and_normalize_fit_args(x, dt)

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

    def path_sample(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:

        # Allocate storage for the simulation
        num_cols = self.drift.shape[1]
        ans = np.zeros(shape=(num_steps + 1, num_cols * num_paths))

        # set the initial value of the simulation
        SimHelperBase.set_x0(ans, x0)

        # Create a Generator instance with the seed
        rng = np.random.default_rng(random_state)

        # fill in Normal noise
        ans[1:num_steps + 1, :] = rng.normal(size=(num_steps, ans.shape[1]))

        tmp = ans[1:num_steps + 1, :]
        tmp = tmp.reshape(-1, num_paths, num_cols)

        # Optionally correlate the noise
        if self.L_ is not None:
            tmp = tmp @ self.L_.T

        # Translate the noise with drift and variance
        tmp = tmp * self.vol * dt**0.5 + self.drift * dt

        # reshape back
        ans[1:num_steps + 1, :] = tmp.reshape(-1, num_paths * num_cols)

        # compound
        ans = np.cumsum(ans, axis=0)

        return ans


class BrownianMotion(BrownianMotionBase):
    r"""A class for simulating Brownian motion paths with given drift and volatility.

    Brownian motion is a continuous-time stochastic process used to model various random phenomena. In finance, it
    is commonly used to model the random behavior of asset prices.

    The stochastic differential equation (SDE) for Brownian motion is:

    .. math::

        dX_t = \mu * dt +  \sigma * dW_t

    where:

    * :math:`dX_t` is the change in the process X at time t,
    * :math:`\mu` is the drift coefficient (annualized drift rate),
    * :math:`\sigma` is the volatility coefficient (annualized volatility rate),
    * :math:`dW_t` is a Wiener process (standard Brownian motion).

    Below an example of 10 Brownian motion paths:

    Examples
    --------

    Generate 3 Brownian motion paths. All paths start at 1, and have 6 steps with
    a stepsize of dt=1/4. The drift is -2 and volatility is 0.7.

    .. exec_code::

        from quantfinlib.sim import BrownianMotion

        bm = BrownianMotion(drift=-2, vol=0.7)
        paths = bm.path_sample(x0=1, dt=1/4, num_steps=6, num_paths=3)

        print(paths)



    Below is a plot of 10 Brownian motion paths of length 252.

    .. code-block:: python


        import plotly.express as px
        from quantfinlib.sim import BrownianMotion

        bm = BrownianMotion(drift=-2, vol=0.7)
        paths = bm.path_sample(x0=1.5, dt=1/252, num_steps=252, num_paths=10)

        fig = px.line(paths)
        fig.show()

    .. plotly::

        import plotly.express as px
        from quantfinlib.sim import BrownianMotion

        bm = BrownianMotion(drift=-2, vol=0.7)
        paths = bm.path_sample(x0=1.5, dt=1/252, num_steps=252, num_paths=10)

        fig = px.line(paths)
        fig.show()


    Properties and Limitations
    --------------------------

    * Drift (:math:`\mu`) and volatility (:math:`\sigma`) are considered constant over time.
    * Simulated values can be both positive and negative, which might not be realistic for certain assets (e.g.,
      stock prices, which cannot be negative).
    * Brownian motion assumes a continuous path, which is a good approximation for high-frequency trading but may
      not capture large jumps or discontinuities in asset prices.


    Use Cases in Finance
    --------------------

    Brownian motion is widely used in financial modeling for:

    * Modeling the returns of assets: Brownian motion is often used to simulate the returns of financial assets,
      assuming that returns follow a random walk.
    * Spread modeling: It can be used to model the spread between different assets, capturing the random fluctuations
      around a mean value.
    * Risk management: Brownian motion can be used to assess the risk and uncertainty in asset price movements over
      short time intervals.


    Member functions
    ----------------
    """

    def __init__(self, drift=0.0, vol=0.1, cor=None):
        r"""Initializes the BrownianMotion instance with specified drift and volatility.

        Parameters
        ----------
        drift : float, optional
            The annualized drift rate (default is 0.0).
        vol : float, optional
            The annualized volatility rate (default is 0.1).
        cor : optional
            Correlation matrix for multivariate Brownian motion (default is None).

        """
        super().__init__(drift=drift, vol=vol, cor=cor)

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series], dt: Optional[float], **kwargs):
        r"""Calibrates the Brownian motion model to the given path(s).

        Parameters
        ----------
        x : Union[np.ndarray, pd.DataFrame, pd.Series]
            The input data for calibration.
        dt : Optional[float]
            The time step between observations.
        **kwargs
            Additional arguments for the fit process.

        Returns
        -------
        self : BrownianMotion
            The fitted model instance.

        """
        values, dt = self.inspect_and_normalize_fit_args(x, dt)
        super().fit(values, dt)
        return self

    def path_sample(
        self,
        x0: Optional[Union[float, np.ndarray, pd.DataFrame, pd.Series, str]] = None,
        dt: Optional[float] = None,
        num_steps: Optional[int] = 252,
        num_paths: Optional[int] = 1,
        label_start=None,
        label_freq: Optional[str] = None,
        columns: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        include_x0: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        r"""Simulates Brownian motion paths.

        Parameters
        ----------
        x0 : Optional[Union[float, np.ndarray, pd.DataFrame, pd.Series, str]], optional
            The initial value(s) of the paths (default is None). The strings "first" and "last"
            will set x0 to the first or last value of the datasets used in a `fit()` call.
        dt : Optional[float], optional
            The time step between observations (default is None). If None we first fall-back to
            a dt value using/derived during fitting `fit()`. If that's unavailable we default to 1/252.
        num_steps : Optional[int], optional
            The number of time steps to simulate (default is 252).
        num_paths : Optional[int], optional
            The number of paths to simulate (default is 1).
        label_start : optional, date-time like.
            The date-time start label for the simulated paths (default is None). When set this
            function will return Pandas DataFrame with a DateTime index.
        label_freq : Optional[str], optional
            The frequency for labeling the time steps (default is None).
        columns : Optional[List[str]], optional
            A list of column names.
        random_state : Optional[int], optional
            The seed for the random number generator (default is None).
        include_x0 : bool, optional
            Whether to include the initial value in the simulated paths (default is True).

        Returns
        -------
        Union[np.ndarray, pd.DataFrame, pd.Series]
            The simulated Brownian motion paths.

        """
        # handle arg defaults
        x0, dt, label_start, label_freq = self.normalize_sim_path_args(x0, dt, label_start, label_freq)

        # do the sims using the actual implementation in the base class
        ans = super().path_sample(x0, dt, num_steps, num_paths, random_state)

        # format the ans
        ans = self.format_ans(ans, label_start, label_freq, columns, include_x0, num_paths)

        return ans
