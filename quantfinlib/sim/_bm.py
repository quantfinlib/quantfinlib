"""
File: quantfinlib/sim/_bm.py

Description:
    Brownian Motion simulation.

Author:    Thijs van den Berg
Email:     thijs@sitmo.com
Copyright: (c) 2024 Thijs van den Berg
License:   MIT License
"""

__all__ = ["BrownianMotion"]


from typing import Optional, Union

import numpy as np
from scipy.stats import multivariate_normal, norm

from quantfinlib.sim._base import SimBase, SimNllMixin, _fill_with_correlated_noise, _to_numpy


class BrownianMotion(SimBase, SimNllMixin):
    r"""A class for simulating Brownian motion paths with given drift and volatility.

    Brownian motion is a continuous-time stochastic process used to model various random phenomena. In finance, it
    is commonly used to model the random behavior of asset prices.


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

        model = BrownianMotion(drift=-2, vol=0.7)
        paths = model.path_sample(x0=1.5, dt=1/252, num_steps=252, num_paths=10)

        fig = px.line(paths)
        fig.show()

    .. plotly::

        import plotly.express as px
        from quantfinlib.sim import BrownianMotion

        model = BrownianMotion(drift=-2, vol=0.7)
        paths = model.path_sample(x0=1.5, dt=1/252, num_steps=252, num_paths=10)

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

    Details
    -------
    The stochastic differential equation (SDE) for Brownian motion is:

    .. math::

        dX_t = \mu  dt +  \sigma  dW_t

    where:

    * :math:`dX_t` is the change in the process X at time t,
    * :math:`\mu` is the drift coefficient (annualized drift rate),
    * :math:`\sigma` is the volatility coefficient (annualized volatility rate),
    * :math:`dW_t` is a Wiener process (standard Brownian motion).

    For path simulations we use the exact solution of the discretize SDE:

    .. math::

        X[t + dt] = X[t] + \mu * dt + \mathcal{N}(0,1) * \sqrt{dt}

    where:

    * :math:`\mu` is the drift coefficient (annualized drift rate).
    * :math:`\sigma` is the volatility coefficient (annualized volatility rate).
    * :math:`\mathcal{N}(0,1)` is standard Normal distributed sample.
    * :math:`dt` the time-step size.


    Member functions
    ----------------
    """

    def __init__(self, drift=0.0, vol=0.1, cor=None):
        r"""Initializes the BrownianMotion instance with specified drift and volatility.

        Parameters
        ----------
        drift : float or array, optional
            The annualized drift rate (default is 0.0).
        vol : float or array, optional
            The annualized volatility (default is 0.1).
        cor : optional
            Correlation matrix for multivariate model (default is None, uncorrelated).

        """
        super().__init__()

        # Parameters
        self.drift = _to_numpy(drift).reshape(1, -1)
        self.vol = _to_numpy(vol).reshape(1, -1)

        # Private attributes
        if cor is None:
            self.cor = None
            self.L_ = None
        else:
            self.cor = np.asarray(cor)
            self.L_ = np.linalg.cholesky(self.cor)

        self.num_parameters_ = len(self.drift) + len(self.vol)
        if self.cor is not None:
            self.num_parameters_ += len(self.drift) * (len(self.drift) - 1) / 2

    def __repr__(self) -> str:
        """Return a string representation of the BrownianMotion instance."""
        return (
            f"BrownianMotion(drift={self.drift}, vol={self.vol}, cor={self.cor})"
        )

    def _fit_np(self, x: np.ndarray, dt: float):

        # changes from one row to the next
        dx = np.diff(x, axis=0)

        # Mean and standard deviation of the changes
        self.drift = np.mean(dx, axis=0, keepdims=True) / dt
        self.vol = np.std(dx, axis=0, ddof=1, keepdims=True) / np.sqrt(dt)

        # Optionally correlations if we have multiple columns
        if x.shape[1] > 1:
            if np.any(self.vol == 0):
                raise ValueError(
                    "Cannot compute a correlation matrix because "
                    "one or more series has zero variance in their changes."
                )
            self.cor = np.corrcoef(dx, rowvar=False)
            self.L_ = np.linalg.cholesky(self.cor)
        else:
            self.cor = None
            self.L_ = None

        return self

    def _path_sample_np(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:

        # Allocate storage for the simulation
        num_variates = self.drift.shape[1]

        # Allocate storage for the simulation
        ans = np.zeros(shape=(num_steps + 1, num_variates * num_paths))

        # set the initial value of the simulation on the first row. Tile vertical if needed
        SimBase.set_x0(ans, x0)

        # Fill a view with noise
        dx = ans[1:, :].reshape(-1, num_variates)
        _fill_with_correlated_noise(
            dx, loc=self.drift * dt, scale=self.vol * dt**0.5, L=self.L_, random_state=random_state
        )

        # compound
        ans = np.cumsum(ans, axis=0)

        return ans

    def _nll(self, x: np.ndarray, dt: float):
        dx = np.diff(x, axis=0)
        mean_ = (self.drift * dt).flatten()
        std_ = (self.vol * dt**0.5).flatten()

        if self.cor is not None:
            D = np.diag(std_)
            cov_ = D @ self.cor @ D
            var = multivariate_normal(mean=mean_, cov=cov_)
        else:
            var = norm(loc=mean_, scale=std_)
        return -1 * np.sum(var.logpdf(dx))
