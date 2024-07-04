"""
Geometirc Brownian Motion simulation.

Classes in this module:

GeometricBrownianMotion()
"""

__all__ = ["GeometricBrownianMotion"]


from typing import Optional, Union

import numpy as np

from quantfinlib.sim._base import SimBase, _fill_with_correlated_noise, _to_numpy


class GeometricBrownianMotion(SimBase):
    r"""A class for simulating geometric Brownian motion paths with given drift and volatility.

    Geometric Brownian motion is a continuous-time stochastic process used to model various random phenomena.
    In finance, it is the most widely used model for the random behavior of stock prices and
    other time series that have a asset prices that are strick positive.


    Below an example of 10 geometric Brownian motion paths:

    Examples
    --------

    Generate 10 geometric Brownian motion paths. All paths start at 100, and have 252 steps with
    a business-day stepsize (dt=1/12). The drift is 0.05 and volatility is 0.30.

    .. exec_code::

        from quantfinlib.sim import GeometricBrownianMotion

        model = GeometricBrownianMotion(drift=0.05, vol=0.30)
        paths = model.path_sample(
            x0=100,
            label_start='2020-01-01',
            label_freq='B',
            num_steps=252,
            num_paths=10
        )

        print(paths)


    .. plotly::

        import plotly.express as px
        from quantfinlib.sim import GeometricBrownianMotion

        model = GeometricBrownianMotion(drift=0.05, vol=0.30)
        paths = model.path_sample(
            x0=100,
            label_start='2020-01-01',
            label_freq='B',
            num_steps=252,
            num_paths=10
        )

        fig = px.line(paths)
        fig.show()


    Properties and Limitations
    --------------------------

    * Drift (:math:`\mu`) and volatility (:math:`\sigma`) are considered constant over time.
    * Simulated values are always positive.
    * Geometric Brownian motion assumes a continuous path.


    Use Cases in Finance
    --------------------

    Geometric Brownian motion is widely used in financial modeling for modeling the random
    prices of stocks, futures, FX rates. Different asset type have different drift values.
    The next table show what drift value to use for what asset types. These drift values
    are based on well known no-arbitrage principles.

    =======================  =====
    Assert Type              Drift
    =======================  =====
    Stocks                   Risk free interest rate (continuouly compounded).
                             (Black & Scholes model)
    Futures and Forwards     Zero.
                             (Black 1976 model)
    Currency exchange rates  Domestic - foreign interest rate (continuouly compounded).
                             (Garman Kohlhagen model)
    =======================  =====


    Details
    -------
    The stochastic differential equation (SDE) for geometric Brownian motion is:

    .. math::

        dX_t = \mu X_t  dt +  \sigma X_t  dW_t

    where:

    * :math:`dX_t` is the change in the process X at time t.
    * :math:`\mu` is the drift coefficient (annualized drift rate).
    * :math:`\sigma` is the volatility coefficient (annualized volatility rate).
    * :math:`dW_t` is a Wiener process (standard Brownian motion).

    For path simulations we use the exact solution of the discretize SDE:

    .. math::

        X[t + dt] = X[t] \exp\left( (\mu - \frac{1}{2} \sigma^2 )  dt + \mathcal{N}(0,1)  \sqrt{dt} \right)

    where:

    * :math:`\mu` is the drift coefficient (annualized drift rate).
    * :math:`\sigma` is the volatility coefficient (annualized volatility rate).
    * :math:`\mathcal{N}(0,1)` is standard Normal distributed sample.
    * :math:`dt` the time-step size.


    Member functions
    ----------------
    """

    def __init__(self, drift=0.0, vol=0.1, cor=None):
        r"""Initializes the geometric BrownianMotion instance with specified drift and volatility.

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

        # Constants in the base class
        self.x0_default = 100

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

    def _fit_np(self, x: np.ndarray, dt: float):

        # changes from one row to the next
        dx = np.log(x[1:, ...] / x[:-1, ...])

        # Mean and standard deviation of the changes
        self.vol = np.std(dx, axis=0, ddof=1, keepdims=True) / np.sqrt(dt)
        self.drift = np.mean(dx, axis=0, keepdims=True) / dt

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
        SimBase.set_x0(ans, np.log(x0))

        # Fill a view with noise
        dx = ans[1:, :].reshape(-1, num_variates)
        _fill_with_correlated_noise(
            dx,
            loc=(self.drift - 0.5 * self.vol**2) * dt,
            scale=self.vol * dt**0.5,
            L=self.L_,
            random_state=random_state,
        )

        # compound
        ans = np.exp(np.cumsum(ans, axis=0))

        return ans
