"""
File: quantfinlib/sim/_ou.py

Description:
    Ornstein-Uhlenbeck process.

Author:    Thijs van den Berg
Email:     thijs@sitmo.com
Copyright: (c) 2024 Thijs van den Berg
License:   MIT License
"""

__all__ = ["OrnsteinUhlenbeck"]


import warnings
from typing import Optional, Union

import numpy as np
from scipy.stats import multivariate_normal, norm

from quantfinlib.sim._base import SimBase, SimNllMixin, _fill_with_correlated_noise, _to_numpy


class OrnsteinUhlenbeck(SimBase, SimNllMixin):
    r"""A class for simulating the mean-reverting Ornstein-Uhlenbeck process.

    The Ornstein-Uhlenbeck process is a continuous-time stochastic mean-reverting process.
    In finance, it is commonly used to model time series that tend to mean-revert to som e
    long term mean like interest rates, volatility and some commodities.


    Below an example of 10 Ornstein-Uhlenbeck paths:

    Examples
    --------

    Generate 10 Ornstein-Uhlenbeck motion paths. All paths start at 1, have a long term mean of 4,
    a mean reversion rate of 2, and a volatility of 1. We simulate 252 timesteps.

    .. exec_code::

        from quantfinlib.sim import OrnsteinUhlenbeck

        model = OrnsteinUhlenbeck(mean=4, mrr=2, vol=1)
        paths = model.path_sample(
            x0=1,
            label_start='2020-01-01',
            label_freq='B',
            num_steps=252,
            num_paths=10
        )

        print(paths)


    .. plotly::

        import plotly.express as px
        from quantfinlib.sim import OrnsteinUhlenbeck

        model = OrnsteinUhlenbeck(mean=4, mrr=2, vol=1)
        paths = model.path_sample(x0=1, label_start='2020-01-01', label_freq='B', num_steps=252, num_paths=10)

        fig = px.line(paths)
        fig.show()


    Properties and Limitations
    --------------------------

    * The mean (:math:`\mu`), mean reversion rate (:math:`\lambda`) and volatility (:math:`\sigma`) are
      considered constant over time.
    * Simulated values can be both positive and negative.
    * The Ornstein-Uhlenbeck assumes a continuous path.


    Use Cases in Finance
    --------------------

    The Ornstein-Uhlenbeck process is popular in finance:

    * Modeling short term interest rate. In this setting the model is also known as the Vasicek model.

    Details
    -------
    The stochastic differential equation (SDE) for Brownian motion is:

    .. math::

        dX_t = \Theta(\mu - X_t)  dt +  \sigma  dW_t

    where:

    * :math:`dX_t` is the change in the process X at time t,
    * :math:`\Theta` the annualized mean-reversion rate,
    * :math:`\mu` the long-term mean,
    * :math:`\sigma` is the volatility coefficient (annualized volatility rate),
    * :math:`dW_t` is a Wiener process (standard Brownian motion).

    * todo: explain that sigma is a cov when MV

    
    Simulation
    ..........

    For path simulations we use the exact solution of the discretize SDE:

    .. math::

        X[t + dt] = \mu  + (X[t] - \mu) e^{-\Theta dt} + %
        \sqrt{\frac{\sigma^2}{2\Theta}\left(1 - e^{-2\Theta dt} \right) } \mathcal{N}(0,1)

    where:

    * :math:`dt` the time-step size.

    Calibration
    ...........

    For fitting we use a linear regession based on the simulation equation

    .. math::

         X[t + dt] = a X[t] + b + c N(0,1)

    where:

    .. math::

        \begin{align}
        a &= e^{-\lambda dt} \\
        b &= \mu(1 - e^{-\lambda dt}) \\
        c &= \sqrt{\frac{\sigma^2}{2\lambda}\left(1 - e^{-2\lambda dt} \right) }
        \end{align}

    :math:`a, b` are estimated with least squares regression, and :math:`c` is estimated from the regression residuals.
    The residuals are also used to estimate the correlations if the data is multi-variate. From these estimates
    the Ornstein-Uhlenbeck parameters are derived as follows:


    .. math::

        \begin{align}
        \lambda &= -\frac{\ln(a)}{dt} \\
        \mu &= \frac{b}{1-a} \\
        \sigma &= \sqrt{ \frac{2 \lambda c^2}{1 - a^2} }
        \end{align}




    Member functions
    ----------------
    """

    def __init__(self, mean=0.0, mrr=1.0, vol=0.1, cor=None):
        r"""Initializes the BrownianMotion instance with specified drift and volatility.

        Parameters
        ----------
        mean : float, optional
            The long term mean (default is 0.0).
        mrr : float, optional
            The annualized mean reversion rat  (default is 0.1).
        vol: float, optional
            The annualized volatility (default is 0.1).
        cor : optional
            Correlation matrix for multivariate model (default is None, uncorrelated).

        """
        super().__init__()

        # Parameters
        self.mean = _to_numpy(mean).reshape(1, -1)
        self.mrr = _to_numpy(mrr).reshape(1, -1)
        self.vol = _to_numpy(vol).reshape(1, -1)

        # Private attributes
        if cor is None:
            self.cor = None
            self.L_ = None
        else:
            self.cor = np.asarray(cor)
            self.L_ = np.linalg.cholesky(self.cor)

        self.num_parameters_ = len(self.mean) + len(self.mrr) + len(self.vol)
        if self.cor is not None:
            self.num_parameters_ += len(self.mean) * (len(self.mean) - 1) / 2

    def _fit_np(self, x: np.ndarray, dt: float):

        SLOPE_TOL = 1e-8

        num_series = x.shape[1]

        # Prepare to store slopes and intercepts
        slopes = np.zeros(num_series)
        intercepts = np.zeros(num_series)
        residuals = np.zeros((x.shape[0] - 1, x.shape[1]))

        # Loop through each time series
        for i in range(num_series):
            # Extract the x and y values for the current series
            lin_x = x[:-1, i]
            lin_y = x[1:, i]

            # Add a column of ones to x to account for the intercept
            A = np.vstack([lin_x, np.ones(len(lin_x))]).T

            # Solve the least squares problem
            slope, intercept = np.linalg.lstsq(A, lin_y, rcond=None)[0]

            if slope <= SLOPE_TOL:
                warnings.warn(
                    f"Fitting column with index {i} did not give a correct value. Setting mrr to very high value.",
                    UserWarning,
                )
                slope = SLOPE_TOL
                intercept = np.mean(lin_y)

            # Store the results
            slopes[i] = slope
            intercepts[i] = intercept

            # Compute the predicted y-values
            y_pred = slope * lin_x + intercept

            # Compute and store the residuals
            residuals[:, i] = lin_y - y_pred

        # Compute correlations if we have multiple columns
        if residuals.shape[1] > 1:
            self.cor = np.corrcoef(residuals, rowvar=False)
            self.L_ = np.linalg.cholesky(self.cor)
        else:
            self.cor = None
            self.L_ = None

        # Compute OU params from slope, intercept, residual_std
        res_std = np.std(residuals, axis=0, ddof=1, keepdims=True)

        self.mrr = -np.log(slope) / dt
        self.mean = intercept / (1 - slope)
        self.vol = np.sqrt((2 * self.mrr * res_std**2) / (1 - slope**2))

        return self

    def _path_sample_np(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:

        # Handy OU constants
        mrr_factor = np.exp(-self.mrr * dt)
        mean_term = self.mean * (1 - mrr_factor)
        scale_factor = np.sqrt(self.vol**2 / (2 * self.mrr) * (1 - np.exp(-2 * self.mrr * dt)))

        # Allocate storage for the simulation
        num_variates = self.mean.shape[1]

        # Allocate storage for the simulation
        ans = np.zeros(shape=(num_steps + 1, num_variates * num_paths))

        # set the initial value of the simulation on the first row. Tile vertical if needed
        SimBase.set_x0(ans, x0)

        # Fill a view with noise
        dx = ans[1:, :].reshape(-1, num_variates)
        _fill_with_correlated_noise(dx, loc=mean_term, scale=scale_factor, L=self.L_, random_state=random_state)

        # Forward step trough OU
        mrr_factor = mrr_factor.reshape(-1)
        for i in range(ans.shape[0] - 1):
            ans[i + 1, :] += ans[i, :] * mrr_factor

        return ans

    def _nll(self, x: np.ndarray, dt: float):

        mrr_factor = np.exp(-self.mrr * dt)
        mean_ = self.mean * (1 - mrr_factor)
        std_ = np.sqrt(self.vol**2 / (2 * self.mrr) * (1 - np.exp(-2 * self.mrr * dt)))

        mean_ = mean_.flatten()
        std_ = std_.flatten()

        dx = x[1:, ...] - mrr_factor * x[:-1, ...]

        if self.cor is not None:
            D = np.diag(std_)
            cov_ = D @ self.cor @ D
            var = multivariate_normal(mean=mean_, cov=cov_)
        else:
            var = norm(loc=mean_, scale=std_)
        return -1 * np.sum(var.logpdf(dx))
