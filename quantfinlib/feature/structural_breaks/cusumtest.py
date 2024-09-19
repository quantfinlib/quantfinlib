"""Module for the Chu-Stinchcombe-White CUSUM test for structural breaks."""

import numpy as np


def chu_stinchcombe_white_cusum_test(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the Chu-Stinchcombe-White CUSUM test for structural breaks.

    We compute the standardized departure of the log-price :math: `y_t` relative to the log-price
    at :math: `y_n` (t > n) as follows:

    .. math::

        S_{n,t} = \frac{y_t - y_n}{\hat{\sigma_t} \sqrt{t - n - 1}}

    where the estimated variance :math:`\sigma_t` is calculated as:

    .. math::

        \hat{\sigma^2} = (t-1)^{-1} \sum_{i=2}^{t} (y_i - \bar{y})^2

    Under the null hypothesis H0: beta_t = 0, S_{n,t} follows a standard normal
    distribution: :math:`S_{n,t} \sim N(0, 1)`.

    The time-dependent critical value for the one-sided test is computed as:

    .. math::

        c_{\alpha}[n, t] = b_{\alpha} + \log(t - n)

    where the constant :math: `b_{\alpha}` is derived via Monte Carlo simulations
    with :math: `b_0.05 = 4.6`.

    We compute :math: `S_t` as the supremum of :math: `S_{n,t}` over a backward-shifting window,
    :math: `n ∈ [1, t]`:

    .. math::
        S_t = \sup{S_{n,t}}

    Parameters
    ----------
    y : np.ndarray
        A time series array representing the price levels.

    Returns
    -------
    S : np.ndarray
        The CUSUM test statistic for each time point t.
    crit_vals : np.ndarray
        The time-dependent critical values for the one-sided test.

    References
    ----------
    - Homm, U., & Breitung, J. (2012). Testing for Speculative Bubbles in Stock Markets:
      A Comparison of Alternative Methods. Journal of Financial Econometrics, 10(1), 198-231.
    """
    T = len(y)
    b_alpha = 4.6  # Derived via Monte Carlo for alpha=0.05
    log_y = np.log(y)  # Precompute log-prices
    # Compute the moving average standard deviation up to each time point
    squared_diffs = (log_y[1:] - log_y[:-1]) ** 2.0
    sigma_hats = np.sqrt(np.cumsum(squared_diffs) / np.arange(1, T))

    # Compute test statistic S_t and critical values
    # Initialize arrays to store the test statistic and critical values
    S = np.zeros(T)
    crit_vals = np.zeros(T)
    # Iterate over all time points
    for t in range(1, T):
        numerator = log_y[t] - log_y[:t]  # Compute the numerator of S_n_t
        denominator = sigma_hats[t - 1] * np.sqrt(np.arange(t, 0, -1))
        S_n_t = numerator / denominator  # Compute the standardized departure
        crit_vals[t] = np.sqrt(b_alpha + np.log(t - np.argmax(S_n_t)))  # Compute the critical value
        S[t] = np.max(S_n_t)
    return S, crit_vals
