"""Implamentation of price sequences features."""

from functools import partial

import numpy as np

from quantfinlib.feature.transform.moving_window_func import moving_average, _moving_generic_1


def high_low_vol(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the high-low volatility.

    See page 284 of Advances in Financial Machine Learning by Marcos Lopez de Prado.

    Parameters
    ----------
    high : np.ndarray
        The high prices.
    low : np.ndarray
        The low prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The high-low volatility.
    """

    sigma_sq = moving_average(x=(np.log(high / low)) ** 2.0, window_size=window) / (4.0 * np.log(2.0))
    return sigma_sq


def cowrin_schultz_spread(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Cowrin-Schultz spread.

    See page 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado.

    Parameters
    ----------
    high : np.ndarray
        The high prices.
    low : np.ndarray
        The low prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Cowrin-Schultz spread.
    """
    beta = _get_beta(high=high, low=low, window=window)
    gamma = _get_gamma(high=high, low=low)
    alpha = _get_alpha(beta=beta, gamma=gamma)
    spread = 2 * (np.exp(alpha) - 1) / (np.exp(alpha) + 1)
    return spread


def _get_gamma(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    h2 = _moving_generic_1(x=high, window_func=partial(np.median, axis=1), window_size=2, post=f"_max{2}")
    l2 = _moving_generic_1(x=low, window_func=partial(np.median, axis=1), window_size=2, post=f"_min{2}")
    gamma = np.log(h2 / l2) ** 2.0
    return gamma


def _get_beta(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    h1 = np.log(high / low) ** 2.0
    beta = _moving_generic_1(x=h1, window_func=partial(np.sum, axis=1), window_size=2, post=f"_sum{2}")
    beta = moving_average(x=beta, window_size=window)
    return beta


def _get_alpha(beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    den = 3 - 2.0 * 2**0.5
    alpha = (2**0.5 - 1) * (beta**0.5) / den
    alpha -= -((gamma / den) ** 0.5)
    alpha[alpha < 0] = 0  # set negative alpha to zero
    return alpha


def _get_sigma(beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2**0.5
    sigma = (2**0.5 - 1) * (beta**0.5) / (k2 * den)
    sigma += (gamma / (k2 * den)) ** 0.5
    sigma[sigma < 0] = 0  # set negative sigma to zero
    return sigma
