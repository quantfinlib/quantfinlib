"""Implamentation of micro_structure features from price sequences."""

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


def garman_klass_volatility(
    high: np.ndarray, low: np.ndarray, open: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    """
    Calculate the Garman-Klass volatility.

    Parameters
    ----------
    high : np.ndarray
        The high prices.
    low : np.ndarray
        The low prices.
    open : np.ndarray
        The open prices.
    close : np.ndarray
        The close prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Garman-Klass volatility.

    References
    ----------
    Volatility Modelling and Trading by Artur Sepp
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2810768
    """
    returns_sq = 0.5 * np.log(high / low) ** 2
    returns_close_open = (2 * np.log(2) - 1) * (np.log(close / open)) ** 2
    sigma_sq = moving_average(x=returns_sq, window_size=window) - moving_average(
        x=returns_close_open, window_size=window
    )
    return sigma_sq


def rogers_satchell_volatility(
    high: np.ndarray, low: np.ndarray, open: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    """
    Calculate the Rogers-Satchell volatility.
    p 20, Volatility Modelling and Trading

    Parameters
    ----------
    high : np.ndarray
        The high prices.
    low : np.ndarray
        The low prices.
    open : np.ndarray
        The open prices.
    close : np.ndarray
        The close prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Rogers-Satchell volatility.

    References
    ----------
    Volatility Modelling and Trading by Artur Sepp
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2810768
    """
    sigma_sq = moving_average(
        x=np.log(high / close) * np.log(high / open) + np.log(low / close) * np.log(low / open), window_size=window
    )
    return sigma_sq


def yang_zhang_volatility(
    high: np.ndarray, low: np.ndarray, open: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    """
    Calculate the Yang-Zhang volatility.

    Parameters
    ----------
    high : np.ndarray
        The high prices.
    low : np.ndarray
        The low prices.
    open : np.ndarray
        The open prices.
    close : np.ndarray
        The close prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Yang-Zhang volatility.

    References
    ----------
    Volatility Modelling and Trading by Artur Sepp
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2810768
    """
    coeff = 0.34 / (1.34 + (window + 1) / (window - 1))

    open_prev_close_ret = np.log(open / close.shift(1))
    close_prev_open_ret = np.log(close / open.shift(1))

    sigma_overnight_sq = moving_average(x=open_prev_close_ret**2, window_size=window)
    sigma_open_to_close_sq = moving_average(x=close_prev_open_ret**2, window_size=window)

    sigma_rs_sq = rogers_satchell_volatility(high=high, low=low, close=close, open=open, window=window)

    sigma_sq = sigma_overnight_sq + coeff * sigma_open_to_close_sq + (1 - coeff) * sigma_rs_sq
    return sigma_sq


def becker_parkinson_volatility(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """Calculate the Becker-Parkinson volatility.

    See page 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado.
    This volatility is useful in the corporate bond market, where there is no centralized order book, and
    trades occur through bids wanted in competition (BWIC).

    Parameters
    ----------
    high : np.ndarray
        The high prices.
    low : np.ndarray
        The low prices.
    close : np.ndarray
        The close prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Becker-Parkinson volatility.
    """
    beta = _get_beta(high=high, low=low, window=window)
    gamma = _get_gamma(high=high, low=low)
    sigma = _get_sigma(beta=beta, gamma=gamma)
    return sigma


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
