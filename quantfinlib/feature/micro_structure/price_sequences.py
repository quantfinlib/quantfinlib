"""Implamentation of micro_structure features from price sequences."""

from functools import partial

import numpy as np

from quantfinlib.feature.transform.moving_window_func import moving_average, _moving_generic_1, moving_cov


def high_low_vol(p_high: np.ndarray, p_low: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the p_high-p_low volatility.

    Parameters
    ----------
    p_high : np.ndarray
        The p_high prices.
    p_low : np.ndarray
        The p_low prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The p_high-p_low volatility.

    References
    ----------
    See page 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado.
    """

    sigma_sq = moving_average(x=(np.log(p_high / p_low)) ** 2.0, window_size=window) / (4.0 * np.log(2.0))
    return sigma_sq


def cowrin_schultz_spread(p_high: np.ndarray, p_low: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Cowrin-Schultz spread.



    Parameters
    ----------
    p_high : np.ndarray
        The p_high prices.
    p_low : np.ndarray
        The p_low prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Cowrin-Schultz spread.

    References
    ----------
    See page 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado.
    """
    beta = _get_beta(p_high=p_high, p_low=p_low, window=window)
    gamma = _get_gamma(p_high=p_high, p_low=p_low)
    alpha = _get_alpha(beta=beta, gamma=gamma)
    spread = 2 * (np.exp(alpha) - 1) / (np.exp(alpha) + 1)
    return spread


def roll_spread(p_close: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Roll spread.

    Parameters
    ----------
    p_close : np.ndarray
        The p_close prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Roll spread

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3270269
    """
    logc = np.log(p_close)
    delta_logc = logc.diff()
    delta_logc_shift = np.roll(delta_logc, shift=1)
    delta_logc_shift[0] = 0
    roll = moving_cov(x=np.stack([delta_logc, delta_logc_shift], axis=1), window_size=window)
    return roll


def garman_klass_volatility(
    p_high: np.ndarray, p_low: np.ndarray, p_open: np.ndarray, p_close: np.ndarray, window: int
) -> np.ndarray:
    """
    Calculate the Garman-Klass volatility.

    Parameters
    ----------
    p_high : np.ndarray
        The p_high prices.
    p_low : np.ndarray
        The p_low prices.
    p_open : np.ndarray
        The p_open prices.
    p_close : np.ndarray
        The p_close prices.
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
    returns_sq = 0.5 * np.log(p_high / p_low) ** 2
    returns_close_open = (2 * np.log(2) - 1) * (np.log(p_close / p_open)) ** 2
    sigma_sq = moving_average(x=returns_sq, window_size=window) - moving_average(
        x=returns_close_open, window_size=window
    )
    return sigma_sq


def rogers_satchell_volatility(
    p_high: np.ndarray, p_low: np.ndarray, p_open: np.ndarray, p_close: np.ndarray, window: int
) -> np.ndarray:
    """
    Calculate the Rogers-Satchell volatility.

    Parameters
    ----------
    p_high : np.ndarray
        The p_high prices.
    p_low : np.ndarray
        The p_low prices.
    p_open : np.ndarray
        The p_open prices.
    p_close : np.ndarray
        The p_close prices.
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
        x=np.log(p_high / p_close) * np.log(p_high / p_open) + np.log(p_low / p_close) * np.log(p_low / p_open),
        window_size=window,
    )
    return sigma_sq


def yang_zhang_volatility(
    p_high: np.ndarray, p_low: np.ndarray, p_open: np.ndarray, p_close: np.ndarray, window: int
) -> np.ndarray:
    """
    Calculate the Yang-Zhang volatility.

    Parameters
    ----------
    p_high : np.ndarray
        The p_high prices.
    p_low : np.ndarray
        The p_low prices.
    p_open : np.ndarray
        The p_open prices.
    p_close : np.ndarray
        The p_close prices.
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

    open_prev_close_ret = np.log(p_open / p_close.shift(1))
    close_prev_open_ret = np.log(p_close / p_open.shift(1))

    sigma_overnight_sq = moving_average(x=open_prev_close_ret**2, window_size=window)
    sigma_open_to_close_sq = moving_average(x=close_prev_open_ret**2, window_size=window)

    sigma_rs_sq = rogers_satchell_volatility(p_high=p_high, p_low=p_low, p_close=p_close, p_open=p_open, window=window)

    sigma_sq = sigma_overnight_sq + coeff * sigma_open_to_close_sq + (1 - coeff) * sigma_rs_sq
    return sigma_sq


def becker_parkinson_volatility(p_high: np.ndarray, p_low: np.ndarray, window: int) -> np.ndarray:
    """Calculate the Becker-Parkinson volatility.

    This volatility is useful in the corporate bond market, where there is no centralized order book, and
    trades occur through bids wanted in competition (BWIC).

    Parameters
    ----------
    p_high : np.ndarray
        The p_high prices.
    p_low : np.ndarray
        The p_low prices.
    p_close : np.ndarray
        The p_close prices.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Becker-Parkinson volatility.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3270269
    See also page 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado.
    """
    beta = _get_beta(p_high=p_high, p_low=p_low, window=window)
    gamma = _get_gamma(p_high=p_high, p_low=p_low)
    sigma = _get_sigma(beta=beta, gamma=gamma)
    return sigma


def _get_gamma(p_high: np.ndarray, p_low: np.ndarray) -> np.ndarray:
    h2 = _moving_generic_1(x=p_high, window_func=partial(np.median, axis=1), window_size=2, post=f"_max{2}")
    l2 = _moving_generic_1(x=p_low, window_func=partial(np.median, axis=1), window_size=2, post=f"_min{2}")
    gamma = np.log(h2 / l2) ** 2.0
    return gamma


def _get_beta(p_high: np.ndarray, p_low: np.ndarray, window: int) -> np.ndarray:
    h1 = np.log(p_high / p_low) ** 2.0
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
