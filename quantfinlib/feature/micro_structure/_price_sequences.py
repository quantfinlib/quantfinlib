"""Implamentation of micro_structure features from price sequences."""

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from quantfinlib.feature.indicators._base import numpy_io_support


@numpy_io_support
def get_close_close_volatility(p_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate the close-close volatility.

    Parameters
    ----------
    p_close : pd.Series
        The close prices.
    window : int
        The window size.

    Returns
    -------
    pd.Series
        The close-close volatility.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2810768.
    """
    sigma_sq = (p_close / p_close.shift(1)).rolling(window=window).var()
    return sigma_sq


@numpy_io_support
def get_high_low_volatility(p_high: pd.Series, p_low: pd.Series, window: int) -> pd.Series:
    """
    Calculate the High-low volatility.

    Parameters
    ----------
    p_high : pd.Series
        The high prices.
    p_low : pd.Series
        The low prices.
    window : int
        The window size.

    Returns
    -------
    pd.Series
        The p_high-p_low volatility.

    References
    ----------
    See page 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado.
    """
    sigma_sq = (np.log(p_high / p_low) ** 2.0).rolling(window=window).mean() / (4.0 * np.log(2.0))
    return sigma_sq


@numpy_io_support
def get_garman_klass_volatility(
    p_high: pd.Series, p_low: pd.Series, p_open: pd.Series, p_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Garman-Klass volatility from high, low, open, and close prices.

    Parameters
    ----------
    p_high : pd.Series
        The high prices.
    p_low : pd.Series
        The low prices.
    p_open : pd.Series
        The open prices.
    p_close : pd.Series
        The close prices.
    window : int
        The window size.

    Returns
    -------
    pd.Series
        The Garman-Klass volatility.

    References
    ----------
    Volatility Modelling and Trading by Artur Sepp
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2810768
    """
    returns_sq = 0.5 * np.log(p_high / p_low) ** 2
    returns_close_open = (2 * np.log(2) - 1) * (np.log(p_close / p_open)) ** 2
    sigma_sq = returns_sq.rolling(window=window).mean() - returns_close_open.rolling(window=window).mean()
    return sigma_sq


@numpy_io_support
def get_rogers_satchell_volatility(
    p_high: pd.Series, p_low: pd.Series, p_open: pd.Series, p_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Rogers-Satchell volatility from high, low, open, and close prices..

    Parameters
    ----------
    p_high : pd.Series
        The high prices.
    p_low : pd.Series
        The low prices.
    p_open : pd.Series
        The open prices.
    p_close : pd.Series
        The close prices.
    window : int
        The window size.

    Returns
    -------
    pd.Series
        The Rogers-Satchell volatility.

    References
    ----------
    Volatility Modelling and Trading by Artur Sepp
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2810768
    """
    series = np.log(p_high / p_close) * np.log(p_high / p_open) + np.log(p_low / p_close) * np.log(p_low / p_open)
    sigma_sq = series.rolling(window=window).mean()
    return sigma_sq


@numpy_io_support
def get_yang_zhang_volatility(
    p_high: pd.Series, p_low: pd.Series, p_open: pd.Series, p_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Yang-Zhang volatility from high, low, open, and close prices.

    Parameters
    ----------
    p_high : pd.Series
        The high prices.
    p_low : pd.Series
        The low prices.
    p_open : pd.Series
        The open prices.
    p_close : pd.Series
        The close prices.
    window : int
        The window size.

    Returns
    -------
    pd.Series
        The Yang-Zhang volatility.

    References
    ----------
    Volatility Modelling and Trading by Artur Sepp
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2810768
    """
    coeff = 0.34 / (1.34 + (window + 1) / (window - 1))
    open_prev_close_ret = np.log(p_open / p_close.shift(1))
    close_prev_open_ret = np.log(p_close / p_open.shift(1))
    sigma_overnight_sq = (open_prev_close_ret**2.0).rolling(window=window).mean()
    sigma_open_to_close_sq = (close_prev_open_ret**2.0).rolling(window=window).mean()
    sigma_rs_sq = get_rogers_satchell_volatility(
        p_high=p_high, p_low=p_low, p_close=p_close, p_open=p_open, window=window
    )
    sigma_sq = sigma_overnight_sq + coeff * sigma_open_to_close_sq + (1 - coeff) * sigma_rs_sq
    return sigma_sq


@numpy_io_support
def get_becker_parkinson_volatility(p_high: pd.Series, p_low: pd.Series, window: int) -> pd.Series:
    """Calculate the Becker-Parkinson volatility from high low prices.

    This volatility is useful in the corporate bond market,
    where there is no centralized order book, and
    trades occur through bids wanted in competition (BWIC).

    Parameters
    ----------
    p_high : pd.Series
        High prices.
    p_low : pd.Series
        Low prices.
    window : int
        Window size.

    Returns
    -------
    pd.Series
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


@numpy_io_support
def get_cowrin_schultz_spread(p_high: pd.Series, p_low: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Cowrin-Schultz spread from high low prices.

    Parameters
    ----------
    p_high : pd.Series
        The high prices.
    p_low : pd.Series
        The low prices.
    window : int
        The window size.

    Returns
    -------
    pd.Series
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


@numpy_io_support
def get_edge_spread(
    p_high: pd.Series, p_low: pd.Series, p_open: pd.Series, p_close: pd.Series, window: int, sign: bool = False
) -> pd.Series:
    """
    Estimate the moving average of edge bid-ask spread from open, high, low, and close prices.

    Implementation of an efficient estimator of bid-ask spreads from open, high,
    low, and close prices as described in Ardia, Guidotti, & Kroencke (2024).
    Prices must be sorted in ascending order of the timestamp.

    Parameters
    ----------
    p_high : pd.Series
        The high prices.
    p_low : pd.Series
        The low prices.
    p_open : pd.Series
        The open prices.
    p_close : pd.Series
        The close prices.
    window : int
        The window size.
    sign : bool
        Whether signed estimates should be returned.


    Returns
    -------
    pd.Series
        The moving average of edge spread.

    References
    ----------
    https://doi.org/10.1016/j.jfineco.2024.103916
    https://github.com/eguidotti/bidask
    """
    price_df = pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close})

    # Define a helper function to calculate the edge spread
    def _apply_edge_spread(df):  # pragma: no cover
        return _edge_spread(p_high=df["high"], p_low=df["low"], p_open=df["open"], p_close=df["close"], sign=sign)

    # Calculate the rolling spread
    rolling_spread = Parallel(n_jobs=-1)(
        delayed(_apply_edge_spread)(price_df.iloc[i : i + window]) for i in range(len(price_df) - window + 1)
    )
    rolling_spread = pd.Series(rolling_spread, index=price_df.index[window - 1 :])

    rolling_spread = rolling_spread.reindex(price_df.index, fill_value=np.nan)
    return rolling_spread


def _edge_spread(
    p_high: pd.Series, p_low: pd.Series, p_open: pd.Series, p_close: pd.Series, sign: bool = False
) -> float:
    """
    Estimate Bid-Ask Spreads from Open, High, Low, and Close Prices.

    Implementation of an efficient estimator of bid-ask spreads from open, high,
    low, and close prices as described in Ardia, Guidotti, & Kroencke (2024).
    Prices must be sorted in ascending order of the timestamp.

    Parameters
    ----------
    p_high : pd.Series
        The high prices.
    p_low : pd.Series
        The low prices.
    p_open : pd.Series
        The open prices.
    p_close : pd.Series
        The close prices.
    sign : bool
        Whether signed estimates should be returned.

    Returns
    -------
    The spread estimate. A value of 0.01 corresponds to a spread of 1%.

    References
    ----------
    https://doi.org/10.1016/j.jfineco.2024.103916
    https://github.com/eguidotti/bidask
    """
    p_o = np.log(p_open.values)
    p_h = np.log(p_high.values)
    p_l = np.log(p_low.values)
    p_c = np.log(p_close.values)
    p_m = (p_h + p_l) / 2.0

    h1, l1, c1, m1 = p_h[:-1], p_l[:-1], p_c[:-1], p_m[:-1]
    p_o, p_h, p_l, p_c, p_m = p_o[1:], p_h[1:], p_l[1:], p_c[1:], p_m[1:]

    tau = (p_h != p_l) | (p_l != c1)
    phi1 = (p_o != p_h) & tau
    phi2 = (p_o != p_l) & tau
    phi3 = (c1 != h1) & tau
    phi4 = (c1 != l1) & tau

    pt = tau.mean()
    po = phi1.mean() + phi2.mean()
    pc = phi3.mean() + phi4.mean()

    if pt == 0 or po == 0 or pc == 0:
        return np.nan

    r1 = p_m - p_o
    r2 = p_o - m1
    r3 = p_m - c1
    r4 = c1 - m1
    r5 = p_o - c1

    d1 = r1 - tau * r1.mean() / pt
    d3 = r3 - tau * r3.mean() / pt
    d5 = r5 - tau * r5.mean() / pt

    x1 = -4.0 / po * d1 * r2 - 4.0 / pc * d3 * r4
    x2 = -4.0 / po * d1 * r5 - 4.0 / pc * d5 * r4

    e1 = x1.mean()
    e2 = x2.mean()

    v1 = x1.var()
    v2 = x2.var()

    s2 = (v2 * e1 + v1 * e2) / (v1 + v2)

    s = np.sqrt(np.abs(s2))
    if sign and s2 < 0:
        s = -s

    return float(s)


@numpy_io_support
def get_roll_measure(p_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Roll measure.

    Parameters
    ----------
    p_close : pd.Series
        The close prices.
    window : int
        The window size.

    Returns
    -------
    pd.Series
        The Roll spread.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3270269
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3345183
    """
    roll = _roll_measure(p_close=p_close, window=window)
    return roll


def _roll_measure(p_close: pd.Series, window: int) -> pd.Series:
    """Calculate the Roll measure."""
    logc = np.log(p_close)
    delta_logc = logc.diff()
    delta_logc_shift = delta_logc.shift(1)
    roll = 2 * np.sqrt(delta_logc.rolling(window=window).cov(delta_logc_shift).abs())
    return roll


@numpy_io_support
def get_roll_impact(p_close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Roll impact.

    It is defined as roll measure divided by the dollar value traded over a certain period.

    Parameters
    ----------
    p_close : pd.Series
        The close prices.
    volume : pd.Series
        The volume (trade volume).
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Roll impact.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3270269
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3345183
    """
    roll = _roll_measure(p_close=p_close, window=window)
    return roll / (volume * p_close)


# Helper functions for calulating the Cowrin-Schultz spread and Becker-Parkinson volatility.


def _get_gamma(p_high: pd.Series, p_low: pd.Series) -> pd.Series:
    """See p. 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado."""
    h2 = p_high.rolling(window=2).max()
    l2 = p_low.rolling(window=2).min()
    gamma = np.log(h2 / l2) ** 2.0
    return gamma


def _get_beta(p_high: pd.Series, p_low: pd.Series, window: int) -> pd.Series:
    """See p. 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado."""
    hl = np.log(p_high / p_low) ** 2.0
    beta = hl.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """See p. 285 of Advances in Financial Machine Learning by Marcos Lopez de Prado."""
    den = 3 - 2.0 * 2**0.5
    alpha = (2**0.5 - 1) * (beta**0.5) / den
    alpha -= -((gamma / den) ** 0.5)
    alpha[alpha < 0] = 0  # set negative alpha to zero
    return alpha


def _get_sigma(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """See p. 286 of Advances in Financial Machine Learning by Marcos Lopez de Prado."""
    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2**0.5
    sigma = (2 ** (-0.5) - 1) * (beta**0.5) / (k2 * den)
    sigma += (gamma / (k2**2 * den)) ** 0.5
    sigma[sigma < 0] = 0  # set negative sigma to zero
    return sigma
