"""Functions to calculate MACD indicator."""

import pandas as pd

from ._base import numpy_io_support


@numpy_io_support
def macd(ts: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator of a stock.

    Parameters
    ----------
    ts : pd.Series
        Time series of stock prices.
    fast : int, optional
        Number of periods for the fast moving average. Default is 12.
    slow : int, optional
        Number of periods for the slow moving average. Default is 26.

    Returns
    -------
    pd.Series
        The calculated MACD values.
    """
    ema_fast = ts.ewm(span=fast).mean()
    ema_slow = ts.ewm(span=slow).mean()
    return ema_fast - ema_slow


@numpy_io_support
def macd_signal(ts: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    Calculate the Moving Average Convergence Divergence (MACD) signal of a stock.

    Parameters
    ----------
    ts : pd.Series
        Time series of stock prices.
    fast : int, optional
        Number of periods for the fast moving average. Default is 12.
    slow : int, optional
        Number of periods for the slow moving average. Default is 26.
    signal : int, optional
        Number of periods for the signal line. Default is 9.

    Returns
    -------
    pd.Series
        The calculated MACD signal values.
    """
    macd_line = macd(ts, fast=fast, slow=slow)
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line
