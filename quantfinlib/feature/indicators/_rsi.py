"""Functions to calculate the Relative Strength Index (RSI) of a stock."""

import pandas as pd

from ._base import numpy_io_support


@numpy_io_support
def rsi(ts: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) of a stock.

    Args:
        ts (pd.Series): Time series of stock prices.
        period (int): Number of periods to consider for RSI calculation. Default is 14.

    Returns:
        pd.Series: The calculated RSI values.
    """
    delta = ts.diff()
    up = delta.where(delta > 0, 0).rolling(window=period).mean()
    down = (-delta).where(delta < 0, 0).rolling(window=period).mean()
    return 100.0 * (up / (up + down))


@numpy_io_support
def ewm_rsi(ts: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) using Exponential Weighted Moving Average (EWMA).

    Args:
        ts (pd.Series): Time series of stock prices.
        period (int): Number of periods to consider for RSI calculation. Default is 14.

    Returns:
        pd.Series: The calculated EWMA RSI values.
    """
    delta = ts.diff()
    up = delta.where(delta > 0, 0).ewm(span=period).mean()
    down = (-delta).where(delta < 0, 0).ewm(span=period).mean()
    return 100.0 * (up / (up + down))
