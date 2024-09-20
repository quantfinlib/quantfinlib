"""Base indicator functions for time series analysis."""

import functools
import numpy as np
import pandas as pd
from typing import Union


def numpy_io_support(func):
    """Decorator to allow numpy arrays to be passed as input and output."""
    @functools.wraps(func)
    def wrapper(*args: Union[pd.Series, np.ndarray], **kwargs) -> Union[pd.Series, np.ndarray]:
        count_np = sum(isinstance(arg, np.ndarray) for arg in args)
        count_pd = sum(isinstance(arg, pd.Series) for arg in args)
        if count_pd + count_np != len(args):
            raise ValueError("All positional arguments must be either numpy arrays or pandas Series.")
        if count_np == 0:
            return func(*args, **kwargs)
        if count_pd == 0:
            return func(*[pd.Series(arg) for arg in args], **kwargs).values
        idx = [s.index for s in args if isinstance(s, pd.Series)][0]
        pd_args = [pd.Series(arg, index=idx) if isinstance(arg, np.ndarray) else arg for arg in args]
        return func(*pd_args, **kwargs)
    return wrapper


@numpy_io_support
def rolling_mean(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling mean of a time series.
    Parameters:
        ts (pd.Series): The input time series.
        window (int): The size of the rolling window. Default is 20.
    Returns:
        pd.Series: The rolling mean of the time series.
    """
    return ts.rolling(window=window).mean().rename(f"{window}D Mean")


@numpy_io_support
def rolling_std(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling standard deviation of a time series.
    Parameters:
        ts (pd.Series): The input time series.
        window (int): The size of the rolling window. Default is 20.
    Returns:
        pd.Series: The rolling standard deviation of the time series.
    """
    return ts.rolling(window=window).std().rename(f"{window}D Std")


@numpy_io_support
def rolling_max(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling maximum of a time series.
    Parameters:
        ts (pd.Series): The input time series.
        window (int): The size of the rolling window. Default is 20.
    Returns:
        pd.Series: The rolling maximum of the time series.
    """
    return ts.rolling(window=window).max().rename(f"{window}D Max")


@numpy_io_support
def rolling_min(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling minimum of a time series.
    Parameters:
        ts (pd.Series): The input time series.
        window (int): The size of the rolling window. Default is 20.
    Returns:
        pd.Series: The rolling minimum of the time series.
    """
    return ts.rolling(window=window).min().rename(f"{window}D Min")


@numpy_io_support
def ewm_mean(ts: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate the exponential weighted moving average of a time series.
    Parameters:
        ts (pd.Series): The input time series.
        span (int): The span of the exponential window. Default is 20.
    Returns:
        pd.Series: The exponential weighted moving average of the time series.
    """
    return ts.ewm(span=span).mean().rename("EWM Mean")


@numpy_io_support
def ewm_std(ts: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate the exponential weighted moving standard deviation of a time series.
    Parameters:
        ts (pd.Series): The input time series.
        span (int): The span of the exponential window. Default is 20.
    Returns:
        pd.Series: The exponential weighted moving standard deviation of the time series.
    """
    return ts.ewm(span=span).std().rename("EWM Std")


@numpy_io_support
def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the average true range of a stock.
    Parameters:
        high (pd.Series): The high prices of the stock.
        low (pd.Series): The low prices of the stock.
        close (pd.Series): The closing prices of the stock.
        window (int): The number of periods to consider. Default is 14.
    Returns:
        pd.Series: The average true range of the stock.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window).mean().rename("ATR")


@numpy_io_support
def rolling_mom(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling momentum of a time series.
    Parameters:
        ts (pd.Series): The input time series.
        window (int): The size of the rolling window. Default is 20.
    Returns:
        pd.Series: The rolling momentum of the time series.
    """
    return (ts.diff(window) / window).rename(f"{window}D Momentum")


@numpy_io_support
def ewm_mom(ts: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate the exponential weighted moving momentum of a time series.
    Parameters:
        ts (pd.Series): The input time series.
        span (int): The span of the exponential window. Default is 20.
    Returns:
        pd.Series: The exponential weighted moving momentum of the time series.
    """
    return ts.diff().ewm(span=span).mean().rename("EWM Momentum")
