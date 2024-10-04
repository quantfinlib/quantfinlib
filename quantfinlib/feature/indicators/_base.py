"""Base indicator functions for time series analysis."""

import functools
import inspect
import numpy as np
import pandas as pd


def numpy_io_support(func):
    """Make a decorator to allow numpy arrays or pandas Series as input/output."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        is_numpy_input = None
        new_args = []
        param_names = list(sig.parameters.keys())  # List of parameter names in the function signature
        for i, arg in enumerate(args):
            param_name = param_names[i]
            expected_type = sig.parameters[param_name].annotation
            if expected_type in [pd.Series, np.ndarray]:
                if isinstance(arg, np.ndarray):
                    if is_numpy_input is None:
                        is_numpy_input = True
                    elif is_numpy_input is False:
                        raise ValueError("Cannot mix numpy arrays with pandas Series in input.")
                    arg = pd.Series(arg)
                elif isinstance(arg, pd.Series):
                    if is_numpy_input is None:
                        is_numpy_input = False
                    elif is_numpy_input is True:
                        raise ValueError("Cannot mix numpy arrays with pandas Series in input.")
            new_args.append(arg)
        result = func(*new_args, **kwargs)
        if is_numpy_input:
            return result.values
        return result

    return wrapper


@numpy_io_support
def rolling_mean(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling mean of a time series.

    Parameters
    ----------
    ts : pd.Series
        The input time series.
    window : int, optional
        The size of the rolling window. Default is 20.

    Returns
    -------
    pd.Series
        The rolling mean of the time series.
    """
    return ts.rolling(window=window).mean().rename(f"{window}D Mean")


@numpy_io_support
def rolling_std(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling standard deviation of a time series.

    Parameters
    ----------
    ts : pd.Series
        The input time series.
    window : int, optional
        The size of the rolling window. Default is 20.

    Returns
    -------
    pd.Series
        The rolling standard deviation of the time series.
    """
    return ts.rolling(window=window).std().rename(f"{window}D Std")


@numpy_io_support
def rolling_max(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling maximum of a time series.

    Parameters
    ----------
    ts : pd.Series
        The input time series.
    window : int, optional
        The size of the rolling window. Default is 20.

    Returns
    -------
    pd.Series
        The rolling maximum of the time series.
    """
    return ts.rolling(window=window).max().rename(f"{window}D Max")


@numpy_io_support
def rolling_min(ts: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling minimum of a time series.

    Parameters
    ----------
    ts : pd.Series
        The input time series.
    window : int, optional
        The size of the rolling window. Default is 20.

    Returns
    -------
    pd.Series
        The rolling minimum of the time series.
    """
    return ts.rolling(window=window).min().rename(f"{window}D Min")


@numpy_io_support
def ewm_mean(ts: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate the exponential weighted moving average of a time series.

    Parameters
    ----------
    ts : pd.Series
        The input time series.
    span : int, optional
        The span of the exponential window. Default is 20.

    Returns
    -------
    pd.Series
        The exponential weighted moving average of the time series.
    """
    return ts.ewm(span=span).mean().rename("EWM Mean")


@numpy_io_support
def ewm_std(ts: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate the exponential weighted moving standard deviation of a time series.

    Parameters
    ----------
    ts : pd.Series
        The input time series.
    span : int, optional
        The span of the exponential window. Default is 20.

    Returns
    -------
    pd.Series
        The exponential weighted moving standard deviation of the time series.
    """
    return ts.ewm(span=span).std().rename("EWM Std")


@numpy_io_support
def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the average true range of a stock.

    Parameters
    ----------
    high : pd.Series
        The high prices of the stock.
    low : pd.Series
        The low prices of the stock.
    close : pd.Series
        The closing prices of the stock.
    window : int, optional
        The number of periods to consider. Default is 14.

    Returns
    -------
    pd.Series
        The average true range of the stock.
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

    Parameters
    ----------
    ts : pd.Series
        The input time series.
    window : int, optional
        The size of the rolling window. Default is 20.

    Returns
    -------
    pd.Series
        The rolling momentum of the time series.
    """
    return (ts.diff(window) / window).rename(f"{window}D Momentum")


@numpy_io_support
def ewm_mom(ts: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate the exponential weighted moving momentum of a time series.

    Parameters
    ----------
    ts : pd.Series
        The input time series.
    span : int, optional
        The span of the exponential window. Default is 20.

    Returns
    -------
    pd.Series
        The exponential weighted moving momentum of the time series.
    """
    return ts.diff().ewm(span=span).mean().rename("EWM Momentum")
