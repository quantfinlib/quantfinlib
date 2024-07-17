"""Module to provide decorators for input validation."""

from functools import wraps
import inspect
from typing import Any, Callable
from quantfinlib.util._dtypes import SeriesOrArray, DataFrameOrArray
import numpy as np
import pandas as pd


def _check_dtype_series_or_array(x: Any) -> None:
    if not isinstance(x, (pd.Series, np.ndarray)):
        raise ValueError(f"Invalid input type: {type(x).__name__}. Must be a pandas Series or a numpy array")


def _check_ndim_shape_1Darray(x: SeriesOrArray) -> None:
    if x.ndim != 1:
        raise ValueError("Input array must be 1D.")
    if x.shape[0] < 2:
        raise ValueError("Number of elements must be greater than 1.")
    

def validate_series_or_1Darray(*arg_names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Check if input is a pandas Series or a numpy array with 1 dimension."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the function signature
            sig = inspect.signature(func)
            bound_args = sig.bind_partial(*args, **kwargs).arguments

            for arg_name in arg_names:
                if arg_name in bound_args:
                    arg = bound_args[arg_name]
                    if arg is not None:  # Handle cases where arg might be None
                        _check_dtype_series_or_array(x=arg)
                        _check_ndim_shape_1Darray(x=arg)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _check_dtype_frame_or_array(X: Any) -> None:
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError(f"Invalid input type: {type(X).__name__}. Must be a pandas DataFrame or a numpy array")


def _check_ndim_shape_2Darray(X: DataFrameOrArray) -> None:
    if X.ndim != 2:
        raise ValueError("Input array must be 2D.")
    if X.shape[1] < 2:
        raise ValueError("Number of columns must be greater than 1.")


def validate_frame_or_2Darray(*arg_names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Check if input is a pandas DataFrame or a numpy array with 2 dimensions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the function signature
            sig = inspect.signature(func)
            bound_args = sig.bind_partial(*args, **kwargs).arguments

            for arg_name in arg_names:
                if arg_name in bound_args:
                    arg = bound_args[arg_name]
                    if arg is not None:  # Handle cases where arg might be None
                        _check_dtype_frame_or_array(X=arg)
                        _check_ndim_shape_2Darray(X=arg)

            return func(*args, **kwargs)

        return wrapper

    return decorator
