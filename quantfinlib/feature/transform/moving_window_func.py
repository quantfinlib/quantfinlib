"""Moving window helper functions."""

from functools import partial
from typing import Callable
import numpy as np
import pandas as pd

from quantfinlib.util.convert import type_to_np, np_to_type
from quantfinlib.util._inspect import _num_columns


def _moving_window_func_c1(x: np.ndarray, window_func: Callable, window_size: int = 30) -> np.ndarray:
    # a windows size of 1 might give trouble, we might investigate this later, for now not
    assert window_size > 1

    # windows size must be smaller than the data size,
    # instead of an alert we can also "return x * np.NaN"
    assert window_size <= len(x)

    # we only support vectors and matrices
    assert x.ndim in (1, 2)

    # save the shape of x, we will return something of this shape at the end
    x_shape = x.shape

    # vectors are converted to a 1 col matrix. Everything below assumed that x is a matrix
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # the matrix dimensions
    num_rows = x.shape[0]
    num_cols = x.shape[1]

    assert window_size <= num_rows

    # create a 3D view: [num_rows - window_size + 1, num_cols, window_size]
    y = np.lib.stride_tricks.sliding_window_view(x, window_shape=(window_size, 1)).squeeze(axis=3)

    # stack all the window
    y = y.reshape(-1, window_size)

    # apply the function [n, window_size] -> [n]
    y = window_func(y)

    # reshape the function output back to
    y = y.reshape(num_rows - window_size + 1, num_cols)

    ans = np.empty_like(x, dtype=float)
    ans[: window_size - 1, :] = np.NaN
    ans[window_size - 1 :, :] = y
    return ans.reshape(x_shape)


def _moving_window_func_cn(x: np.ndarray, window_func: Callable, window_size: int = 30, transpose: bool = False):
    # trivial checks
    assert window_size > 1
    assert window_size <= len(x)
    assert x.ndim in (1, 2)
    # vectors are converted to a 1 col matirx. Everything below assumed that x is a matrix
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    num_rows = x.shape[0]
    # create a 3D view: [num_rows - window_size + 1, num_cols, window_size]
    y = np.lib.stride_tricks.sliding_window_view(x, window_shape=(window_size, 1)).squeeze(axis=3)
    ans = None
    # loop over all windows
    for i in range(y.shape[0]):
        # call the transformation function
        if transpose:
            z = window_func(y[i, ...].T)
        else:
            z = window_func(y[i, ...])
        # flatten the transformation output to a vector
        z = z.flatten()
        # allocate return data if we had not already
        if ans is None:
            ans = np.empty((num_rows, len(z)), dtype=float)
            ans[: window_size - 1, :] = np.NaN
        # store the window result in the output matrix
        ans[i + window_size - 1, :] = z
    return ans


def _moving_generic_1(x, window_func, window_size=30, post=""):
    return np_to_type(_moving_window_func_c1(type_to_np(x), window_func, window_size=window_size), x, post=post)


def moving_average(x, window_size=30):
    r"""Compute the windowed moving average.

    Moving average computes the sliding windows average of a time series.

    .. math::

        y_i &= \text{mean}(x_{i,w}) \\
        &\text{with} \\
        x_{i,w} &= \{ \overbrace{ x_i, x_{i-1}, \cdots, x_{i-w+1} }^{w\text{ elements}} \}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from quantfinlib.feature.transform import moving_average

        np.random.seed(42)
        x = np.random.lognormal(mean=0.1, sigma=0.1, size=1000)


        plt.plot(x, 'k', alpha=.5, label='signal')
        plt.plot(moving_average(x, 20), 'orangered', label='20 day moving avg.')
        plt.plot(moving_average(x, 60), 'dodgerblue', label='60 day moving avg.')
        plt.legend()
        plt.grid()
        plt.show()

    Parameters
    ----------
    x : array-like
        Input.
    window_size : int, default=30
        Window size,

    Returns
    -------
    array-like
        Moving averages.
    """
    return _moving_generic_1(x, partial(np.mean, axis=1), window_size, f"_average{window_size}")


def moving_cov(x, window_size=30):
    r"""Compute the windowed moving covariance.

    Parameters
    ----------
    x : array-like
        Input 2d array, with individual time series in the columns.
    window_size : int, default=30
        Window size.

    Returns
    -------
    array-like
        Sliding window covariance. Returns the elements of the upper
        triangular part of the covariance matrix.
    """
    assert _num_columns(x) > 0
    columns = None
    if isinstance(x, pd.DataFrame):
        ci, cj = np.triu_indices(x.shape[1])
        columns = [f"{x.columns[i]}_{x.columns[j]}" for i, j in zip(ci, cj)]
    return np_to_type(
        _moving_window_func_cn(
            type_to_np(x),
            lambda a: np.atleast_2d(np.cov(a, rowvar=False))[np.triu_indices(a.shape[0])],
            window_size=window_size,
        ),
        x,
        columns=columns,
        post=f"_cov{window_size}",
    )
