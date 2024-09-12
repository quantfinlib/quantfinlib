"""Moving window helper functions."""

from functools import partial
from typing import Callable
import numpy as np

from quantfinlib.util.convert import type_to_np, np_to_type


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
    return _moving_generic_1(x, partial(np.mean, axis=1), window_size, f'_average{window_size}')
