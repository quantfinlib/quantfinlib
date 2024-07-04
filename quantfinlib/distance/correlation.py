"""Functions to compute distance metrics from correlations."""

from typing import Callable, Union
from functools import partial

import numpy as np


def _fix_diagonal_dist(dist: np.ndarray) -> np.ndarray:
    """Set the diagonal elements of the distance matrix to zero."""
    np.fill_diagonal(dist, 0)
    return dist


def _clip_values(values: Union[float, np.ndarray], min_val: float, max_val: float) -> Union[float, np.ndarray]:
    """Clip the values to be within the range [min_val, max_val]."""
    return np.clip(values, a_min=min_val, a_max=max_val)


def _validate_correlation_matrix(corr: np.ndarray) -> None:
    """Validate the input correlation matrix."""
    assert isinstance(corr, np.ndarray), "Input must be a numpy array"
    assert corr.ndim == 2, "Input must be a 2D matrix"
    assert corr.shape[0] == corr.shape[1], "Input must be a square matrix"
    assert np.allclose(corr, corr.T), "Correlation matrix must be symmetric"
    assert np.all(np.diag(corr) == 1), "Diagonal elements of a correlation matrix must be 1"


def _convert_correlation_to_distance(
    corr: Union[float, np.ndarray], conversion_function: Callable
) -> Union[float, np.ndarray]:
    if isinstance(corr, np.ndarray):
        _validate_correlation_matrix(corr)
    corr = _clip_values(corr, -1.0, 1.0)
    dist = conversion_function(corr)
    dist = _clip_values(dist, 0.0, 1.0)
    if isinstance(corr, np.ndarray):
        return _fix_diagonal_dist(dist)
    return dist


def _conversion_function(x: Union[float, np.ndarray], conversion_method: str) -> Union[float, np.ndarray]:
    if conversion_method == "angular":
        return np.sqrt((1.0 - x) / 2.0)
    elif conversion_method == "abs_angular":
        return np.sqrt(1.0 - np.abs(x))
    elif conversion_method == "squared_angular":
        return np.sqrt(1.0 - x**2)
    else:
        raise NotImplementedError(f"Conversion method {conversion_method} is not implemented.")


def corr_to_angular_dist(corr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Convert a correlation to angular distance.

    .. math::
        \text{angular distance} = \sqrt{\frac{1 - \text{corr}}{2}}

    Parameters
    ----------
    corr : np.ndarray or float
        The input correlation matrix or value.

    Returns
    -------
    np.ndarray or float
        The angular distance matrix or value.
    """
    return _convert_correlation_to_distance(corr, partial(_conversion_function, conversion_method="angular"))


def corr_to_abs_angular_dist(corr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Convert correlation to an absolute angular distance.

    .. math::
        \text{abs angular distance} = \sqrt{1 - \text{abs}(\text{corr})}

    Parameters
    ----------
    corr : np.ndarray or float
        The input correlation matrix or value.

    Returns
    -------
    np.ndarray or float
        The absolute angular distance matrix or value.
    """
    return _convert_correlation_to_distance(corr, partial(_conversion_function, conversion_method="abs_angular"))


def corr_to_squared_angular_dist(corr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Convert correlation to squared angular distance.

    .. math::
        \text{squared angular distance} = \sqrt{1 - \text{corr}^{2}}

    Parameters
    ----------
    corr : np.ndarray or float
        The input correlation matrix or value.

    Returns
    -------
    np.ndarray or float
        The squared angular distance matrix or value.
    """
    return _convert_correlation_to_distance(corr, partial(_conversion_function, conversion_method="squared_angular"))


def pair_angular_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the angular distance between two vectors x and y.

    Parameters
    ----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.

    Returns
    -------
    float
        The angular distance between x and y.
    """
    corr = np.corrcoef(x, y)[0, 1]
    return corr_to_angular_dist(corr)


def pair_abs_angular_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the absolute angular distance between two vectors x and y.

    Parameters
    ----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.

    Returns
    -------
    float
        The absolute angular distance between x and y.
    """
    corr = np.corrcoef(x, y)[0, 1]
    return corr_to_abs_angular_dist(corr)


def pair_squared_angular_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the squared angular distance between two vectors x and y.

    Parameters
    ----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.

    Returns
    -------
    float
        The squared angular distance between x and y.
    """
    corr = np.corrcoef(x, y)[0, 1]
    return corr_to_squared_angular_dist(corr)
