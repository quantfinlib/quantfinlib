"""Functions for computing distance matrices for a table/array/dataframe of variables/time series."""

from functools import partial
from itertools import combinations
import logging
from typing import Callable, Optional
import pandas as pd
import numpy as np

from quantfinlib.distance.correlation import (
    corr_to_abs_angular_dist,
    corr_to_angular_dist,
    corr_to_squared_angular_dist,
)

from quantfinlib.distance.information import mutual_info, var_info
from quantfinlib.util import (
    validate_frame_or_2Darray,
    DataFrameOrArray,
)

logger = logging.getLogger("quantfinlib")
logger.setLevel(logging.WARNING)

CORR_TO_DIST_METHOD_MAP = {
    "angular": corr_to_angular_dist,
    "abs_angular": corr_to_abs_angular_dist,
    "squared_angular": corr_to_squared_angular_dist,
}


@validate_frame_or_2Darray("X")
def _get_info_distance_matrix(X: DataFrameOrArray, func: Callable, **kwargs) -> np.ndarray:
    """Calculate distance matrix between columns of a dataset."""
    assert func in [mutual_info, var_info], "Invalid function. Must be mutual_info or var_info."
    inp_pd_df = isinstance(X, pd.DataFrame)
    if inp_pd_df:
        X, colnames = X.values, X.columns
    _, m = X.shape
    idx = np.arange(m)
    res = np.zeros((m, m))
    for i, j in combinations(idx, 2):
        res_ij = partial(func, **kwargs)(X[:, i], X[:, j])
        res[i, j] = res_ij
        res[j, i] = res_ij
    if func == mutual_info:
        np.fill_diagonal(res, 1)
    if inp_pd_df:
        res = pd.DataFrame(res, index=colnames, columns=colnames)
    return res


def _check_info_method(method: str) -> None:
    """Check if the information method is valid."""
    assert method in ["mutual_info", "var_info"], "Invalid method. Must be one of mutual_info or var_info."


def get_info_distance_matrix(X: DataFrameOrArray, method: str = "var_info", **kwargs) -> np.ndarray:
    """Calculate distance matrix between columns of a dataset.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        The input dataset.
    method : str
        The distance calculation method. Default is 'var_info'.
    **kwargs
        Additional keyword arguments for the information calculation.

    Returns
    -------
    np.ndarray
        The distance matrix.
    """
    _check_info_method(method)
    if method == "mutual_info":
        logger.warning("Mutual information is not a metric. Consider using var_info instead.")
        return _get_info_distance_matrix(X, mutual_info, **kwargs)
    elif method == "var_info":
        return _get_info_distance_matrix(X, var_info, **kwargs)
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")


@validate_frame_or_2Darray("X")
def _calculate_correlation(X: DataFrameOrArray, corr_method: str = "pearson", **kwargs) -> pd.DataFrame:
    """Calculate correlation matrix between columns of a dataset."""
    inp_numpy_array = isinstance(X, np.ndarray)
    if inp_numpy_array:
        logger.info("Input is a NumPy array. Converting to Pandas DataFrame for correlation calculation.")
        X = pd.DataFrame(X)  # Convert NumPy array to Pandas DataFrame
    corr = X.corr(method=corr_method, **kwargs)
    return corr.values if inp_numpy_array else corr


def _check_corr_calculation_method(corr_method: str) -> None:
    """Check if the correlation method is valid."""
    assert corr_method in ["pearson", "spearman"], "Invalid correlation method. Must be pearson or spearman."


def _check_corr_to_dist_method(method: str) -> None:
    """Check if the correlation to distance method is valid."""
    assert method in [
        "angular",
        "abs_angular",
        "squared_angular",
    ], "Invalid method. Must be one of angular, abs_angular, squared_angular."


def get_corr_distance_matrix(
    X: DataFrameOrArray,
    corr_method: str = "pearson",
    method: str = "angular",
    corr_transformer: Optional[Callable] = None,
    **kwargs,
) -> np.ndarray:
    """Calculate distance matrix between columns of a dataset.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        The input dataset.
    corr_method : str, optional
        The correlation method. Default is 'pearson'.
    method : str, optional
        The distance calculation method. Default is 'angular'.
    corr_transformer : Callable, optional
        A function to transform the correlation matrix before calculating distance.
    **kwargs
        Additional keyword arguments for the correlation calculation with pandas.

    Returns
    -------
    np.ndarray
        The distance matrix.
    """
    _check_corr_calculation_method(corr_method)
    _check_corr_to_dist_method(method)
    corr = _calculate_correlation(X, corr_method, **kwargs)
    if corr_transformer is not None:
        corr = corr_transformer(corr)
    func = CORR_TO_DIST_METHOD_MAP.get(method)
    if func is None:
        raise ValueError(f"Invalid method. Must be one of {list(CORR_TO_DIST_METHOD_MAP.keys())}.")
    return func(corr)
