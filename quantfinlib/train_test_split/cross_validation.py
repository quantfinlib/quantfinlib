"""Base Purged and Embargoed Cross-Validation Functions."""

from numbers import Integral
from typing import Union

import numpy as np
import pandas as pd


def _remove_overlapping_groups(
    train_index: np.ndarray, test_index: np.ndarray, indices: np.ndarray, groups: Union[np.ndarray, pd.DatetimeIndex]
) -> np.ndarray:
    """Remove train indices whose groups overlap between the training and test sets and add them to the test set.

    Parameters
    ----------
    train_index : np.ndarray
        The initial training indices.
    test_index : np.ndarray
        The test indices.
    indices : np.ndarray
        All indices.
    groups : Union[np.ndarray, pd.DatetimeIndex]
        The groups.

    Returns
    -------
    np.ndarray
        The updated training indices.
    """
    train_groups, test_groups = groups[train_index], groups[test_index]
    overlapping_groups = np.intersect1d(train_groups, test_groups)
    overlapping_index = indices[np.isin(groups, overlapping_groups)]
    train_index = np.sort(np.setdiff1d(train_index, overlapping_index))
    test_index = np.sort(np.union1d(test_index, overlapping_index))
    return train_index, test_index


def _validate_purge_embargo_inputs(
    train_index: np.ndarray,
    test_index: np.ndarray,
    indices: np.ndarray,
    groups: Union[np.ndarray, pd.DatetimeIndex],
    delta_t: Union[int, pd.Timedelta],
) -> None:  # pragma: no cover
    """Validate the inputs for the `purge` and `embargo` functions.

    Parameters
    ----------
    train_index : np.ndarray
        The initial training indices.
    test_index : np.ndarray
        The test indices.
    indices : np.ndarray
        All indices.
    groups : Union[np.ndarray, pd.DatetimeIndex]
        The groups.
    delta_t : Union[int, pd.Timedelta]
        The time period for purging or embargoing.
    """
    if not all(isinstance(x, np.ndarray) for x in [train_index, test_index, indices]):
        raise TypeError("train_index, test_index, and indices must be numpy arrays")
    if isinstance(delta_t, Integral) and isinstance(groups, np.ndarray):
        if delta_t < 0:
            raise ValueError("delta_t must be a non-negative integer")
        if not np.all(np.diff(groups) >= 0):
            raise ValueError("groups must be sorted in ascending order")
    elif isinstance(delta_t, pd.Timedelta) and isinstance(groups, pd.DatetimeIndex):
        if delta_t < pd.Timedelta(0):
            raise ValueError("delta_t must be a non-negative Timedelta")
        if not groups.is_monotonic_increasing:
            raise ValueError("groups must be sorted in ascending order")
    else:
        raise TypeError(
            "delta_t must be an integer and groups must be a numpy array, or delta_t must be a Timedelta and groups must be a DatetimeIndex"
        )


def _purge(
    train_index: np.ndarray,
    test_index: np.ndarray,
    indices: np.ndarray,
    groups: Union[np.ndarray, pd.DatetimeIndex],
    n_purge: Union[int, pd.Timedelta],
) -> np.ndarray:
    """Remove training indices whose groups fall within `n_purge` from the last test group.

    Parameters
    ----------
    train_index : np.ndarray
        The initial training indices.
    test_index : np.ndarray
        The test indices.
    indices : np.ndarray
        All indices.
    groups : Union[np.ndarray, pd.DatetimeIndex]
        The groups
    n_purge : Union[int, pd.Timedelta]
        Purge period.

    Returns
    -------
    np.ndarra
        The purged training indices.
    """
    _validate_purge_embargo_inputs(train_index, test_index, indices, groups, n_purge)
    max_test_group = groups[test_index].max()
    purged_index = indices[(groups <= max_test_group + n_purge) & (groups > max_test_group)]
    return np.setdiff1d(train_index, purged_index)


def _embargo(
    train_index: np.ndarray,
    test_index: np.ndarray,
    indices: np.ndarray,
    groups: Union[np.ndarray, pd.DatetimeIndex],
    n_embargo: Union[int, pd.Timedelta],
) -> np.ndarray:
    """Remove training indices whose groups fall within `n_embargo` from the first test group.

    Parameters
    ----------
    train_index : np.ndarray
        The initial training indices.
    test_index : np.ndarray
        The test indices.
    indices : np.ndarray
        All indices.
    groups : Union[np.ndarray, pd.DatetimeIndex]
        The groups
    n_embargo : Union[int, pd.Timedelta]
        Embargo period.

    Returns
    -------
    np.ndarray
        The embargoed training indices.
    """
    _validate_purge_embargo_inputs(train_index, test_index, indices, groups, n_embargo)
    min_test_group = groups[test_index].min()
    embargoed_index = indices[(groups >= min_test_group - n_embargo) & (groups < min_test_group)]
    return np.setdiff1d(train_index, embargoed_index)
