from numbers import Integral
from typing import Union

import numpy as np
import pandas as pd


def _remove_overlapping_groups(    
    train_index: np.ndarray,
    test_index: np.ndarray,
    indices: np.ndarray,
    groups: Union[np.ndarray, pd.DatetimeIndex]
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
    if not all(isinstance(x, np.ndarray) for x in [train_index, test_index, indices]):
        raise TypeError("train_index, test_index, and indices must be numpy arrays")
    if isinstance(n_purge, Integral) and isinstance(groups, np.ndarray):
        if n_purge < 0:
            raise ValueError("n_purge must be a non-negative integer")
        if not np.all(np.diff(groups) >= 0):
            raise ValueError("groups must be sorted in ascending order")
        max_test_group = groups[max(test_index)]
        purged_index = indices[(groups <= max_test_group + n_purge) & (groups > max_test_group)]
        train_index = np.setdiff1d(train_index, purged_index)
    elif isinstance(n_purge, pd.Timedelta) and isinstance(groups, pd.DatetimeIndex):
        if n_purge < pd.Timedelta(0):
            raise ValueError("n_purge must be a non-negative Timedelta")
        if not groups.is_monotonic_increasing:
            raise ValueError("groups must be sorted in ascending order")
        max_test_group = groups[test_index].max()
        purged_index = indices[(groups <= max_test_group + n_purge) & (groups > max_test_group)]
        train_index = np.setdiff1d(train_index, purged_index)
    else:
        raise TypeError(
            "n_purge must be an integer and groups must be a numpy array, or n_purge must be a Timedelta and groups must be a DatetimeIndex"
        )
    return train_index


def _embarge(
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
    if not all(isinstance(x, np.ndarray) for x in [train_index, test_index, indices]):
        raise TypeError("train_index, test_index, and indices must be numpy arrays")
    if isinstance(n_embargo, Integral) and isinstance(groups, np.ndarray):
        if n_embargo < 0:
            raise ValueError("n_embargo must be a non-negative integer")
        if not np.all(np.diff(groups) >= 0):
            raise ValueError("groups must be sorted in ascending order")
        min_test_group = groups[min(test_index)]
        embargoed_index = indices[(groups >= min_test_group - n_embargo) & (groups < min_test_group)]
        train_index = np.setdiff1d(train_index, embargoed_index)
    elif isinstance(n_embargo, pd.Timedelta) and isinstance(groups, pd.DatetimeIndex):
        if n_embargo < pd.Timedelta(0):
            raise ValueError("n_embargo must be a non-negative Timedelta")
        if not groups.is_monotonic_increasing:
            raise ValueError("groups must be sorted in ascending order")
        min_test_group = groups[test_index].min()
        embargoed_index = indices[(groups >= min_test_group - n_embargo) & (groups < min_test_group)]
        train_index = np.setdiff1d(train_index, embargoed_index)
    else:
        raise TypeError(
            "n_embargo must be an integer and groups must be a numpy array, or n_embargo must be a Timedelta and groups must be a DatetimeIndex"
        )
    return train_index
