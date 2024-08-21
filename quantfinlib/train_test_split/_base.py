"""Base Purged and Embargoed Cross-Validation Functions."""

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, Optional, Union

import numpy as np
import pandas as pd



def _validate_purge_embargo_inputs(
    train_index: np.ndarray,
    test_index: np.ndarray,
    indices: np.ndarray,
    groups: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    delta_t: Union[Integral, pd.Timedelta],
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
    groups : Union[np.ndarray, pd.Series, pd.DatetimeIndex]
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
    elif isinstance(delta_t, pd.Timedelta) and isinstance(groups, (pd.Series, pd.DatetimeIndex)):
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
    groups: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    n_purge: Union[Integral, pd.Timedelta],
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
    groups : Union[np.ndarray, pd.Series, pd.DatetimeIndex]
        The groups
    n_purge : Union[int, pd.Timedelta]
        Purge period.

    Returns
    -------
    np.ndarra
        The purged training indices.
    
    Raises
    ------
    ValueError
        If all training indices were purged
    """
    _validate_purge_embargo_inputs(train_index, test_index, indices, groups, n_purge)
    max_test_group = groups[test_index].max()
    purged_index = indices[(groups <= max_test_group + n_purge) & (groups > max_test_group)]
    train_index = np.setdiff1d(train_index, purged_index)
    if len(train_index) == 0:
        raise ValueError("All training indices were purged. No training data available.")
    return np.setdiff1d(train_index, purged_index)


def _embargo(
    train_index: np.ndarray,
    test_index: np.ndarray,
    indices: np.ndarray,
    groups: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    n_embargo: Union[Integral, pd.Timedelta],
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
    groups : Union[np.ndarray, pd.Series, pd.DatetimeIndex]
        The groups
    n_embargo : Union[int, pd.Timedelta]
        Embargo period.

    Returns
    -------
    np.ndarray
        The embargoed training indices.

    Raises
    ------
    ValueError
        If all training indices were embargoed.
    """
    _validate_purge_embargo_inputs(train_index, test_index, indices, groups, n_embargo)
    min_test_group = groups[test_index].min()
    embargoed_index = indices[(groups >= min_test_group - n_embargo) & (groups < min_test_group)]
    train_index = np.setdiff1d(train_index, embargoed_index)
    if len(train_index) == 0:
        raise ValueError("All training indices were embargoed. No training data available.")
    return train_index


class BaseCV(ABC):
    """Base class for cross-validation splitters."""

    def __init__(self, n_embargo: Union[Integral, pd.Timedelta], n_purge: Union[Integral, pd.Timedelta]) -> None:
        if not isinstance(n_embargo, (Integral, pd.Timedelta)):
            raise TypeError("n_embargo must be an integer or a Timedelta")
        if not isinstance(n_purge, (Integral, pd.Timedelta)):
            raise TypeError("n_purge must be an integer or a Timedelta")
        if isinstance(n_embargo, Integral) and n_embargo < 0:
            raise ValueError("n_embargo must be a non-negative integer")
        if isinstance(n_purge, Integral) and n_purge < 0:
            raise ValueError("n_purge must be a non-negative integer")
        if isinstance(n_embargo, pd.Timedelta) and n_embargo < pd.Timedelta(0):
            raise ValueError("n_embargo must be a non-negative Timedelta")
        if isinstance(n_purge, pd.Timedelta) and n_purge < pd.Timedelta(0):
            raise ValueError("n_purge must be a non-negative Timedelta")
        self.n_embargo = n_embargo
        self.n_purge = n_purge

    @abstractmethod
    def get_n_splits(self) -> Any:
        """Return the number of splits."""
        raise NotImplementedError("Subclasses must implement get_n_split method")

    @abstractmethod
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None,
    ) -> Any:
        """Generate indices to split data into training and test sets."""
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be a numpy array or a DataFrame")
        if y is not None and not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy array or a Series")
        if groups is not None and not isinstance(groups, (np.ndarray, pd.Series, pd.DatetimeIndex)):
            raise TypeError("groups must be a numpy array or a Series")
        if isinstance(groups, np.ndarray) and not np.all(np.diff(groups) >= 0):
            raise ValueError("groups must be sorted in ascending order")
        if isinstance(groups, (pd.Series, pd.DatetimeIndex)) and not groups.is_monotonic_increasing:
            raise ValueError("groups must be sorted in ascending order")
