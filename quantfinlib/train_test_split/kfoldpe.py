"""Implementation of time-aware KFold Cross Validation with Purge and Embargo."""

from numbers import Integral
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd

from quantfinlib.train_test_split._base import BaseCV, _purge, _embargo
from quantfinlib.train_test_split._dtypes import BoundPerFoldType, GroupType, IndexType, XType, YType


def _validate_split_settings(n_samples: Integral, n_folds: Integral, n_embargo: Integral, n_purge: Integral) -> None:
    """Validate the split settings.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_folds : int
        Number of folds.
    n_embargo : int
        Number of groups to embargo.
    n_purge : int
        Number of groups to purge.

    Raises
    ------
    ValueError:
        If the number of samples is less than the number of folds.
        If the sum of n_purge and n_embargo is greater than n_samples.
        If n_purge or n_embargo is greater than max_train_size.

    """
    if n_samples < n_folds:
        raise ValueError("Number of samples must be greater than the number of folds.")
    if n_purge + n_embargo >= n_samples:
        raise ValueError(f"The sum {n_purge + n_embargo = } must be less than {n_samples = }.")
    max_train_size: Integral = n_samples - n_samples // n_folds
    if (n_purge > max_train_size) or (n_embargo > max_train_size) or (n_embargo + n_purge > max_train_size):
        raise ValueError(f"n_purge and n_embargo and their sum must be less than {max_train_size = }.")


class TimeAwareKFold(BaseCV):
    """Purge-embargo K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets. The training set is purged and embargoed.

    Attributes
    ----------
    n_folds : int
        Number of folds. This is the number of train-test splits if look_forward is False.
        Otherwise, the number of splits is n_folds - 1.
    n_purge : int
        Number of groups to purge.
    n_embargo : int
        Number of groups to embargo.
    look_forward : bool
        Whether the test set is always ahead of the training set.
        If True, the number of splits is n_folds - 1.
    freq : Optional[str]
        The frequency of the groups provided to split method.

    Methods
    -------
    get_n_splits
        Return the number of splits.

    split(X, y, groups)
        Generate indices to split data into training and test sets.
    """

    def __init__(
        self,
        n_folds: Integral,
        n_embargo: Integral,
        n_purge: Integral,
        look_forward: bool = False,
        freq: Optional[str] = None,
    ) -> None:
        super().__init__(n_embargo=n_embargo, n_purge=n_purge)
        if not isinstance(n_folds, Integral):
            raise ValueError("Number of folds must be an integer.")
        if n_folds < 2:
            raise ValueError("Number of folds must be at least 2.")
        self.n_folds = n_folds
        self.n_purge = n_purge
        self.n_embargo = n_embargo
        self.look_forward = look_forward
        self.freq = freq

    def get_n_splits(
        self, X: Optional[XType] = None, y: Optional[YType] = None, groups: Optional[GroupType] = None
    ) -> int:
        """Return the number of splits.
        This method return n_folds - 1 if look_forward is True, otherwise n_folds.

        Returns
        -------
        int
            Number of splits.
        """
        if self.look_forward:
            return self.n_folds - 1
        return self.n_folds

    def split(
        self,
        X: XType,
        y: Optional[YType] = None,
        groups: Optional[GroupType] = None,
    ) -> Generator[Tuple[IndexType, IndexType], None, None]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The data to split.
        y : Optional[Union[np.ndarray, pd.Series]], optional
            The target variable, by default None.
        groups : Optional[Union[np.ndarray, pd.Series]], optional
            The groups, indicating datetime. by default None.

        Yields
        ------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            Train and test indices.
        """
        super().split(X, y, groups)
        _validate_split_settings(
            n_samples=X.shape[0], n_folds=self.n_folds, n_embargo=self.n_embargo, n_purge=self.n_purge
        )
        if groups is not None and self.freq is not None:
            n_embargo = pd.Timedelta(self.n_embargo, unit=self.freq)
            n_purge = pd.Timedelta(self.n_purge, unit=self.freq)
        else:
            n_embargo = self.n_embargo
            n_purge = self.n_purge
        if groups is None:
            groups = np.arange(X.shape[0])
        indices = np.arange(X.shape[0])
        assert len(groups) == len(indices), "Groups and indices must have the same length."
        test_indices_per_fold = np.array_split(indices, self.n_folds)
        # fix the bounds of the folds so that there is no overlap between their groups
        bound_per_fold = np.array([[test_indices[0], test_indices[-1]] for test_indices in test_indices_per_fold])
        bound_per_fold = _modify_fold_bounds(bound_per_fold=bound_per_fold, groups=groups)
        test_indices_per_fold = [np.arange(bound[0], bound[1] + 1) for bound in bound_per_fold]
        if self.look_forward:
            test_indices_per_fold = test_indices_per_fold[1:]
        for test_index in test_indices_per_fold:
            train_index = _get_train_index_per_split(
                indices=indices, test_index=test_index, groups=groups, look_forward=self.look_forward
            )
            train_index = _purge(
                train_index=train_index, test_index=test_index, indices=indices, groups=groups, n_purge=n_purge
            )
            train_index = _embargo(
                train_index=train_index, test_index=test_index, indices=indices, groups=groups, n_embargo=n_embargo
            )
            yield train_index, test_index


def _modify_fold_bounds(bound_per_fold: BoundPerFoldType, groups: GroupType) -> BoundPerFoldType:
    """
    Modify fold bounds.

    Modifies the bounds of folds such that there is no overlap between their groups.

    Parameters
    ----------
    bound_per_fold : np.ndarray
        Bound per fold.
    groups : np.ndarray
        Groups.
    n_folds : int
        Number of folds.

    Returns
    -------
    np.ndarray
        Modified bound per fold.
    """
    n_folds = bound_per_fold.shape[0]
    for fold in range(n_folds - 1):
        last_group_curr_fold = groups[bound_per_fold[fold, 1]]
        first_group_next_fold = groups[bound_per_fold[fold + 1, 0]]
        if last_group_curr_fold == first_group_next_fold:
            next_group = groups[groups > last_group_curr_fold][0]
            bound_per_fold[fold, 1] = np.where(groups == last_group_curr_fold)[0][-1]
            bound_per_fold[fold + 1, 0] = np.where(groups == next_group)[0][0]
    return bound_per_fold


def _get_train_index_per_split(
    indices: IndexType,
    test_index: IndexType,
    groups: GroupType,
    look_forward: bool = False,
) -> IndexType:
    """Find train index for a given split.

    Parameters
    ----------
    indices : np.ndarray
        All indices.
    test_index : np.ndarray
        Test indices.
    groups : Union[np.ndarray, pd.TimedeltaIndex]
        Groups.
    look_forward : bool
        Whether the test set is always ahead of the training set.

    Returns
    -------
    np.ndarray
        Train indices for a given split.
    """
    train_index = np.setdiff1d(indices, test_index)
    if look_forward:
        return np.setdiff1d(train_index, indices[groups > groups[test_index[0]]])
    return train_index
