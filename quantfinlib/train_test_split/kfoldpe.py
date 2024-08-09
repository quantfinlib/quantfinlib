"""Implementation of time-aware KFold Cross Validation with Purge and Embargo."""

from numbers import Integral
from typing import Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quantfinlib.train_test_split._base import _BaseCV, _purge, _embargo


def _validate_split_settings(n_samples: int, n_splits: int, n_embargo: int, n_purge: int) -> None:
    """Validate the split settings."""
    if n_samples < n_splits:
        raise ValueError("Number of samples must be greater than the number of splits.")
    if n_purge + n_embargo >= n_samples:
        raise ValueError(f"The sum {n_purge + n_embargo = } must be less than {n_samples = }.")
    max_train_size = n_samples - n_samples // n_splits
    if n_purge > max_train_size:
        raise ValueError(f"{n_purge = } must be less than {max_train_size = }.")
    if n_embargo > max_train_size:
        raise ValueError(f"{n_embargo = } must be less than {max_train_size = }.")
    if n_purge + n_embargo > max_train_size:
        raise ValueError(f"The sum {n_purge + n_embargo = } must be less than {max_train_size = }.")


class KFoldPE(_BaseCV):
    """Purge-embargo K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets. The training set is purged and embargoed.

    Attributes
    ----------
    n_splits : int
        Number of splits.
    n_purge : Union[int, pd.Timedelta]
        Number of groups to purge.
    n_embargo : Union[int, pd.Timedelta]
        Number of groups to embargo.
    look_forward : bool
        Whether to look forward.

    Methods
    -------
    get_n_splits()
        Return the number of splits.

    split(X, y, groups)
        Generate indices to split data into training and test sets.

    modify_fold_bnds(bnd_per_fold, groups)
        Modify fold bounds so that there is no overlap between their groups.

    get_train_index_per_split(indices, test_index, groups)
        Find train index for a given split.
    """

    def __init__(
        self,
        n_splits: int,
        n_embargo: Union[int, pd.Timedelta],
        n_purge: Union[int, pd.Timedelta],
        look_forward: bool = False,
    ) -> None:
        super().__init__(n_embargo=n_embargo, n_purge=n_purge)
        if not isinstance(n_splits, Integral):
            raise ValueError("Number of splits must be an integer.")
        if n_splits < 2:
            raise ValueError("Number of splits must be at least 2.")
        self.n_splits = n_splits
        self.n_purge = n_purge
        self.n_embargo = n_embargo
        self.look_forward = look_forward

    def get_n_splits(self) -> int:
        """Return the number of splits.

        Returns
        -------
        int
            Number of splits.
        """
        if self.look_forward:
            return self.n_splits - 1
        return self.n_splits

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The data to split.
        y : Optional[Union[np.ndarray, pd.Series]], optional
            The target variable, by default None.
        groups : Optional[Union[np.ndarray, pd.Series]], optional
            The groups, by default None.

        Yields
        ------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            Train and test indices.
        """
        _validate_split_settings(
            n_samples=X.shape[0], n_splits=self.n_splits, n_embargo=self.n_embargo, n_purge=self.n_purge
        )
        if groups is None:
            groups = np.arange(X.shape[0])
        indices = np.arange(X.shape[0])
        test_indices_per_fold = np.array_split(indices, self.n_splits)
        bnd_per_fold = np.array([[test_indices[0], test_indices[-1]] for test_indices in test_indices_per_fold])
        if self.look_forward:
            test_indices_per_fold = test_indices_per_fold[1:]
        bnd_per_fold = self.modify_fold_bnds(bnd_per_fold, groups)
        for test_index in test_indices_per_fold:
            train_index = self.get_train_index_per_split(indices, test_index, groups)
            train_index = _purge(
                train_index=train_index, test_index=test_index, indices=indices, groups=groups, n_purge=self.n_purge
            )
            train_index = _embargo(
                train_index=train_index, test_index=test_index, indices=indices, groups=groups, n_embargo=self.n_embargo
            )
            yield train_index, test_index

    def modify_fold_bnds(self, bnd_per_fold: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        Modify fold bounds.

        Modifies the bounds of folds such that there is no overlap between their groups.

        Parameters
        ----------
        bnd_per_fold : np.ndarray
            Bound per fold.
        groups : np.ndarray
            Groups.

        Returns
        -------
        np.ndarray
            Modified bound per fold.
        """
        for fold in range(self.n_splits - 1):
            last_group_curr_fold = groups[bnd_per_fold[fold, 1]]
            first_group_next_fold = groups[bnd_per_fold[fold + 1, 0]]
            if last_group_curr_fold == first_group_next_fold:
                next_group = groups[groups > last_group_curr_fold][0]
                bnd_per_fold[fold, 1] = np.where(groups == last_group_curr_fold)[0][-1]
                bnd_per_fold[fold + 1, 0] = np.where(groups == next_group)[0][0]
        return bnd_per_fold

    def get_train_index_per_split(
        self, indices: np.ndarray, test_index: np.ndarray, groups: Union[np.ndarray, pd.TimedeltaIndex]
    ) -> np.ndarray:
        """Find train index for a given split.

        Parameters
        ----------
        indices : np.ndarray
            All indices.
        test_index : np.ndarray
            Test indices.
        groups : Union[np.ndarray, pd.TimedeltaIndex]
            Groups.

        Returns
        -------
        np.ndarray
            Train indices for a given split.
        """
        train_index = np.setdiff1d(indices, test_index)
        if self.look_forward:
            train_index = np.setdiff1d(train_index, indices[groups < groups[test_index[0]]])
        return train_index
