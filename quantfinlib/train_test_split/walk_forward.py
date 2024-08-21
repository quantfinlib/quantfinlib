"""Implementation of Walk forward."""

from numbers import Integral
from typing import Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quantfinlib.train_test_split._base import BaseCV, _embargo


class WalkForward(BaseCV):
    """Walk forward train-test split."""

    def __init__(
        self,
        n_embargo: Union[Integral, pd.Timedelta],
        n_purge: Union[Integral, pd.Timedelta] = 0,
        train_window: Union[Integral, pd.Timedelta] = 0,
        test_window: Union[Integral, pd.Timedelta] = 0,
        wf_split_type: str = "expanding",
        freq: Optional[str] = None,
    ) -> None:
        super().__init__(n_embargo=n_embargo, n_purge=n_purge)
        self.train_window = train_window
        self.test_window = test_window
        self.wf_split_type = wf_split_type
        if wf_split_type not in ["expanding", "rolling"]:
            raise ValueError("Invalid value for 'wf_split_type'. Must be either 'expanding' or 'rolling'.")
        self.freq = freq
        self.n_splits = 0
        self._finalized_splits = False

    def __repr__(self):
        """Return the string representation of the object."""
        return (
            f"WalkForward(n_embargo={self.n_embargo}, n_purge={self.n_purge}, "
            f"train_window={self.train_window}, test_window={self.test_window}, "
            f"wf_split_type={self.wf_split_type})"
        )

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        if self._finalized_splits:
            return self.n_splits
        else:
            raise ValueError("Splits have not been finalized. Call the split method first.")

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
            The groups, indicating datetime. by default None.

        Yields
        ------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            Train and test indices.

        """
        super().split(X, y, groups)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        groups = groups if groups is not None else np.arange(n_samples)
        time_unit = pd.Timedelta(1, unit=self.freq) if self.freq is not None else 1

        # Calculate the boundaries of the training set
        train_start_group = groups[0]
        train_end_group = groups[
            np.searchsorted(groups, train_start_group + self.train_window - time_unit, side="left")
        ]
        # Calculate the boundaries of the test set
        test_start_group = groups[
            np.searchsorted(groups, train_start_group + self.train_window - time_unit, side="right")
        ]
        test_end_group = groups[np.searchsorted(groups, test_start_group + self.test_window - time_unit, side="left")]
        while train_end_group < groups[-1]:
            train_index = indices[(groups >= train_start_group) & (groups <= train_end_group)]
            test_index = indices[(groups >= test_start_group) & (groups <= test_end_group)]
            if (isinstance(self.n_embargo, Integral) and self.n_embargo > 0) or (
                isinstance(self.n_embargo, pd.Timedelta) and self.n_embargo > pd.Timedelta(0)
            ):
                train_index = _embargo(
                    train_index=train_index,
                    test_index=test_index,
                    indices=indices,
                    groups=groups,
                    n_embargo=self.n_embargo,
                )
            yield train_index, test_index
            self.n_splits += 1
            # if the current end group of the test set is the last group, then we have reached the end of the data
            if test_end_group >= groups[-1]:
                self._finalized_splits = True
                break
            # update the boundaries of the training and test sets
            if self.wf_split_type == "rolling":
                train_start_group = train_start_group + self.test_window
            train_end_group = groups[np.searchsorted(groups, train_end_group + self.test_window, side="left")]
            test_start_group = groups[np.searchsorted(groups, test_start_group + self.test_window, side="left")]
            test_end_group = groups[
                np.searchsorted(groups, min(test_end_group + self.test_window, groups[-1]), side="left")
            ]
