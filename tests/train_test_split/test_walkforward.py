from numbers import Integral
import logging
from quantfinlib.train_test_split.walk_forward import WalkForward, _to_timedelta

import numpy as np
import pandas as pd
import pytest

from itertools import product

np.random.seed(42)

int_test_data = list(product(np.arange(7), [9, 10, 11], np.arange(1, 10), ["expanding", "rolling"]))


@pytest.mark.parametrize("n_embargo, train_window, test_window, split_type", int_test_data)
def test_walk_forward_int(n_embargo, train_window, test_window, split_type):
    wf = WalkForward(
        n_embargo=n_embargo,
        n_purge=0,
        train_window=train_window,
        test_window=test_window,
        split_type=split_type
    )
    groups = np.random.randint(0, 33, 100)
    # make sure that all groups between 0 and 32 are present in the data
    groups = np.unique(groups, return_inverse=True)[1].reshape(groups.shape)
    groups.sort()
    X = np.random.randn(100, 2)
    for train_index, test_index in wf.split(X=X, groups=groups):
        train_groups = groups[train_index]
        test_groups = groups[test_index]
        assert len(np.intersect1d(train_groups, test_groups)) == 0
        assert min(test_groups) - max(train_groups) >= wf.n_embargo
        assert train_groups.min() - train_groups.max() <= wf.train_window - wf.n_embargo
        assert test_groups.min() - test_groups.max() <= wf.test_window
        if wf.split_type == "expanding":
            assert train_groups.min() == groups[0]
    assert test_groups.max() == groups[-1]


@pytest.mark.parametrize("n_embargo, train_window, test_window, split_type", int_test_data)
def test_walk_forward_datetime(n_embargo, train_window, test_window, split_type):
    groups = pd.date_range("2020-01-01", periods=100, freq="D")
    group_indices = np.random.randint(0, 60, 100)
    # make sure that all groups between 0 and 32 are present in the data
    group_indices = np.unique(group_indices, return_inverse=True)[1].reshape(group_indices.shape)
    group_indices.sort()
    groups = groups[group_indices]
    X = np.random.randn(100, 2)
    
    wf = WalkForward(
        n_embargo=n_embargo,
        n_purge=0,
        train_window=train_window,
        test_window=test_window,
        split_type=split_type,
        freq="D",
    )
    with pytest.warns(UserWarning, match='Call the split method first to get the number of splits for WalkForward.'):
        n_splits = wf.get_n_splits()
        assert n_splits == 0

    assert wf.__repr__() == (
        f"WalkForward(n_embargo={_to_timedelta(n_embargo, 'D')}, n_purge=0, train_window={_to_timedelta(train_window, 'D')}, "
        f"test_window={_to_timedelta(test_window, 'D')}, split_type={split_type})"
    )
    assert wf._finalized_splits == False
    for train_index, test_index in wf.split(X=X, groups=groups):
        train_groups = groups[train_index]
        test_groups = groups[test_index]
        assert len(np.intersect1d(train_groups, test_groups)) == 0
        assert min(test_groups) - max(train_groups) >= wf.n_embargo
        assert train_groups.min() - train_groups.max() <= wf.train_window - wf.n_embargo
        assert test_groups.min() - test_groups.max() <= wf.test_window
        if wf.split_type == "expanding":
            assert train_groups.min() == groups[0]
        train_end_group = train_groups.max()
        assert train_end_group < groups[-1]
    assert test_groups.max() == groups[-1]
    assert wf._finalized_splits == True
    assert isinstance(wf.get_n_splits(), Integral)
    assert wf.get_n_splits() > 0
    assert isinstance(wf.get_n_splits(), Integral)
    assert wf.get_n_splits() == len(list(wf.split(X=X, groups=groups)))


def test_invalid_split_type():
    with pytest.raises(ValueError):
        WalkForward(n_embargo=0, n_purge=0, train_window=1, test_window=1, split_type="invalid")
    