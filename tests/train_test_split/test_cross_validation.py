from cycler import K
from quantfinlib.train_test_split.cross_validation import _purge, _embarge, _remove_overlapping_groups
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import pytest


@pytest.fixture 
def kfold_data_integer_groups_even_sampling():# -> tuple[Iterator[tuple[ndarray[Any, Any], ndarray[Any, Any]...:
    n_splits = 5
    n_samples = 100
    indices = np.arange(n_samples)
    groups = np.hstack([np.zeros(10) + i for i in range(10)])
    kfold = KFold(n_splits=n_splits)
    return kfold.split(indices), indices, groups


@pytest.fixture
def kfold_data_integer_groups_uneven_sampling():
    n_splits = 5
    n_samples = 100
    indices = np.arange(n_samples)
    groups = np.random.randint(0, 33, n_samples)
    # make sure that all groups between 0 and 32 are present in the data
    groups = np.unique(groups, return_inverse=True)[1].reshape(groups.shape)
    groups.sort()
    print(groups)
    kfold = KFold(n_splits=n_splits)
    return kfold.split(indices), indices, groups


def get_train_test_groups(groups, train_index, test_index):
    train_groups = groups[train_index]
    test_groups = groups[test_index]
    return train_groups, test_groups


@pytest.mark.parametrize("n_purge", np.arange(0, 5))
def test_purge_integers_even_sampling(n_purge, kfold_data_integer_groups_even_sampling):
    spliter, indices, groups = kfold_data_integer_groups_even_sampling
    for train_index, test_index in spliter:
        train_index, test_index = _remove_overlapping_groups(train_index, test_index, indices, groups)
        purged_train_index = _purge(train_index, test_index, indices, groups, n_purge)
        train_groups, test_groups = get_train_test_groups(groups, train_index, test_index)
        train_groups_purged, _ = get_train_test_groups(groups, purged_train_index, test_index)
        assert len(np.intersect1d(train_groups_purged, test_groups)) == 0
        assert len(np.unique(train_groups)) - len(np.unique(train_groups_purged)) <= n_purge
        if any(train_index > max(test_index)):
            expected_purged_groups = max(test_groups) + np.arange(1, n_purge + 1)
            expected_purged_groups = expected_purged_groups[expected_purged_groups <= max(groups)] # Correct for the upper bound of the groups
            assert len(np.intersect1d(train_groups_purged, expected_purged_groups)) == 0, "Expected purged groups not to be in the final training set."
            assert sorted(np.setdiff1d(train_groups, train_groups_purged)) == sorted(expected_purged_groups), "Expected purged groups to be removed from the final training set."
            assert len(purged_train_index) >= len(train_index) - n_purge * 10
            if max(train_groups) <= max(test_groups)+n_purge:
                assert max(train_groups_purged) < min(test_groups)   
        else:
            assert len(purged_train_index) == len(train_index)
            assert len(train_groups_purged) == len(train_groups)



@pytest.mark.parametrize("n_purge", np.arange(0, 5))
def test_purge_integers_uneven_sampling(n_purge, kfold_data_integer_groups_uneven_sampling):
    spliter, indices, groups = kfold_data_integer_groups_uneven_sampling
    for train_index, test_index in spliter:
        train_index, test_index = _remove_overlapping_groups(train_index, test_index, indices, groups)
        purged_train_index = _purge(train_index, test_index, indices, groups, n_purge)
        train_groups, test_groups = get_train_test_groups(groups, train_index, test_index)
        train_groups_purged, _ = get_train_test_groups(groups, purged_train_index, test_index)
        assert len(np.intersect1d(train_groups_purged, test_groups)) == 0
        assert len(np.unique(train_groups)) - len(np.unique(train_groups_purged)) <= n_purge
        if any(train_index > max(test_index)):
            expected_purged_groups = max(test_groups) + np.arange(1, n_purge + 1)
            expected_purged_groups = expected_purged_groups[expected_purged_groups <= max(groups)] # Correct for the upper bound of the groups
            assert len(np.intersect1d(train_groups_purged, expected_purged_groups)) == 0, "Expected purged groups not to be in the final training set."
            assert sorted(np.setdiff1d(train_groups, train_groups_purged)) == sorted(expected_purged_groups), "Expected purged groups to be removed from the final training set."
            if max(train_groups) <= max(test_groups)+n_purge:
                assert max(train_groups_purged) < min(test_groups)   
        else:
            assert len(purged_train_index) == len(train_index)
            assert len(train_groups_purged) == len(train_groups)


@pytest.mark.parametrize("n_purge", [pd.Timedelta(days=1), -1, 1.34, 1j+1, "1"])
def test_invalid_n_purge_integer_groups(n_purge, kfold_data_integer_groups_even_sampling):
    spliter, indices, groups = kfold_data_integer_groups_even_sampling
    for train_index, test_index in spliter:
        train_index, test_index = _remove_overlapping_groups(train_index, test_index, indices, groups)
        if n_purge == -1:
            with pytest.raises(ValueError):
                _purge(train_index, test_index, indices, groups, n_purge)
        else: 
            with pytest.raises(TypeError):
                _purge(train_index, test_index, indices, groups, n_purge)
