from itertools import product

from quantfinlib.train_test_split.kfoldpe import TimeAwareKFold, _validate_split_settings

import numpy as np
import pandas as pd
import pytest


np.random.seed(123)


def test_validate_split_settings():
    with pytest.raises(ValueError):
        _validate_split_settings(n_samples=10, n_folds=11, n_embargo=0, n_purge=0)
    with pytest.raises(ValueError):
        _validate_split_settings(n_samples=10, n_folds=5, n_embargo=6, n_purge=6)
    with pytest.raises(ValueError):
        _validate_split_settings(n_samples=10, n_folds=5, n_embargo=0, n_purge=9)
    with pytest.raises(ValueError):
        _validate_split_settings(n_samples=10, n_folds=5, n_embargo=9, n_purge=0)
    with pytest.raises(ValueError):
        _validate_split_settings(n_samples=10, n_folds=5, n_embargo=5, n_purge=5)


def test_invalid_n_folds():
    with pytest.raises(ValueError):
        TimeAwareKFold(n_folds=0, n_embargo=0, n_purge=0)
    with pytest.raises(ValueError):
        TimeAwareKFold(n_folds=5.5, n_embargo=0, n_purge=0)


def test_get_n_splits():
    kfold = TimeAwareKFold(n_folds=5, n_embargo=0, n_purge=0)
    assert kfold.get_n_splits() == 5
    kfold = TimeAwareKFold(n_folds=10, n_embargo=0, n_purge=0, look_forward=True)
    assert kfold.get_n_splits() == 9



test_data = list(product(np.arange(5,10), [1,2,3], [1,2,3], [True, False]))

@pytest.mark.parametrize("n_folds, n_embargo, n_purge, look_forward", test_data)
def test_split(n_folds, n_embargo, n_purge, look_forward):
    kfold = TimeAwareKFold(n_folds=n_folds, n_embargo=n_embargo, n_purge=n_purge, look_forward=look_forward)
    X = np.random.randn(1000, 2)
    groups = np.random.randint(0, 100, 1000)
    groups = np.unique(groups, return_inverse=True)[1].reshape(groups.shape)
    groups.sort()
    for train_index, test_index in kfold.split(X=X, groups=groups):
        assert len(np.intersect1d(train_index, test_index)) == 0
        train_groups = groups[train_index]
        test_groups = groups[test_index]
        assert len(np.intersect1d(train_groups, test_groups)) == 0
        if any(train_index < min(test_index)):
            assert len(np.unique(groups)) - len(np.unique(train_groups)) - len(np.unique(test_groups)) >= kfold.n_embargo  
        if look_forward:
            assert min(test_groups) - max(train_groups) >= kfold.n_embargo

#def test_