from typing import Union

from quantfinlib.train_test_split._base import _purge, _embargo
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import pytest



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

def integer_even_groups():
    return np.hstack([np.zeros(10) + i for i in range(10)])


def integer_uneven_groups():
    groups = np.random.randint(0, 33, 100)
    # make sure that all groups between 0 and 32 are present in the data
    groups = np.unique(groups, return_inverse=True)[1].reshape(groups.shape)
    groups.sort()
    return groups


def datetime_even_groups():
    return pd.date_range("2020-01-01", periods=100, freq="D")


def datetime_uneven_groups():
    groups = pd.date_range("2020-01-01", periods=100, freq="D")
    group_indices = np.random.randint(0, 60, 100)
    # make sure that all groups between 0 and 32 are present in the data
    group_indices = np.unique(group_indices, return_inverse=True)[1].reshape(group_indices.shape)
    group_indices.sort()
    return groups[group_indices]


@pytest.fixture(
    params=[
        ("integer", np.arange(100), integer_even_groups()),
        ("integer_uneven", np.arange(100), integer_uneven_groups()),
        ("datetime_even", np.arange(100), datetime_even_groups()),
        ("datetime_uneven", np.arange(100), datetime_uneven_groups()),
    ]
)
def kfold_data(request):
    data_type, indices, groups = request.param
    kfold = KFold(n_splits=5)
    return kfold.split(indices), indices, groups, data_type


def test_remove_overlapping_groups(kfold_data):
    spliter, indices, groups, _ = kfold_data
    for train_index, test_index in spliter:
        train_index, test_index = _remove_overlapping_groups(train_index, test_index, indices, groups)
        train_groups = groups[train_index]
        test_groups = groups[test_index]
        assert len(np.intersect1d(train_groups, test_groups)) == 0


def generic_test_purge(kfold_data, n_purge):
    spliter, indices, groups, data_type = kfold_data
    for train_index, test_index in spliter:
        train_index, test_index = _remove_overlapping_groups(train_index, test_index, indices, groups)
        purged_train_index = _purge(train_index, test_index, indices, groups, n_purge)
        train_groups = groups[train_index]
        test_groups = groups[test_index]
        train_groups_purged = groups[purged_train_index]
        assert len(np.intersect1d(train_groups_purged, test_groups)) == 0
        assert len(np.unique(train_groups)) - len(np.unique(train_groups_purged)) <= n_purge
        assert len(np.unique(train_groups)) - len(np.unique(train_groups_purged)) >= 0
        if any(train_index > max(test_index)):
            expected_purged_groups = max(test_groups) + np.arange(1, n_purge + 1)
            expected_purged_groups = expected_purged_groups[
                expected_purged_groups <= max(groups)
            ]  # Correct for the upper bound of the groups
            assert len(np.intersect1d(train_groups_purged, expected_purged_groups)) == 0
            assert sorted(np.setdiff1d(train_groups, train_groups_purged)) == sorted(expected_purged_groups)
            if data_type == "integer_even":
                assert len(purged_train_index) >= len(train_index) - n_purge * 10
            if data_type == "datetime_even":
                assert len(purged_train_index) >= len(train_index) - n_purge
        else:
            assert len(purged_train_index) == len(train_index)
            assert len(train_groups_purged) == len(train_groups)


def generic_test_embargo(kfold_data, n_embargo):
    spliter, indices, groups, data_type = kfold_data
    for train_index, test_index in spliter:
        train_index, test_index = _remove_overlapping_groups(train_index, test_index, indices, groups)
        embargod_train_index = _embargo(train_index, test_index, indices, groups, n_embargo)
        train_groups = groups[train_index]
        test_groups = groups[test_index]
        train_groups_embargod = groups[embargod_train_index]
        assert len(np.intersect1d(train_groups_embargod, test_groups)) == 0
        assert len(np.unique(train_groups)) - len(np.unique(train_groups_embargod)) >= 0
        assert len(np.unique(train_groups)) - len(np.unique(train_groups_embargod)) <= n_embargo
        if any(train_index < max(test_index)):
            expected_embargod_groups = min(test_groups) - np.arange(1, n_embargo + 1)
            expected_embargod_groups = expected_embargod_groups[
                expected_embargod_groups >= min(groups)
            ]  # Correct for the lower bound of the groups
            assert len(np.intersect1d(train_groups_embargod, expected_embargod_groups)) == 0
            assert sorted(np.setdiff1d(train_groups, train_groups_embargod)) == sorted(expected_embargod_groups)
            if data_type == "integer_even":
                assert len(embargod_train_index) >= len(train_index) - n_embargo * 10
            if data_type == "datetime_even":
                assert len(embargod_train_index) >= len(train_index) - n_embargo
        else:
            assert len(embargod_train_index) == len(train_index)
            assert len(train_groups_embargod) == len(train_groups)


@pytest.mark.parametrize("n_purge", [np.arange(0, 5)] + [pd.timedelta_range(start="1 day", periods=5, freq="D")])
def test_purge_per_n_purge(n_purge, kfold_data):
    correct_int_input_type = isinstance(n_purge, int) and (kfold_data[-1] in ["integer_even", "integer_uneven"])
    correct_datetime_input_type = isinstance(n_purge, pd.Timedelta) and (
        kfold_data[-1] in ["datetime_even", "datetime_uneven"]
    )
    correct_input_type = correct_int_input_type or correct_datetime_input_type
    if correct_input_type:
        generic_test_purge(kfold_data, n_purge)
    else:
        with pytest.raises(TypeError):
            generic_test_purge(kfold_data, n_purge)


@pytest.mark.parametrize("n_embargo", [np.arange(0, 5)] + [pd.timedelta_range(start="1 day", periods=5, freq="D")])
def test_embargo_per_n_embargo(n_embargo, kfold_data):
    correct_int_input_type = isinstance(n_embargo, int) and (kfold_data[-1] in ["integer_even", "integer_uneven"])
    correct_datetime_input_type = isinstance(n_embargo, pd.Timedelta) and (
        kfold_data[-1] in ["datetime_even", "datetime_uneven"]
    )
    correct_input_type = correct_int_input_type or correct_datetime_input_type
    if correct_input_type:
        generic_test_embargo(kfold_data, n_embargo)
    else:
        with pytest.raises(TypeError):
            generic_test_embargo(kfold_data, n_embargo)
