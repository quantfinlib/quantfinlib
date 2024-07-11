from quantfinlib.util._validate import validate_series_or_1Darray, validate_frame_or_2Darray

import pytest

import numpy as np
import pandas as pd

def test_validate_series_or_1Darray():
    @validate_series_or_1Darray("x")
    def test_func(x):
        return x

    x = np.array([1, 2, 3])
    assert np.array_equal(test_func(x), x)

    x = pd.Series([1, 2, 3])
    assert np.array_equal(test_func(x), x)

    with pytest.raises(ValueError):
        test_func(1)

    with pytest.raises(ValueError):
        test_func(np.array([[1, 2], [3, 4]]))

    with pytest.raises(ValueError):
        test_func(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_validate_frame_or_2Darray():
    @validate_frame_or_2Darray("X")
    def test_func(X):
        return X

    X = np.array([[1, 2], [3, 4]])
    assert np.array_equal(test_func(X), X)

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert np.array_equal(test_func(X), X)

    with pytest.raises(ValueError):
        test_func(1)

    with pytest.raises(ValueError):
        test_func(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        test_func(pd.Series([1, 2, 3]))


def test_validate_series_or_1Darray_with_multiple_inputs():

    @validate_series_or_1Darray("x", "y")
    def test_func(x, y):
        return x, y

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert np.array_equal(test_func(x, y), (x, y))

    x = pd.Series([1, 2, 3])
    y = pd.Series([4, 5, 6])
    assert np.array_equal(test_func(x, y), (x, y))

    with pytest.raises(ValueError):
        test_func(1, y)

    with pytest.raises(ValueError):
        test_func(x, 1)

    with pytest.raises(ValueError):
        test_func(np.array([[1, 2], [3, 4]]), y)

    with pytest.raises(ValueError):
        test_func(x, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_validate_frame_or_2Darray_with_multiple_inputs():

    @validate_frame_or_2Darray("X", "Y")
    def test_func(X, Y):
        return X, Y

    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[5, 6], [7, 8]])
    assert np.array_equal(test_func(X, Y), (X, Y))

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    Y = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    assert np.array_equal(test_func(X, Y), (X, Y))

    with pytest.raises(ValueError):
        test_func(1, Y)

    with pytest.raises(ValueError):
        test_func(X, 1)

    with pytest.raises(ValueError):
        test_func(np.array([1, 2, 3]), Y)

    with pytest.raises(ValueError):
        test_func(X, pd.Series([5, 6, 7]))
