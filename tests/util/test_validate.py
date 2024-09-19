from quantfinlib.util._validate import (
    validate_series_or_1Darray,
    validate_frame_or_2Darray,
    _check_dtype_frame_or_array,
    _check_dtype_series_or_array,
    _check_ndim_shape_1Darray,
    _check_ndim_shape_2Darray,
)

import pytest

import numpy as np
import pandas as pd


def test_validate_series_or_1Darray():
    @validate_series_or_1Darray("x")
    def test_func(x):
        return x

    x = np.array([1, 2, 3])
    assert np.array_equal(test_func(x), x)
    assert np.array_equal(test_func(x=x), x)

    x = pd.Series([1, 2, 3])
    assert np.array_equal(test_func(x), x)
    assert np.array_equal(test_func(x=x), x)

    with pytest.raises(ValueError):
        test_func(1)
        test_func(x=1)

    with pytest.raises(ValueError):
        test_func(np.array([[1, 2], [3, 4]]))
        test_func(x=np.array([[1, 2], [3, 4]]))

    with pytest.raises(ValueError):
        test_func(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        test_func(x=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_validate_frame_or_2Darray():
    @validate_frame_or_2Darray("X")
    def test_func(X):
        return X

    X = np.array([[1, 2], [3, 4]])
    assert np.array_equal(test_func(X), X)  # Test positional arg
    assert np.array_equal(test_func(X=X), X)  # Test keyword arg

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert np.array_equal(test_func(X), X)  # Test positional arg
    assert np.array_equal(test_func(X=X), X)  # Test keyword arg

    with pytest.raises(ValueError):
        test_func(1)
        test_func(X=1)

    with pytest.raises(ValueError):
        test_func(np.array([1, 2, 3]))
        test_func(X=np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        test_func(pd.Series([1, 2, 3]))
        test_func(X=pd.Series([1, 2, 3]))


def test_validate_series_or_1Darray_with_multiple_inputs():

    @validate_series_or_1Darray("x", "y")
    def test_func(x, y):
        return x, y

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert np.array_equal(test_func(x, y), (x, y))
    assert np.array_equal(test_func(x=x, y=y), (x, y))

    x = pd.Series([1, 2, 3])
    y = pd.Series([4, 5, 6])
    assert np.array_equal(test_func(x, y), (x, y))
    assert np.array_equal(test_func(x=x, y=y), (x, y))

    with pytest.raises(ValueError):
        test_func(1, y)
        test_func(x=1, y=y)

    with pytest.raises(ValueError):
        test_func(x, 1)
        test_func(x=x, y=1)

    with pytest.raises(ValueError):
        test_func(np.array([[1, 2], [3, 4]]), y)
        test_func(x=np.array([[1, 2], [3, 4]]), y=y)

    with pytest.raises(ValueError):
        test_func(x, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        test_func(x=x, y=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_validate_frame_or_2Darray_with_multiple_inputs():

    @validate_frame_or_2Darray("X", "Y")
    def test_func(X, Y):
        return X, Y

    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[5, 6], [7, 8]])
    assert np.array_equal(test_func(X, Y), (X, Y))
    assert np.array_equal(test_func(X=X, Y=Y), (X, Y))

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    Y = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    assert np.array_equal(test_func(X, Y), (X, Y))
    assert np.array_equal(test_func(X=X, Y=Y), (X, Y))

    with pytest.raises(ValueError):
        test_func(1, Y)
        test_func(X=1, Y=Y)

    with pytest.raises(ValueError):
        test_func(X, 1)
        test_func(X=X, Y=1)

    with pytest.raises(ValueError):
        test_func(np.array([1, 2, 3]), Y)
        test_func(X=np.array([1, 2, 3]), Y=Y)

    with pytest.raises(ValueError):
        test_func(X, pd.Series([5, 6, 7]))
        test_func(X=X, Y=pd.Series([5, 6, 7]))


def test_check_dtype_series_or_array():
    x = [1, 2, 3]
    with pytest.raises(ValueError):
        _check_dtype_series_or_array(x)

    x = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        _check_dtype_series_or_array(x)

    x = np.array([1, 2, 3])
    _check_dtype_series_or_array(x)

    x = pd.Series([1, 2, 3])
    _check_dtype_series_or_array(x)


def test_check_ndim_shape_1Darray():
    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        _check_ndim_shape_1Darray(x)

    x = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        _check_ndim_shape_1Darray(x)

    x = np.array([1, 2, 3])
    _check_ndim_shape_1Darray(x)

    x = pd.Series([1, 2, 3])
    _check_ndim_shape_1Darray(x)

    x = pd.Series([1])
    with pytest.raises(ValueError):
        _check_ndim_shape_1Darray(x)

    x = np.array([1])
    with pytest.raises(ValueError):
        _check_ndim_shape_1Darray(x)


def test_check_dtype_frame_or_array():
    X = [1, 2, 3]
    with pytest.raises(ValueError):
        _check_dtype_frame_or_array(X)

    X = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        _check_dtype_frame_or_array(X)

    X = np.array([1, 2, 3])
    _check_dtype_frame_or_array(X)

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _check_dtype_frame_or_array(X)


def test_check_ndim_shape_2Darray():
    X = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        _check_ndim_shape_2Darray(X)

    X = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        _check_ndim_shape_2Darray(X)

    X = np.array([[1, 2], [3, 4]])
    _check_ndim_shape_2Darray(X)

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _check_ndim_shape_2Darray(X)

    X = np.array([[1], [2], [3]])
    with pytest.raises(ValueError):
        _check_ndim_shape_2Darray(X)
