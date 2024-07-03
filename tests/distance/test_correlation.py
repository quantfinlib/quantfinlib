import numpy as np
from quantfinlib.distance.correlation import (
    corr_to_angular_dist,
    corr_to_abs_angular_dist,
    corr_to_squared_angular_dist,
    pair_abs_angular_distance,
    pair_angular_distance,
    pair_squared_angular_distance,
)


def test_generic_corr_to_angular_dist():
    assert corr_to_angular_dist(1) == 0
    assert corr_to_angular_dist(0) == np.sqrt(.5)
    assert corr_to_angular_dist(-1) == 1


def test_generic_corr_to_abs_angular_dist():
    assert corr_to_abs_angular_dist(1) == 0
    assert corr_to_abs_angular_dist(0) == 1
    assert corr_to_abs_angular_dist(-1) == 0


def test_generic_corr_to_squared_angular_dist():
    assert corr_to_squared_angular_dist(1) == 0
    assert corr_to_squared_angular_dist(0) == 1
    assert corr_to_squared_angular_dist(-1) == 0


def test_pair_angular_distance():
    x = np.array([1, 0])
    y = np.array([0, 1])
    np.testing.assert_almost_equal(pair_angular_distance(x, y), 1, decimal=2)

    x = np.array([1, 0])
    y = np.array([1, 0])
    np.testing.assert_almost_equal(pair_angular_distance(x, y), 0, decimal=2)

    x = np.array([1, 0])
    y = np.array([-1, 0])
    np.testing.assert_almost_equal(pair_angular_distance(x, y), 1, decimal=2)

    x = np.random.rand(20000)
    y = np.random.rand(20000)
    np.testing.assert_almost_equal(pair_angular_distance(x, y), np.sqrt(0.5), decimal=2)


def test_pair_abs_angular_distance():

    x = np.array([1, 0])
    y = np.array([0, 1])
    np.testing.assert_almost_equal(pair_abs_angular_distance(x, y), 0, decimal=2)

    x = np.array([1, 0])
    y = np.array([1, 0])
    np.testing.assert_almost_equal(pair_abs_angular_distance(x, y), 0, decimal=2)

    x = np.array([1, 0])
    y = np.array([-1, 0])
    np.testing.assert_almost_equal(pair_abs_angular_distance(x, y), 0, decimal=2)

    x = np.random.rand(20000)
    y = np.random.rand(20000)
    np.testing.assert_almost_equal(pair_abs_angular_distance(x, y), 1, decimal=2)


def test_pair_squared_angular_distance():
    x = np.array([1, 0])
    y = np.array([0, 1])
    np.testing.assert_almost_equal(pair_squared_angular_distance(x, y), 0, decimal=2)

    x = np.array([1, 0])
    y = np.array([1, 0])
    np.testing.assert_almost_equal(pair_squared_angular_distance(x, y), 0, decimal=2)

    x = np.array([1, 0])
    y = np.array([-1, 0])
    np.testing.assert_almost_equal(pair_squared_angular_distance(x, y), 0, decimal=2)

    x = np.random.rand(20000)
    y = np.random.rand(20000)
    np.testing.assert_almost_equal(pair_squared_angular_distance(x, y), 1, decimal=2)


np.random.seed(123)

def test_pair_distance_matrix_with_random_data():
    input_corr = [[1, 0, -1], [0, 1, 0], [-1, 0, 1]]
    x = np.random.multivariate_normal([0, 0, 0], input_corr, 100000)
    sample_corr = np.corrcoef(x.T)
    np.fill_diagonal(sample_corr, 1)
    angular_dist = corr_to_angular_dist(sample_corr)
    abs_angular_dist = corr_to_abs_angular_dist(sample_corr)
    squared_angular_dist = corr_to_squared_angular_dist(sample_corr)

    np.testing.assert_equal(angular_dist.shape , sample_corr.shape, "expected angular_dist to have the same shape as the input correlation matrix.")
    np.testing.assert_equal(abs_angular_dist.shape, sample_corr.shape, "expected abs_angular_dist to have the same shape as the input correlation matrix.")
    np.testing.assert_equal(squared_angular_dist.shape, sample_corr.shape, "expected squared_angular_dist to have the same shape as the input correlation matrix.")

    np.testing.assert_almost_equal(angular_dist.T, angular_dist, err_msg = "expected angular_dist to be symmetric.")
    np.testing.assert_almost_equal(abs_angular_dist.T, abs_angular_dist, err_msg ="expected abs_angular_dist to be symmetric.")
    np.testing.assert_almost_equal(squared_angular_dist.T, squared_angular_dist, err_msg ="expected squared_angular_dist to be symmetric.")

    assert np.all(angular_dist >= 0), "expected angular_dist to be greater than or equal to 0."
    assert np.all(abs_angular_dist >= 0), "expected abs_angular_dist to be greater than or equal to 0."
    assert np.all(squared_angular_dist >= 0), "expected squared_angular_dist to be greater than or equal to 0."

    assert np.all(angular_dist <= 1), "expected angular_dist to be less than or equal to 1."
    assert np.all(abs_angular_dist <= 1), "expected abs_angular_dist to be less than or equal to 1."
    assert np.all(squared_angular_dist <= 1), "expected squared_angular_dist to be less than or equal to 1."

    np.testing.assert_equal(np.diag(angular_dist), np.array([0, 0, 0]), "expected the diagonal elements of angular_dist to be 0.")
    np.testing.assert_equal(np.diag(abs_angular_dist), np.array([0, 0, 0]), "expected the diagonal elements of angular_dist to be 0.")
    np.testing.assert_equal(np.diag(squared_angular_dist), np.array([0, 0, 0]), "expected the diagonal elements of angular_dist to be 0.")

    expected_angular_dist = np.array([[0, .5**.5, 1], [.5**.5, 0, .5**.5], [1, .5**.5, 0]])
    expected_abs_angular_dist = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    expected_squared_angular_dist = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    np.testing.assert_almost_equal(angular_dist, expected_angular_dist, decimal=2)
    np.testing.assert_almost_equal(abs_angular_dist, expected_abs_angular_dist, decimal=2)
    np.testing.assert_almost_equal(squared_angular_dist, expected_squared_angular_dist, decimal=2)
