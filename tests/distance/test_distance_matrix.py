from itertools import product

from quantfinlib.distance.distance_matrix import (
    _calculate_correlation,
    get_corr_distance_matrix,
    get_info_distance_matrix,
)

from quantfinlib.distance.correlation import (
    corr_to_abs_angular_dist,
    corr_to_angular_dist,
    corr_to_squared_angular_dist,
)

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr


CORR_TO_DIST_METHOD = ["angular", "abs_angular", "squared_angular"]
CORR_TO_DIST_METHOD_MAP = dict(
    zip(CORR_TO_DIST_METHOD, [corr_to_angular_dist, corr_to_abs_angular_dist, corr_to_squared_angular_dist])
)
CORR_METHOD = ["spearman", "pearson"]
INFO_METHOD = ["mutual_info", "var_info"]

np.random.seed(42)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_corr_vs_scipy(method):
    X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 10000)
    corr = _calculate_correlation(X, corr_method=method)
    calculated = corr[0][1]
    expected = spearmanr(X[:, 0], X[:, 1])[0] if method == "spearman" else np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    np.testing.assert_almost_equal(calculated, expected, decimal=2)


@pytest.fixture
def input_cov():
    return np.array([[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]])


def rho_to_nvi(rho):
    expected_vi = np.log(2 * np.pi * np.e) + np.log(1 - rho**2)
    expected_hxy = np.log(2 * np.pi * np.e) + 0.5 * np.log(1 - rho**2)
    return expected_vi / expected_hxy


def rho_to_nmi(rho):
    expected_mi = -0.5 * np.log(1 - rho**2)
    expected_hx = 0.5 * np.log(2 * np.pi * np.e)
    expected_hy = 0.5 * np.log(2 * np.pi * np.e)
    return expected_mi / (expected_hx * expected_hy) ** 0.5


@pytest.fixture
def expected_nvi():
    return np.array(
        [
            [0, rho_to_nvi(0.5), rho_to_nvi(0.3)],
            [rho_to_nvi(0.5), 0, rho_to_nvi(0.2)],
            [rho_to_nvi(0.3), rho_to_nvi(0.2), 0],
        ]
    )


@pytest.fixture
def expected_nmi():
    return np.array(
        [
            [0, rho_to_nmi(0.5), rho_to_nmi(0.3)],
            [rho_to_nmi(0.5), 0, rho_to_nmi(0.2)],
            [rho_to_nmi(0.3), rho_to_nmi(0.2), 0],
        ]
    )


@pytest.fixture
def multivar_normal_X(input_cov):
    return np.random.multivariate_normal([0, 0, 0], input_cov, 100000)


@pytest.mark.parametrize("corr_method, corr_to_dist_method", list(product(CORR_METHOD, CORR_TO_DIST_METHOD)))
def test_corr_distance_matrix(corr_method, corr_to_dist_method, multivar_normal_X, input_cov):
    df = pd.DataFrame(multivar_normal_X)
    dist_matrix = get_corr_distance_matrix(df, corr_to_dist_method=corr_to_dist_method, corr_method=corr_method).values
    assert dist_matrix.shape == input_cov.shape
    assert np.allclose(dist_matrix, dist_matrix.T), "Expected distance matrix to be symmetric."
    assert np.all(dist_matrix >= 0), "Expected all elements of the distance matrix to be non-negative."
    assert np.all(dist_matrix <= 1), "Expected all elements of the distance matrix to be less than or equal to 1."
    np.testing.assert_equal(
        np.diag(dist_matrix), np.array([0, 0, 0]), "Expected the diagonal elements of the distance matrix to be 0."
    )

    if corr_to_dist_method == "angular":
        expected = (0.5 * (1 - input_cov)) ** 0.5
    elif corr_to_dist_method == "abs_angular":
        expected = (1 - np.abs(input_cov)) ** 0.5
    else:
        expected = (1 - input_cov**2) ** 0.5
    if corr_method == "pearson":
        assert np.allclose(
            dist_matrix, expected, atol=1e-2
        ), "Expected distance matrix to be close to the expected value."


@pytest.mark.parametrize("info_method", INFO_METHOD)
def test_info_distance_matrix(info_method, multivar_normal_X, input_cov, expected_nmi, expected_nvi):
    df = pd.DataFrame(multivar_normal_X)
    print(info_method)
    dist_matrix = get_info_distance_matrix(df, method=info_method)
    assert dist_matrix.shape == input_cov.shape
    assert np.allclose(dist_matrix, dist_matrix.T), "Expected distance matrix to be symmetric."
    assert np.all(dist_matrix >= 0), "Expected all elements of the distance matrix to be non-negative."
    if info_method == "var_info":
        assert np.all(dist_matrix <= 1), "Expected all elements of the distance matrix to be less than or equal to 1."
        np.testing.assert_equal(
            np.diag(dist_matrix),
            np.zeros((df.shape[1])),
            "Expected the diagonal elements of the distance matrix to be 0.",
        )
        np.allclose(
            dist_matrix, expected_nvi, atol=1e-2
        ), "Expected estimated distance matrix of multivariate Gaussian to be close to the expected matrix derived from theory."
    else:
        np.testing.assert_equal(
            np.diag(dist_matrix),
            np.ones((df.shape[1])),
            "Expected the diagonal elements of the distance matrix to be 1.",
        )
        np.allclose(
            dist_matrix, expected_nmi, atol=1e-2
        ), "Expected estimated distance matrix of multivariate Gaussian to be close to the expected matrix derived from theory."


@pytest.mark.parametrize("dtype", ["numpy array", "pandas dataframe"])
def test_check_in_out_type(dtype):
    dtype_map = {"numpy array": np.ndarray, "pandas dataframe": pd.DataFrame}
    X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 10000)
    if dtype == "pandas dataframe":
        X = pd.DataFrame(X)
    dist_corr_matrix = get_corr_distance_matrix(X, corr_to_dist_method="angular", corr_method="pearson")
    dist_info_matrix = get_info_distance_matrix(X, method="mutual_info")
    assert isinstance(dist_corr_matrix, dtype_map[dtype]), f"Expected output to be a {dtype}."
    assert isinstance(dist_info_matrix, dtype_map[dtype]), f"Expected output to be a {dtype}."


def corr_transformer(corr):
    corr_ = corr.copy()
    corr_[0, 1] = corr[1, 2]
    corr_[1, 0] = corr[2, 1]
    corr_[0, 2] = corr[0, 1]
    corr_[2, 0] = corr[1, 0]
    return corr_


@pytest.mark.parametrize("corr_method, corr_to_dist_method", list(product(CORR_METHOD, CORR_TO_DIST_METHOD)))
def test_dist_with_corr_transformer(corr_method, corr_to_dist_method, multivar_normal_X, input_cov):
    dist_matrix = get_corr_distance_matrix(
        multivar_normal_X, corr_to_dist_method=corr_to_dist_method, corr_method=corr_method, corr_transformer=corr_transformer
    )
    assert np.allclose(dist_matrix, dist_matrix.T), "Expected distance matrix to be symmetric."
    assert np.all(dist_matrix >= 0), "Expected all elements of the distance matrix to be non-negative."
    assert np.all(dist_matrix <= 1), "Expected all elements of the distance matrix to be less than or equal to 1."
    np.testing.assert_equal(
        np.diag(dist_matrix), np.array([0, 0, 0]), "Expected the diagonal elements of the distance matrix to be 0."
    )
    transformed_cov = corr_transformer(input_cov)
    if corr_method == "pearson":
        expected = CORR_TO_DIST_METHOD_MAP[corr_to_dist_method](transformed_cov)
        assert np.allclose(
            dist_matrix, expected, atol=1e-2
        ), "Expected distance matrix to be close to the expected value."


def test_wrong_input_shape():
    X = np.random.normal(0, 1, (1000, 2, 2))

    with pytest.raises(ValueError):
        get_corr_distance_matrix(X, corr_to_dist_method="angular", corr_method="pearson")
    with pytest.raises(ValueError):
        get_info_distance_matrix(X, method="mutual_info")


def test_wrong_corr_to_dist_method():
    X = np.random.normal(0, 1, (1000, 2))
    with pytest.raises(AssertionError, match="Invalid corr_to_dist method. Must be one of angular, abs_angular, squared_angular."):
        get_corr_distance_matrix(X, corr_to_dist_method="wrong", corr_method="pearson")


def test_wrong_info_method():
    X = np.random.normal(0, 1, (1000, 2))
    with pytest.raises(AssertionError, match="Invalid method. Must be one of mutual_info or var_info."):
        get_info_distance_matrix(X, method="wrong")


def test_wrong_corr_method():
    X = np.random.normal(0, 1, (1000, 2))
    with pytest.raises(AssertionError, match="Invalid correlation method. Must be pearson or spearman."):
        get_corr_distance_matrix(X, corr_to_dist_method="angular", corr_method="wrong")
