import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, multivariate_t

from quantfinlib.distance.information import (
    compute_entropies,
    kl_divergence_xy,
    mutual_info,
    var_info,
    _get_nb_bins_from_xy,
    _get_optimal_nb_bins,
)

np.random.seed(123456789)


@pytest.mark.parametrize("rho", np.linspace(-0.7, 0.7, 10))
def test_information_for_correlated_gaussians(rho):
    num_realizations = 20
    mi_values = []
    vi_values = []
    kl_values = []
    for _ in range(num_realizations):
        correlated_data = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 10000)
        x, y = correlated_data[:, 0], correlated_data[:, 1]
        mi_values.append(mutual_info(x, y, norm=True))
        vi_values.append(var_info(x, y, norm=True))
        kl_values.append(kl_divergence_xy(x, y))
    expected_mi = -0.5 * np.log(1 - rho**2)
    expected_vi = np.log(2 * np.pi * np.e) + np.log(1 - rho**2)
    expected_hxy = np.log(2 * np.pi * np.e) + 0.5 * np.log(1 - rho**2)
    expected_vi_normalized = expected_vi / expected_hxy
    expected_hx = 0.5 * np.log(2 * np.pi * np.e)
    expected_hy = 0.5 * np.log(2 * np.pi * np.e)
    expected_mi_normalized = expected_mi / (expected_hx * expected_hy) ** 0.5
    expected_kl = 0
    np.testing.assert_almost_equal(np.mean(mi_values), expected_mi_normalized, decimal=1)
    np.testing.assert_almost_equal(np.mean(vi_values), expected_vi_normalized, decimal=1)
    np.testing.assert_almost_equal(np.mean(kl_values), expected_kl, decimal=1)


def test_information_for_uncorrelated():
    x = np.random.normal(size=1000)
    e = np.random.normal(loc=0, scale=0.1, size=1000)
    y = 0 * x + e
    nmi = mutual_info(x, y, norm=True)
    nmi_expected = 0
    np.testing.assert_almost_equal(nmi, nmi_expected, decimal=1)
    vi = var_info(x, y, norm=True)
    vi_expected = 1
    np.testing.assert_almost_equal(vi, vi_expected, decimal=1)


def test_information_for_y_abs_x():
    x = np.random.normal(size=10000)
    e = np.random.normal(loc=0, scale=0.1, size=10000)
    y = 100 * np.abs(x) + e
    nmi = mutual_info(x, y, norm=True)
    assert nmi > 0.5, "expected nmi to be greater than .5"
    nvi = var_info(x, y, norm=True)
    assert nvi <= 1, "expected vi to be less than 1"


@pytest.mark.parametrize("dist", ["multivariate_t", "multivariate_normal"])
def test_range_generic_info(dist):
    if dist == "multivariate_t":
        data = multivariate_t([1.0, -0.5], [[2.1, 0.3], [0.3, 1.5]], df=2).rvs(size=1000)
    else:
        data = multivariate_normal([1.0, -0.5], [[2.1, 0.3], [0.3, 1.5]]).rvs(size=1000)
    x, y = data[:, 0], data[:, 1]
    nmi = mutual_info(x, y, norm=True)
    assert 0 <= nmi <= 1, "expected nmi to be greater than or equal to 0 or less than or equal to 1."
    nvi = var_info(x, y, norm=True)
    assert 0 <= nvi <= 1, "expected vi to be less than or equal to 1 and greater than or equal to 0."
    mi = mutual_info(x, y, norm=False)
    assert 0 <= mi, "expected mi to be greater than or equal to 0."
    vi = var_info(x, y, norm=False)
    assert 0 <= vi, "expected vi to be greater than or equal to 0."
    assert isinstance(nmi, float), "expected nmi to be a float."
    assert isinstance(nvi, float), "expected nvi to be a float."
    assert isinstance(mi, float), "expected mi to be a float."
    assert isinstance(vi, float), "expected vi to be a float."
    kl = kl_divergence_xy(x, y)
    assert 0 <= kl, "expected kl to be greater than or equal to 0."
    assert isinstance(kl, float), "expected kl to be a float."


def test_metric_properties():
    data = multivariate_normal([0, 0, 0], [[1, 0.1, 0.3], [0.1, 1, 0.1], [0.3, 0.1, 1]]).rvs(size=1000)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    vi_xy = var_info(x, y, norm=False)
    vi_yz = var_info(y, z, norm=False)
    vi_xz = var_info(x, z, norm=False)
    assert vi_xy + vi_yz >= vi_xz, "triangle inequality not satisfied for variation of information."
    assert vi_xy + vi_xz >= vi_yz, "triangle inequality not satisfied for variation of information."
    assert vi_yz + vi_xz >= vi_xy, "triangle inequality not satisfied for variation of information."
    nvi_xy = var_info(x, y, norm=True)
    nvi_yz = var_info(y, z, norm=True)
    nvi_xz = var_info(x, z, norm=True)
    assert nvi_xy + nvi_yz >= nvi_xz, "triangle inequality not satisfied for normalized variation of information."
    assert nvi_xy + nvi_xz >= nvi_yz, "triangle inequality not satisfied for normalized variation of information."
    assert nvi_yz + nvi_xz >= nvi_xy, "triangle inequality not satisfied for normalized variation of information."
    vi_yx = var_info(y, x, norm=False)
    np.testing.assert_almost_equal(
        vi_xy, vi_yx, decimal=5, err_msg="expected normalized mutual information to be symmetric."
    )
    nvi_yx = var_info(y, x, norm=True)
    np.testing.assert_almost_equal(
        nvi_xy, nvi_yx, decimal=5, err_msg="expected normalized mutual information to be symmetric."
    )
    mi_xy = mutual_info(x, y, norm=False)
    mi_yx = mutual_info(y, x, norm=False)
    np.testing.assert_almost_equal(
        mi_xy, mi_yx, decimal=5, err_msg="expected normalized mutual information to be symmetric."
    )
    nmi_xy = mutual_info(x, y, norm=True)
    nmi_yx = mutual_info(y, x, norm=True)
    np.testing.assert_almost_equal(
        nmi_xy, nmi_yx, decimal=5, err_msg="expected normalized mutual information to be symmetric."
    )


def test_entropies():
    x = np.random.normal(0, 1, 1000)
    y = 10 * x + np.random.normal(0, 0.1, 1000)
    h_x, h_y, h_xy = compute_entropies(x, y, bins=10)
    assert h_x > 0, "expected entropy of x to be greater than 0."
    assert h_y > 0, "expected entropy of y to be greater than 0."
    assert h_xy > 0, "expected joint entropy of x and y to be greater than 0."
    assert h_x <= h_xy, "expected entropy of x to be less than or equal to joint entropy of x and y."
    assert h_y <= h_xy, "expected entropy of y to be less than or equal to joint entropy of x and y."
    assert isinstance(h_x, float), "expected entropy of x to be a float."
    assert isinstance(h_y, float), "expected entropy of y to be a float."
    assert isinstance(h_xy, float), "expected joint entropy of x and y to be a float."
    _, _, h_yx = compute_entropies(y, x, bins=10)
    np.testing.assert_almost_equal(
        h_yx, h_xy, decimal=5, err_msg="expected joint entropy of xy to be the same as joint entropy of yx."
    )


@pytest.fixture
def data():
    x_, e_ = np.random.normal(size=100000), np.random.normal(0, 0.1, 100000)
    data = [[x_, 10 * x_ + e_], [x_, 10 * x_**2 + e_], [x_, 10 * np.abs(x_) + e_]]
    return data


@pytest.mark.parametrize("index", [0, 1, 2])
def test_input_type(index, data):
    x, y = data[index]
    x_pd, y_pd = pd.Series(x), pd.Series(y)
    nmi = mutual_info(x, y, norm=True)
    nmi_pd = mutual_info(x_pd, y_pd, norm=True)
    nvi = var_info(x, y, norm=True)
    nvi_pd = var_info(x_pd, y_pd, norm=True)
    mi = mutual_info(x, y, norm=False)
    mi_pd = mutual_info(x_pd, y_pd, norm=False)
    vi = var_info(x, y, norm=False)
    vi_pd = var_info(x_pd, y_pd, norm=False)
    kl = kl_divergence_xy(x, y)
    kl_pd = kl_divergence_xy(x_pd, y_pd)
    h_x, h_y, h_xy = compute_entropies(x, y, bins=10)
    h_x_pd, h_y_pd, h_xy_pd = compute_entropies(x_pd, y_pd, bins=10)
    np.testing.assert_almost_equal(
        nmi, nmi_pd, decimal=5, err_msg="expected mutual information to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        nvi, nvi_pd, decimal=5, err_msg="expected normalized mutual information to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        mi, mi_pd, decimal=5, err_msg="expected mutual information to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        vi, vi_pd, decimal=5, err_msg="expected variation of information to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        kl, kl_pd, decimal=5, err_msg="expected kl divergence to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        h_x, h_x_pd, decimal=5, err_msg="expected entropy of x to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        h_y, h_y_pd, decimal=5, err_msg="expected entropy of y to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        h_xy, h_xy_pd, decimal=5, err_msg="expected joint entropy of x and y to be the same for numpy and pandas."
    )
    # Test with user defined bin number
    nmi_10 = mutual_info(x, y, norm=True, bins=10)
    nmi_pd_10 = mutual_info(x_pd, y_pd, norm=True, bins=10)
    nvi_10 = var_info(x, y, norm=True, bins=10)
    nvi_pd_10 = var_info(x_pd, y_pd, norm=True, bins=10)
    mi_10 = mutual_info(x, y, norm=False, bins=10)
    mi_pd_10 = mutual_info(x_pd, y_pd, norm=False, bins=10)
    vi_10 = var_info(x, y, norm=False, bins=10)
    vi_pd_10 = var_info(x_pd, y_pd, norm=False, bins=10)
    kl_10 = kl_divergence_xy(x, y, bins=10)
    kl_pd_10 = kl_divergence_xy(x_pd, y_pd, bins=10)
    np.testing.assert_almost_equal(
        nmi_10, nmi_pd_10, decimal=5, err_msg="expected mutual information to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        nvi_10,
        nvi_pd_10,
        decimal=5,
        err_msg="expected normalized mutual information to be the same for numpy and pandas.",
    )
    np.testing.assert_almost_equal(
        mi_10, mi_pd_10, decimal=5, err_msg="expected mutual information to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        vi_10, vi_pd_10, decimal=5, err_msg="expected variation of information to be the same for numpy and pandas."
    )
    np.testing.assert_almost_equal(
        kl_10, kl_pd_10, decimal=5, err_msg="expected kl divergence to be the same for numpy and pandas."
    )


def test_get_nb_bins_from_xy():
    x = np.random.normal(size=1000)
    calculated_nbins = _get_nb_bins_from_xy(x, x)
    expected_nbins = _get_optimal_nb_bins(x.shape[0])
    assert calculated_nbins == expected_nbins, "expected number of bins to be optimal number of bins."


def test_univariate_get_optimal_nb_bins():
    n_obs = 10
    z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs**2) ** 0.5) ** (1 / 3.0)
    n_bins = int(round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3))
    expected_nbins = _get_optimal_nb_bins(n_obs)
    assert n_bins == expected_nbins, "expected number of bins to be optimal number of bins."
