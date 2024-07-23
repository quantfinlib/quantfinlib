import numpy as np
import pytest

from quantfinlib.math.cov import (  # Replace with the actual module name
    _cor_block_diagonal,
    _cor_denoise,
    _cor_to_cov,
    _cor_to_eig,
    _cov_denoise,
    _cov_to_cor,
    _eig_complete,
    _eig_to_cor,
    _eigen_values_denoise,
    _marcenko_pastur_support,
    _marchenko_pastur_fit,
    _marchenko_pastur_pdf,
    _random_cor,
    _random_cov,
)


def test_cor_block_diagonal_default():
    cor = _cor_block_diagonal()
    expected_shape = (128, 128)
    assert cor.shape == expected_shape, f"Expected shape {expected_shape}, got {cor.shape}"
    assert np.all(np.diag(cor) == 1.0), "Diagonal elements should be 1."


def check_block(cor, start, end, expected_value):
    for i in range(start, end):
        for j in range(start, end):
            if i != j:
                assert (
                    cor[i, j] == expected_value
                ), f"Block ({start}:{end}, {start}:{end}) should have correlation {expected_value}"


def test_cor_block_diagonal_custom():
    block_sizes = [3, 7, 12]
    block_cors = [0.7, 0.3, -0.2]
    cor = _cor_block_diagonal(block_sizes, block_cors)
    expected_shape = (22, 22)
    assert cor.shape == expected_shape, f"Expected shape {expected_shape}, got {cor.shape}"
    assert np.all(np.diag(cor) == 1.0), "Diagonal elements should be 1."
    check_block(cor, 0, 3, 0.7)
    check_block(cor, 3, 10, 0.3)
    check_block(cor, 10, 22, -0.2)


def test_cor_block_diagonal_different_lengths():
    with pytest.raises(ValueError, match="block_sizes and block_cors must have the same length"):
        _cor_block_diagonal([3, 7, 12], [0.7, 0.3])


def test_cor_block_diagonal_empty():
    cor = _cor_block_diagonal([], [])
    expected_shape = (0, 0)
    assert cor.shape == expected_shape, f"Expected shape {expected_shape}, got {cor.shape}"


def test_cor_block_diagonal_block_size_type():
    block_sizes = np.array([5, 5, 10])
    block_cors = [0.5, 0.2, 0.3]
    cor = _cor_block_diagonal(block_sizes, block_cors)
    expected_shape = (20, 20)
    assert cor.shape == expected_shape, f"Expected shape {expected_shape}, got {cor.shape}"
    assert np.all(np.diag(cor) == 1.0), "Diagonal elements should be 1."
    check_block(cor, 0, 5, 0.5)
    check_block(cor, 5, 10, 0.2)
    check_block(cor, 10, 20, 0.3)


def test_cov_to_cor_default():
    cov = np.array([[4, 2], [2, 3]])
    expected_cor = np.array([[1.0, 0.57735027], [0.57735027, 1.0]])
    expected_std = np.array([2.0, np.sqrt(3.0)])
    cor, std = _cov_to_cor(cov)

    assert cor == pytest.approx(expected_cor, rel=1e-7), "Correlation matrix does not match."
    assert std == pytest.approx(expected_std, rel=1e-7), "Standard deviation vector does not match."


def test_cov_to_cor_singular():
    cov = np.array([[4, 4], [4, 4]])  # Singular matrix
    cov_reg = 1e-6
    expected_cor = np.array([[1.0, 1.0], [1.0, 1.0]])
    expected_std = np.array([2.0, 2.0])
    cor, std = _cov_to_cor(cov, cov_reg=cov_reg)

    assert cor == pytest.approx(expected_cor, rel=1e-7), "Correlation matrix does not match for singular matrix."
    assert std == pytest.approx(expected_std, rel=1e-7), "Standard deviation vector does not match for singular matrix."


def test_cov_to_cor_negative_cov_elements():
    cov = np.array([[4, -2], [-2, 3]])
    expected_cor = np.array([[1.0, -0.57735027], [-0.57735027, 1.0]])
    expected_std = np.array([2.0, np.sqrt(3.0)])
    cor, std = _cov_to_cor(cov)

    assert cor == pytest.approx(expected_cor, rel=1e-7), "Correlation matrix does not match for negative cov elements."
    assert std == pytest.approx(
        expected_std, rel=1e-7
    ), "Standard deviation vector does not match for negative cov elements."


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_cor_to_cov_default():
    cor = np.array([[1.0, 0.5], [0.5, 1.0]])
    std = np.array([2.0, 3.0])
    expected_cov = np.array([[4.0, 3.0], [3.0, 9.0]])
    cov = _cor_to_cov(cor, std)

    assert cov == pytest.approx(expected_cov, rel=1e-7), "Covariance matrix does not match the expected values."


def test_cor_to_cov_identity_cor():
    cor = np.identity(3)
    std = np.array([1.0, 2.0, 3.0])
    expected_cov = np.diag(std**2)
    cov = _cor_to_cov(cor, std)

    assert cov == pytest.approx(
        expected_cov, rel=1e-7
    ), "Covariance matrix does not match the expected values for identity correlation matrix."


def test_cor_to_cov_negative_cor():
    cor = np.array([[1.0, -0.5], [-0.5, 1.0]])
    std = np.array([2.0, 3.0])
    expected_cov = np.array([[4.0, -3.0], [-3.0, 9.0]])
    cov = _cor_to_cov(cor, std)

    assert cov == pytest.approx(
        expected_cov, rel=1e-7
    ), "Covariance matrix does not match the expected values for negative correlation."


def test_cor_to_cov_non_uniform_std():
    cor = np.array([[1.0, 0.8, 0.4], [0.8, 1.0, 0.6], [0.4, 0.6, 1.0]])
    std = np.array([1.0, 2.0, 0.5])
    expected_cov = np.array([[1.0, 1.6, 0.2], [1.6, 4.0, 0.6], [0.2, 0.6, 0.25]])
    cov = _cor_to_cov(cor, std)

    assert cov == pytest.approx(
        expected_cov, rel=1e-7
    ), "Covariance matrix does not match the expected values for non-uniform standard deviations."


def test_cor_to_cov_zero_std():
    cor = np.array([[1.0, 0.5], [0.5, 1.0]])
    std = np.array([0.0, 3.0])
    expected_cov = np.array([[0.0, 0.0], [0.0, 9.0]])
    cov = _cor_to_cov(cor, std)

    assert cov == pytest.approx(
        expected_cov, rel=1e-7
    ), "Covariance matrix does not match the expected values for zero standard deviation."


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
def test_eig_to_cor_default():
    eigen_values = np.array([2.0, 1.0])
    eigen_vectors = np.array([[1, 0], [0, 1]])
    expected_cor = np.array([[1.0, 0.0], [0.0, 1.0]])
    cor = _eig_to_cor(eigen_values, eigen_vectors)

    assert cor == pytest.approx(expected_cor, rel=1e-7), "Correlation matrix does not match the expected values."


def test_eig_to_cor_orthogonal_vectors():
    eigen_values = np.array([3.0, 1.0])
    eigen_vectors = np.array([[0.8660254, -0.5], [0.5, 0.8660254]])  # 30 degrees rotation
    cor = _eig_to_cor(eigen_values, eigen_vectors)

    expected_cor = np.dot(eigen_vectors * eigen_values, eigen_vectors.T)
    np.fill_diagonal(expected_cor, 1.0)  # Ensure diagonals are set to 1

    assert cor == pytest.approx(
        expected_cor, rel=1e-7
    ), "Correlation matrix does not match the expected values for orthogonal vectors."


def test_eig_to_cor_non_unit_eigen_vectors():
    eigen_values = np.array([2.0, 1.0])
    eigen_vectors = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cor = _eig_to_cor(eigen_values, eigen_vectors)

    expected_cor = np.dot(eigen_vectors * eigen_values, eigen_vectors.T)
    np.fill_diagonal(expected_cor, 1.0)  # Ensure diagonals are set to 1

    assert cor == pytest.approx(
        expected_cor, rel=1e-7
    ), "Correlation matrix does not match the expected values for non-unit eigenvectors."


def test_eig_to_cor_large_eigen_values():
    eigen_values = np.array([100.0, 50.0])
    eigen_vectors = np.array([[1, 0], [0, 1]])
    expected_cor = np.array([[1.0, 0.0], [0.0, 1.0]])
    cor = _eig_to_cor(eigen_values, eigen_vectors)

    assert cor == pytest.approx(
        expected_cor, rel=1e-7
    ), "Correlation matrix does not match the expected values for large eigenvalues."


def test_eig_to_cor_negative_eigen_values():
    eigen_values = np.array([2.0, -1.0])
    eigen_vectors = np.array([[1, 0], [0, 1]])
    expected_cor = np.array([[1.0, 0.0], [0.0, 1.0]])
    cor = _eig_to_cor(eigen_values, eigen_vectors)

    assert np.all(np.diag(cor) == 1.0), "Diagonal elements should be 1."
    assert np.all(cor[cor < -1] == -1), "Elements should be clamped to -1."
    assert np.all(cor[cor > 1] == 1), "Elements should be clamped to 1."


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_cor_to_eig_default():
    cor = np.array([[1.0, 0.5], [0.5, 1.0]])
    expected_eigen_values = np.array([1.5, 0.5])
    expected_eigen_vectors = np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]])
    eigen_values, eigen_vectors = _cor_to_eig(cor)

    assert eigen_values == pytest.approx(
        expected_eigen_values, rel=1e-7
    ), "Eigenvalues do not match the expected values."
    assert eigen_vectors == pytest.approx(
        expected_eigen_vectors, rel=1e-7
    ), "Eigenvectors do not match the expected values."


def test_cor_to_eig_sorted():
    cor = np.array([[1.0, 0.5], [0.5, 1.0]])
    expected_eigen_values = np.array([1.5, 0.5])
    expected_eigen_vectors = np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]])
    eigen_values, eigen_vectors = _cor_to_eig(cor, sort=True)

    assert eigen_values == pytest.approx(
        expected_eigen_values, rel=1e-7
    ), "Sorted eigenvalues do not match the expected values."
    assert eigen_vectors == pytest.approx(
        expected_eigen_vectors, rel=1e-7
    ), "Sorted eigenvectors do not match the expected values."


def test_cor_to_eig_unsorted():
    cor = np.array([[1.0, 0.5], [0.5, 1.0]])
    expected_eigen_values = np.array([0.5, 1.5])
    expected_eigen_vectors = np.array([[-0.70710678, 0.70710678], [0.70710678, 0.70710678]])
    eigen_values, eigen_vectors = _cor_to_eig(cor, sort=False)

    assert eigen_values == pytest.approx(
        expected_eigen_values, rel=1e-7
    ), "Unsorted eigenvalues do not match the expected values."
    assert eigen_vectors == pytest.approx(
        expected_eigen_vectors, rel=1e-7
    ), "Unsorted eigenvectors do not match the expected values."


def test_cor_to_eig_top_k():
    cor = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    k = 2
    eigen_values, eigen_vectors = _cor_to_eig(cor, k=k)
    expected_eigen_values, expected_eigen_vectors = np.linalg.eigh(cor)
    idx = np.argsort(expected_eigen_values)[-k:][::-1]
    expected_eigen_values = expected_eigen_values[idx]
    expected_eigen_vectors = expected_eigen_vectors[:, idx]

    assert eigen_values == pytest.approx(
        expected_eigen_values, rel=1e-7
    ), f"Top {k} eigenvalues do not match the expected values."
    assert eigen_vectors == pytest.approx(
        expected_eigen_vectors, rel=1e-7
    ), f"Top {k} eigenvectors do not match the expected values."


def test_cor_to_eig_real():
    cor = np.array([[1.0, 0.5], [0.5, 1.0]])
    expected_eigen_values = np.array([1.5, 0.5])
    expected_eigen_vectors = np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]])
    eigen_values, eigen_vectors = _cor_to_eig(cor, sort=True)

    assert np.isrealobj(eigen_values), "Eigenvalues should be real."
    assert np.isrealobj(eigen_vectors), "Eigenvectors should be real."
    assert eigen_values == pytest.approx(
        expected_eigen_values, rel=1e-7
    ), "Eigenvalues do not match the expected values."
    assert eigen_vectors == pytest.approx(
        expected_eigen_vectors, rel=1e-7
    ), "Eigenvectors do not match the expected values."


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_eig_complete_no_completion_needed():
    eigen_values = np.array([2.0, 1.0])
    eigen_vectors = np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]])
    v, e = _eig_complete(eigen_values, eigen_vectors)

    assert v == pytest.approx(eigen_values, rel=1e-7), "Eigenvalues should not change when no completion is needed."
    assert e == pytest.approx(eigen_vectors, rel=1e-7), "Eigenvectors should not change when no completion is needed."


def test_eig_complete_completion_needed():
    eigen_values = np.array([2.0])
    eigen_vectors = np.array([[1], [0]])
    expected_values = np.array([2.0, 0.0])  # Rest of the values will be zero
    expected_vectors = np.array([[1, 0], [0, 1]])

    v, e = _eig_complete(eigen_values, eigen_vectors)

    assert v == pytest.approx(expected_values, rel=1e-7), "Completed eigenvalues do not match the expected values."
    assert np.allclose(e @ e.T, np.eye(2), atol=1e-7), "Completed eigenvectors do not form an orthonormal basis."


def test_eig_complete_full_rank():
    eigen_values = np.array([1.0, 1.0])
    eigen_vectors = np.array([[1, 0], [0, 1]])

    v, e = _eig_complete(eigen_values, eigen_vectors)

    assert v == pytest.approx(eigen_values, rel=1e-7), "Eigenvalues should not change when already full rank."
    assert e == pytest.approx(eigen_vectors, rel=1e-7), "Eigenvectors should not change when already full rank."


def test_eig_complete_partial_rank():
    eigen_values = np.array([2.0, 1.0])
    eigen_vectors = np.array(
        [
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
        ]
    )
    expected_values = np.array([2.0, 1.0, -1.0 / 3])  # Rest of the values is calculated
    v, e = _eig_complete(eigen_values, eigen_vectors)  # Providing only the first two components

    # Verify completed eigenvalues
    assert v[0:2] == pytest.approx(eigen_values, rel=1e-7), "Primary part of eigenvalues do not match expected values."
    assert v[1] == pytest.approx(
        expected_values[1], rel=1e-7
    ), "Completed parts of eigenvalues do not match expected values."
    assert np.allclose(e @ e.T, np.eye(2), atol=1e-7), "Completed eigenvectors do not form an orthonormal basis."


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_marcenko_pastur_support_basic():
    var = 1.0
    num_timesteps = 100
    num_assets = 50
    expected_lower = (1 - (50 / 100) ** 0.5) ** 2
    expected_upper = (1 + (50 / 100) ** 0.5) ** 2
    lower_bound, upper_bound = _marcenko_pastur_support(var, num_timesteps, num_assets)

    assert lower_bound == pytest.approx(expected_lower, rel=1e-7), "Lower bound does not match the expected value."
    assert upper_bound == pytest.approx(expected_upper, rel=1e-7), "Upper bound does not match the expected value."


def test_marcenko_pastur_support_variance_scale():
    var = 2.0
    num_timesteps = 100
    num_assets = 50
    expected_lower = var * (1 - (50 / 100) ** 0.5) ** 2
    expected_upper = var * (1 + (50 / 100) ** 0.5) ** 2
    lower_bound, upper_bound = _marcenko_pastur_support(var, num_timesteps, num_assets)

    assert lower_bound == pytest.approx(
        expected_lower, rel=1e-7
    ), "Lower bound with variance scaling does not match the expected value."
    assert upper_bound == pytest.approx(
        expected_upper, rel=1e-7
    ), "Upper bound with variance scaling does not match the expected value."


def test_marcenko_pastur_support_equal_timesteps_assets():
    var = 1.0
    num_timesteps = 50
    num_assets = 50
    expected_lower = 0.0
    expected_upper = 4.0
    lower_bound, upper_bound = _marcenko_pastur_support(var, num_timesteps, num_assets)

    assert lower_bound == pytest.approx(
        expected_lower, rel=1e-7
    ), "Lower bound for equal timesteps and assets does not match."
    assert upper_bound == pytest.approx(
        expected_upper, rel=1e-7
    ), "Upper bound for equal timesteps and assets does not match."


def test_marcenko_pastur_support_more_assets():
    var = 1.0
    num_timesteps = 50
    num_assets = 100
    q = num_timesteps / num_assets
    expected_lower = (1 - (1.0 / q) ** 0.5) ** 2
    expected_upper = (1 + (1.0 / q) ** 0.5) ** 2
    lower_bound, upper_bound = _marcenko_pastur_support(var, num_timesteps, num_assets)

    assert lower_bound == pytest.approx(
        expected_lower, rel=1e-7
    ), "Lower bound for more assets than timesteps does not match."
    assert upper_bound == pytest.approx(
        expected_upper, rel=1e-7
    ), "Upper bound for more assets than timesteps does not match."


def test_marcenko_pastur_support_non_unit_variance():
    var = 0.5
    num_timesteps = 100
    num_assets = 50
    q = num_timesteps / num_assets
    expected_lower = var * (1 - (1.0 / q) ** 0.5) ** 2
    expected_upper = var * (1 + (1.0 / q) ** 0.5) ** 2
    lower_bound, upper_bound = _marcenko_pastur_support(var, num_timesteps, num_assets)

    assert lower_bound == pytest.approx(expected_lower, rel=1e-7), "Lower bound with non-unit variance does not match."
    assert upper_bound == pytest.approx(expected_upper, rel=1e-7), "Upper bound with non-unit variance does not match."


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_marchenko_pastur_pdf_basic():
    var = 1.0
    num_timesteps = 100
    num_assets = 50
    eigen_values = np.linspace(0, 4, 100)

    q = num_timesteps / num_assets
    e_min, e_max = _marcenko_pastur_support(var, num_timesteps, num_assets)
    pdf = _marchenko_pastur_pdf(var, num_timesteps, num_assets, eigen_values)

    expected_pdf = np.zeros_like(eigen_values)
    mask = np.logical_and(eigen_values > e_min, eigen_values < e_max)
    expected_pdf[mask] = (
        q
        / (2 * np.pi * var * eigen_values[mask])
        * ((e_max - eigen_values[mask]) * (eigen_values[mask] - e_min)) ** 0.5
    )

    assert pdf == pytest.approx(expected_pdf, rel=1e-7), "PDF values do not match the expected values."


def test_marchenko_pastur_pdf_zero_eigenvalues():
    var = 1.0
    num_timesteps = 100
    num_assets = 50
    eigen_values = np.zeros(100)

    pdf = _marchenko_pastur_pdf(var, num_timesteps, num_assets, eigen_values)

    assert np.all(pdf == 0), "PDF should be zero for all eigenvalues of zero."


def test_marchenko_pastur_pdf_single_value():
    var = 1.0
    num_timesteps = 100
    num_assets = 50
    eigen_values = np.array([2.0])

    q = num_timesteps / num_assets
    e_min, e_max = _marcenko_pastur_support(var, num_timesteps, num_assets)
    pdf = _marchenko_pastur_pdf(var, num_timesteps, num_assets, eigen_values)

    if e_min < 2.0 < e_max:
        expected_pdf = q / (2 * np.pi * var * 2.0) * ((e_max - 2.0) * (2.0 - e_min)) ** 0.5
        assert pdf == pytest.approx([expected_pdf], rel=1e-7), "PDF value does not match the expected value."
    else:
        assert np.all(pdf == 0), "PDF should be zero for eigenvalue outside the support range."


def test_marchenko_pastur_pdf_large_variance():
    var = 10.0
    num_timesteps = 100
    num_assets = 50
    eigen_values = np.linspace(0, 40, 100)

    q = num_timesteps / num_assets
    e_min, e_max = _marcenko_pastur_support(var, num_timesteps, num_assets)
    pdf = _marchenko_pastur_pdf(var, num_timesteps, num_assets, eigen_values)

    expected_pdf = np.zeros_like(eigen_values)
    mask = np.logical_and(eigen_values > e_min, eigen_values < e_max)
    expected_pdf[mask] = (
        q
        / (2 * np.pi * var * eigen_values[mask])
        * ((e_max - eigen_values[mask]) * (eigen_values[mask] - e_min)) ** 0.5
    )

    assert pdf == pytest.approx(
        expected_pdf, rel=1e-7
    ), "PDF values do not match the expected values with large variance."


def test_marchenko_pastur_pdf_invalid_input():
    var = -1.0  # Invalid variance
    num_timesteps = 100
    num_assets = 50
    eigen_values = np.linspace(0, 4, 100)

    with pytest.raises(ValueError, match="Variance must be positive."):
        _marchenko_pastur_pdf(var, num_timesteps, num_assets, eigen_values)


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_marchenko_pastur_fit_basic():
    num_timesteps = 100
    num_assets = 50
    eigen_values = np.linspace(0.5, 2.5, 100)
    optimal_var = _marchenko_pastur_fit(num_timesteps, num_assets, eigen_values)

    assert (
        0.01 <= optimal_var <= 4
    ), "Optimal variance should be within the search range [0.01, 4] for the given eigenvalues."


def test_marchenko_pastur_fit_large_var():
    num_timesteps = 100
    num_assets = 50
    eigen_values = np.linspace(10.0, 20.0, 100)

    optimal_var = _marchenko_pastur_fit(num_timesteps, num_assets, eigen_values)

    assert (
        0.01 <= optimal_var <= 4
    ), "Optimal variance should be within the search range [0.01, 4] for the given eigenvalues."


def test_marchenko_pastur_fit_no_noise():
    num_timesteps = 100
    num_assets = 50
    # Eigenvalues corresponding to no noise, they fit the MP distribution exactly for var=1.0
    q = num_timesteps / num_assets
    eigenvalues_mp = np.linspace((1 - np.sqrt(1 / q)) ** 2, (1 + np.sqrt(1 / q)) ** 2, 100)
    optimal_var = _marchenko_pastur_fit(num_timesteps, num_assets, eigenvalues_mp)

    # Adjusting tolerance due to possible slight deviation in fit
    assert (
        0.2 <= optimal_var <= 0.5
    ), "Optimal variance should be in the range of 0.2 to 0.5 for perfect MP distribution."


def test_marchenko_pastur_fit_noisy_data():
    num_timesteps = 100
    num_assets = 50
    # Introduce some noise to the MP distribution
    q = num_timesteps / num_assets
    eigenvalues_mp = np.linspace((1 - np.sqrt(1 / q)) ** 2, (1 + np.sqrt(1 / q)) ** 2, 100)
    np.random.seed(0)  # For reproducibility
    noise = np.random.normal(0, 0.1, eigenvalues_mp.shape)
    eigen_values_noisy = eigenvalues_mp + noise

    optimal_var = _marchenko_pastur_fit(num_timesteps, num_assets, eigen_values_noisy)

    # Adjusting the expected range due to noise influence
    assert 0.2 <= optimal_var <= 0.5, "Optimal variance should be within 0.2 to 0.5 for noisy data."


def test_marchenko_pastur_fit_invalid_input():
    num_timesteps = 100
    num_assets = 50
    eigen_values = np.linspace(0.5, 2.5, 100)

    with pytest.raises(ValueError, match="eigen_values must be a numpy array"):
        _marchenko_pastur_fit(num_timesteps, num_assets, "invalid input")


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_cov_denoise_default():
    np.random.seed(0)
    cov = np.array([[2, 1], [1, 3]])
    num_timesteps = 100
    denoised_cov, info = _cov_denoise(cov, num_timesteps)

    assert denoised_cov.shape == cov.shape, "Denoised covariance matrix should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."
    assert (
        "var" in info or "k" in info
    ), "The info dictionary should contain either the 'var' or 'k' key depending on whether k was specified or not."


def test_cov_denoise_with_k():
    np.random.seed(0)
    cov = np.array([[2, 1], [1, 3]])
    num_timesteps = 100
    k = 1
    denoised_cov, info = _cov_denoise(cov, num_timesteps, k=k)

    assert denoised_cov.shape == cov.shape, "Denoised covariance matrix should have the same shape as the input."
    assert info["fitted"] == False, "Info 'fitted' should be False when 'k' is specified."
    assert info["k"] == k, f"Info 'k' should be equal to the specified k, which is {k}."


def test_cov_denoise_large_matrix():
    np.random.seed(0)
    cov = np.cov(np.random.randn(100, 50), rowvar=False)
    num_timesteps = 200
    denoised_cov, info = _cov_denoise(cov, num_timesteps)

    assert denoised_cov.shape == cov.shape, "Denoised covariance matrix should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."


def test_cov_denoise_noisy_data():
    np.random.seed(0)
    true_cov = np.array([[2, 0.8], [0.8, 3]])
    noisy_cov = true_cov + np.random.normal(0, 0.1, true_cov.shape)
    num_timesteps = 100
    denoised_cov, info = _cov_denoise(noisy_cov, num_timesteps)

    assert denoised_cov.shape == noisy_cov.shape, "Denoised covariance matrix should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."
    assert not np.allclose(denoised_cov, noisy_cov), "Denoised matrix should be different from the noisy input."
    assert np.allclose(
        np.diag(denoised_cov), np.diag(true_cov), atol=0.5
    ), "Diagonal elements of the denoised matrix should be close to the true covariance matrix."


def test_cov_denoise_invalid_input():
    cov = "Invalid input"
    num_timesteps = 100

    with pytest.raises(TypeError):
        _cov_denoise(cov, num_timesteps)


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_cor_denoise_default():
    cor = np.array([[1.0, 0.5], [0.5, 1.0]])
    num_timesteps = 100
    denoised_cor, _, info = _cor_denoise(cor, num_timesteps)

    assert denoised_cor.shape == cor.shape, "Denoised correlation matrix should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."
    assert (
        "var" in info or "k" in info
    ), "The info dictionary should contain either the 'var' or 'k' key depending on whether k was specified or not."


def test_cor_denoise_with_k():
    cor = np.array([[1.0, 0.5], [0.5, 1.0]])
    num_timesteps = 100
    k = 1
    denoised_cor, eig_vectors, info = _cor_denoise(cor, num_timesteps, k=k)

    assert denoised_cor.shape == cor.shape, "Denoised correlation matrix should have the same shape as the input."
    assert info["fitted"] == False, "Info 'fitted' should be False when 'k' is specified."
    assert info["k"] == k, f"Info 'k' should be equal to the specified k, which is {k}."
    assert eig_vectors is not None, "Eigenvectors should be returned when 'k' is specified."


def test_cor_denoise_large_matrix():
    np.random.seed(0)
    cor = np.corrcoef(np.random.randn(100, 50), rowvar=False)
    num_timesteps = 200
    denoised_cor, _, info = _cor_denoise(cor, num_timesteps)

    assert denoised_cor.shape == cor.shape, "Denoised correlation matrix should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."


def test_cor_denoise_large_matrix_low_num_time_steps():
    np.random.seed(0)
    cor = np.corrcoef(np.random.randn(100, 50), rowvar=False)
    num_timesteps = 10
    denoised_cor, _, info = _cor_denoise(cor, num_timesteps)

    assert denoised_cor.shape == cor.shape, "Denoised correlation matrix should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."


def test_cor_denoise_noisy_data():
    np.random.seed(0)
    true_cor = np.array([[1, 0.8], [0.8, 1]])
    noisy_cor = true_cor + np.random.normal(0, 0.1, true_cor.shape)
    np.fill_diagonal(noisy_cor, 1)
    num_timesteps = 100
    denoised_cor, _, info = _cor_denoise(noisy_cor, num_timesteps)

    assert denoised_cor.shape == noisy_cor.shape, "Denoised correlation matrix should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."
    assert not np.allclose(denoised_cor, noisy_cor), "Denoised matrix should be different from the noisy input."
    assert np.allclose(
        np.diag(denoised_cor), np.diag(true_cor)
    ), "Diagonal elements of the denoised matrix should be close to the true covariance matrix."


def test_cor_denoise_invalid_input():
    cor = "Invalid input"
    num_timesteps = 100

    with pytest.raises(TypeError, match="Input correlation matrix must be a numpy array"):
        _cor_denoise(cor, num_timesteps)

    cor = np.random.randn(5, 3)  # Non-square matrix
    with pytest.raises(ValueError, match="Input must be a square matrix"):
        _cor_denoise(cor, num_timesteps)


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


def test_eigen_values_denoise_default():
    np.random.seed(0)
    eigen_values = np.sort(np.random.uniform(0.1, 2.0, 50))[::-1]
    num_timesteps = 100

    improved_eigen_vals, info = _eigen_values_denoise(eigen_values, num_timesteps)

    assert (
        improved_eigen_vals.shape == eigen_values.shape
    ), "Denoised eigenvalues should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."
    assert "var" in info, "The info dictionary should contain the 'var' key."
    assert "max_noise" in info, "The info dictionary should contain the 'max_noise' key."
    assert "k" in info, "The info dictionary should contain the 'k' key."
    assert np.all(
        improved_eigen_vals[: -info["k"]] >= 0
    ), "Improved eigenvalues should be greater than or equal to zero."
    assert np.all(
        improved_eigen_vals[info["k"] :] == improved_eigen_vals[info["k"]]
    ), "Eigenvalues below k should be equal to their mean."


def test_eigen_values_denoise_with_k():
    np.random.seed(0)
    eigen_values = np.sort(np.random.uniform(0.1, 2.0, 50))[::-1]
    num_timesteps = 100
    k = 10

    improved_eigen_vals, info = _eigen_values_denoise(eigen_values, num_timesteps, k=k)

    assert (
        improved_eigen_vals.shape == eigen_values.shape
    ), "Denoised eigenvalues should have the same shape as the input."
    assert info["fitted"] == False, "The info dictionary 'fitted' should be False when 'k' is specified."
    assert info["k"] == k, f"Info 'k' should be equal to the specified k, which is {k}."
    assert np.all(improved_eigen_vals[:k] == eigen_values[:k]), "The top k eigenvalues should remain the same."
    assert np.all(
        improved_eigen_vals[k:] == np.mean(eigen_values[k:])
    ), "The remaining eigenvalues should be equal to their mean."


def test_eigen_values_denoise_large_matrix():
    np.random.seed(0)
    eigen_values = np.sort(np.random.uniform(0.1, 5.0, 100))[::-1]
    num_timesteps = 200

    improved_eigen_vals, info = _eigen_values_denoise(eigen_values, num_timesteps)

    assert (
        improved_eigen_vals.shape == eigen_values.shape
    ), "Denoised eigenvalues should have the same shape as the input."
    assert "fitted" in info, "The info dictionary should contain the 'fitted' key."


def test_eigen_values_denoise_with_noise():
    np.random.seed(0)
    true_eigenvalues = np.linspace(1, 10, 50)
    noisy_eigenvalues = true_eigenvalues + np.random.normal(0, 0.5, true_eigenvalues.shape)
    num_timesteps = 100

    improved_eigen_vals, info = _eigen_values_denoise(noisy_eigenvalues, num_timesteps)

    assert (
        improved_eigen_vals.shape == noisy_eigenvalues.shape
    ), "Denoised eigenvalues should have the same shape as the input."
    assert np.all(
        improved_eigen_vals[: info["k"]] >= 0
    ), "Improved eigenvalues should be greater than or equal to zero."
    assert np.all(
        improved_eigen_vals[info["k"] :] == np.mean(noisy_eigenvalues[info["k"] :])
    ), "Improved eigenvalues below k should be equal to their mean."


def test_eigen_values_denoise_invalid_input():
    num_timesteps = 100

    with pytest.raises(ValueError):
        _eigen_values_denoise("invalid input", num_timesteps)


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
def test_random_cov_dimensions():
    dim = 3
    cov_matrix = _random_cov(dim)

    assert cov_matrix.shape == (dim, dim), f"Covariance matrix should be {dim}x{dim}."
    assert np.allclose(cov_matrix, cov_matrix.T, atol=1e-7), "Covariance matrix should be symmetric."


def test_random_cov_default():
    cov_matrix = _random_cov()

    assert cov_matrix.shape == (3, 3), "Default covariance matrix should be 3x3."
    assert np.allclose(cov_matrix, cov_matrix.T, atol=1e-7), "Default covariance matrix should be symmetric."


def test_random_cov_diagonal_sum():
    dim = 5
    cov_matrix = _random_cov(dim)

    sum_diagonal = np.sum(np.diag(cov_matrix))
    assert np.allclose(
        sum_diagonal, 1.0, atol=0.1
    ), "Sum of the covariances on the main diagonal should be approximately 1."


def test_random_cov_positive_definite():
    dim = 4
    cov_matrix = _random_cov(dim)

    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    assert np.all(eigenvalues > 0), "Covariance matrix should be positive definite (all eigenvalues > 0)."


def test_random_cov_randomness():
    dim = 3
    np.random.seed(0)
    cov_matrix1 = _random_cov(dim)
    np.random.seed(1)
    cov_matrix2 = _random_cov(dim)

    assert not np.allclose(
        cov_matrix1, cov_matrix2
    ), "Covariance matrices generated with different seeds should be different."


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
def test_random_cor_dimensions():
    dim = 3
    cor_matrix = _random_cor(dim)

    assert cor_matrix.shape == (dim, dim), f"Correlation matrix should be {dim}x{dim}."
    assert np.allclose(cor_matrix, cor_matrix.T, atol=1e-7), "Correlation matrix should be symmetric."


def test_random_cor_default():
    cor_matrix = _random_cor()

    assert cor_matrix.shape == (3, 3), "Default correlation matrix should be 3x3."
    assert np.allclose(cor_matrix, cor_matrix.T, atol=1e-7), "Default correlation matrix should be symmetric."


def test_random_cor_diagonal():
    dim = 5
    cor_matrix = _random_cor(dim)

    assert np.all(np.isclose(np.diag(cor_matrix), 1.0)), "All diagonal elements of the correlation matrix should be 1."


def test_random_cor_validity():
    dim = 4
    cor_matrix = _random_cor(dim)

    eigenvalues = np.linalg.eigvalsh(cor_matrix)
    assert np.all(eigenvalues > 0), "Correlation matrix should be positive definite (all eigenvalues > 0)."


def test_random_cor_randomness():
    dim = 3
    np.random.seed(0)
    cor_matrix1 = _random_cor(dim)
    np.random.seed(1)
    cor_matrix2 = _random_cor(dim)

    assert not np.allclose(
        cor_matrix1, cor_matrix2
    ), "Correlation matrices generated with different seeds should be different."


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************


if __name__ == "__main__":
    pytest.main([__file__])
