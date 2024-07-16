from typing import Dict, Tuple, Union
import numpy as np


def _cor_block_diagonal(
    block_sizes: Union[list, np.ndarray] = [10, 10, 20, 24, 64],
    block_cors: Union[list, np.ndarray] = [0.5, 0.2, 0.3, 0.5, 0.3],
) -> np.ndarray:
    """
    Generate a correlation matrix with blocks of correlated assets.

    Example
    -------
    Generating a block covariance matrix:

    .. plot::

        import matplotlib.pyplot as plt
        from samcoml.stats.cov import cor_block_diagonal

        cor = cor_block_diagonal(block_sizes=[3, 7, 12], block_cors=[0.7,  0.3, -0.2])
        plt.matshow(cor)
        plt.show()


    Parameters
    ----------
    block_sizes: Union[list, np.ndarray]
        A list of asset block sizes.
    block_cors: Union[list, np.ndarray]
        A list of correlation values for each block.

    Returns
    -------
        np.dnarray
        np.dnarray

    np.dnarray

    """
    N = np.sum(block_sizes)
    cor = np.zeros((N, N))
    i = 0
    for block_size, block_cor in zip(block_sizes, block_cors):
        cor[i : i + block_size, i : i + block_size] = block_cor
        i += block_size
    np.fill_diagonal(cor, 1.0)
    return cor


def _cov_to_cor(cov: np.ndarray, cov_reg: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a covariance matrix into a correlation matrix + standard deviation vector.

    Parameters
    ----------
    cov: np.ndarray
        Covariance matrix.
    cov_reg: float, optional, default=1E-6
        Optional minimal value for the standard deviation, to prevent the matrix from becoming singular.

    Returns
    -------
    cor, std
    """
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.maximum(np.diag(cov), cov_reg))

    cor = cov / np.outer(std, std)
    cor[cor < -1.0], cor[cor > 1.0] = -1.0, 1.0
    return cor, std


def _cor_to_cov(cor: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix into a covariance matrix.

    Parameters
    ----------
    cor: np.ndarray
        Correlation matrix.
    std: np.ndarray
        Standard deviation vector.

    Returns
    -------
    cov
    """
    cov = cor * np.outer(std, std)
    return cov


def _eig_to_cor(eigen_values: np.ndarray, eigen_vectors: np.ndarray) -> np.ndarray:
    """
    Convert eigenvalues/eigenvectors into a correlation matrix.

    Parameters
    ----------
    eigen_values: np.ndarray
        Eigenvalues vector.
    eigen_vectors: np.ndarray
        An eigenvector matrix, with the eigenvectors in columns.

    Returns
    -------
    cor
    """
    cor = np.dot(eigen_vectors, eigen_values[:, np.newaxis] * eigen_vectors.T)

    # fix numerical issues
    cor[cor < -1] = -1
    cor[cor > 1] = 1
    np.fill_diagonal(cor, 1.0)
    return cor


def _cor_to_eig(cor: np.ndarray, sort: bool = True, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a correlation matrix into eigenvalues and eigenvectors.

    Parameters
    ----------
    cor: np.ndarray
        Correlation matrix.
    sort: bool, default=True
        Sort the eigenvalues/vectors in decreasing order, with the largest eigenvalue first.
    k:  int, optional, default=None
        When specified, return only the k largest eigenvalues/vectors.

    Returns
    -------
    eigen_values, eigenvectors (with the eigenvectors in columns)
    """
    if k:
        N = cor.shape[0]
        eigen_values, eigen_vectors = np.linalg.eigh(cor, subset_by_index=(N - k, N - 1))
    else:
        eigen_values, eigen_vectors = np.linalg.eigh(cor)

    if sort:
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
    real = True
    if real:
        eigen_vectors = eigen_vectors.real
        eigen_values = eigen_values.real
    return eigen_values, eigen_vectors


def _eig_complete(eigen_values: np.ndarray, eigen_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete a subset of eigenvalues/eigenvectors into a full set.

    Parameters
    ----------
    eigen_values: np.ndarray
        A vector of eigenvalues.
    eigen_vectors: np.ndarray
        A matrix [dim, number of vectors] of eigenvectors, with the eigenvectors in columns.

    Returns
    -------
    eigen_values, eigen_vectors
    """
    N = eigen_vectors.shape[0]
    k = eigen_vectors.shape[1]
    assert k > 0

    if k < N:
        v = np.zeros(N)
        e = np.zeros((N, N))

        v[:k] = eigen_values
        v[k:] = (N - np.sum(eigen_values)) / (N - k)
        e, _, _ = np.linalg.svd(eigen_vectors)
        e[:, :k] = eigen_vectors
        return v, e
    return eigen_values, eigen_vectors


def _marcenko_pastur_support(var: float, num_timesteps: int, num_assets: int) -> Tuple[float, float]:
    """
    Compute the range (support) where the Marcenko Pastur distribution is non-zero.

    The Marchenko–Pastur distribution is the distribution of the eigenvalues of the correlation matrix
    of random uncorrelated time series.

    Parameters
    ----------
    var: float
        Variance distribution parameter.
    num_timesteps:
        Number of time-steps used in the correlation matrix estimate.
    num_assets:
        Number of assets used in the correlation matrix estimate.

    Returns
    -------
    lower_bound, upper_bound
    """
    q = num_timesteps / num_assets
    return var * (1 - (1.0 / q) ** 0.5) ** 2, var * (1 + (1.0 / q) ** 0.5) ** 2


def _marchenko_pastur_pdf(var: float, num_timesteps: int, num_assets: int, eigen_values) -> np.ndarray:
    """
    The Marchenko–Pastur probability density function.

    The Marchenko–Pastur distribution is the distribution of the eigenvalues of the correlation matrix
    of random uncorrelated time series.

    Parameters
    ----------
    var: float
        Variance distribution parameter.
    num_timesteps:
        Number of time-steps used in the correlation matrix estimate.
    num_assets:
        Number of assets used in the correlation matrix estimate.
    eigen_values:
        A list of eigen_values (x-values) for which we want to evaluate the probability density function.

    Returns
    -------
        A list of probability density values.
        A list of probability density values.

    A list of probability density values.

    """
    q = num_timesteps / num_assets
    e_min, e_max = _marcenko_pastur_support(var, num_timesteps, num_assets)
    pdf = np.zeros_like(eigen_values)
    mask = np.logical_and(eigen_values > e_min, eigen_values < e_max)
    pdf[mask] = (
        q
        / (2 * np.pi * var * eigen_values[mask])
        * ((e_max - eigen_values[mask]) * (eigen_values[mask] - e_min)) ** 0.5
    )
    return pdf


def _marchenko_pastur_fit(num_timesteps: int, num_assets: int, eigen_values: np.ndarray) -> float:
    """
    Fit the Marchenko–Pastur distribution (estimate the variance parameter) to a list of eigenvalues.

    The Marchenko–Pastur distribution is the distribution of the eigenvalues of the correlation matrix
    of random uncorrelated time series.

    Parameters
    ----------
    num_timesteps: int
        Number of time-steps used in the correlation matrix estimate.
    num_assets: int
        Number of assets used in the correlation matrix estimate.
    eigen_values: np.ndarray
        A list of eigenvalues.

    Returns
    -------
    The optimal var parameter.
    """
    max_ll = -1e9
    max_var = 0
    for var in np.linspace(0.01, 4, 1000):
        pdf = _marchenko_pastur_pdf(var, num_timesteps, num_assets, eigen_values)
        noise_mask = pdf > 0.0
        pdf[~noise_mask] = 1.0 / np.max(eigen_values)
        ll = np.sum(np.log(pdf))
        if ll > max_ll:
            max_ll = ll
            max_var = var
    return max_var


def _cov_denoise(cov: np.ndarray, num_timesteps: int, k: int = None) -> Tuple[np.ndarray, dict]:
    """
    Denoise a covariance matrix.

    Parameters
    ----------
    cov: np. ndarray
        A covariance matrix.
    num_timesteps: int
        Number of time-steps used in the covariance matrix estimate.
    k: int, optional, default=None
        When specified, denoise assuming that the top-k largest eigenvectors are the signal.
        When omitted, determine the optimal k by fitting the Marchenko–Pastur distribution.

    Returns
    -------
    cov, info
    """
    cor, std = _cov_to_cor(cov)
    cor_, info = _cor_denoise(cor, num_timesteps, k=k)
    return _cor_to_cov(cor_, std), info


def _cor_denoise(cor: np.ndarray, num_timesteps: int, k: int = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Denoise a correlation matrix.

    Parameters
    ----------
    cor: np. ndarray
        A correlation matrix.
    num_timesteps: int
        Number of time-steps used in the covariance matrix estimate.
    k: int, optional, default=None
        When specified, denoise assuming that the top-k largest eigenvectors are the signal.
        When omitted, determine the optimal k by fitting the Marchenko–Pastur distribution.

    Returns
    -------
    cor, eigen_vectors, info
    """
    if k is None:
        # Estimate number of factors
        eigen_values, eigen_vectors = _cor_to_eig(cor)
        improved_eigen_values, info = _eigen_values_denoise(eigen_values, num_timesteps, k=None)
        return _eig_to_cor(improved_eigen_values, eigen_vectors), info
    else:
        # Manual provide number of factors
        eigen_values, eigen_vectors = _cor_to_eig(cor, k=k)
        v, e = _eig_complete(eigen_values, eigen_vectors)
        info = {"fitted": False, "k": k}
        return v, e, info


def _eigen_values_denoise(eigen_values: np.ndarray, num_timesteps: int, k: int = None) -> Tuple[np.ndarray, dict]:
    """
    Denoise a list of eigenvalues.

    Parameters
    ----------
    eigen_values: np.ndarray
        A list of eigenvalues.
    num_timesteps: int
        Number of time-steps used in the covariance matrix estimate.
    k: int, optional, default=None
        When specified, denoise assuming that the top-k largest eigenvectors are the signal.
        When omitted, determine the optimal k by fitting the Marchenko–Pastur distribution.

    Returns
    -------
    improved_eigen_values, info
    """
    info = {}
    info["fitted"] = False

    if k is None:
        # Fit the Marchenko Pastur distribution to the eigenvalues and find the noise-var shape parameter,
        num_assets = len(eigen_values)
        best_var = _marchenko_pastur_fit(num_timesteps, num_assets, eigen_values)
        _, max_noise_eig_val = _marcenko_pastur_support(best_var, num_timesteps, num_assets)
        k = np.sum(1 * (eigen_values >= max_noise_eig_val))

        info["fitted"] = True
        info["var"] = best_var
        info["max_noise"] = max_noise_eig_val

    info["k"] = k
    # All eigenvalues below "max_noise_eig_val" are considered noise components,
    # we will replace those with their mean.
    improved_eigen_values = eigen_values.copy()
    improved_eigen_values[k:] = np.mean(eigen_values[k:])
    return improved_eigen_values, info


def _random_cov(dim: int = 3) -> np.ndarray:
    """
    Create a random covariance matrix.

    The sum of the covariances on the main diagonal is approximately 1.

    Parameters
    ----------
    dim: int
        Number of dimensions

    Returns
    -------
    A random covariance matrix.
    """
    rows = 2 * dim
    x = np.random.normal(scale=2**0.5 / rows, size=(rows, dim))
    return np.matmul(x.T, x)


def _random_cor(dim: int = 3) -> np.ndarray:
    """
    Create a random correlation matrix.

    Parameters
    ----------
    dim: int
        Number of dimensions

    Returns
    -------
    A random correlation matrix
    """
    cov = _random_cov(dim=dim)
    cor, std = _cov_to_cor(cov)
    return cor


def _decorrelate(x: np.ndarray, ddof: int = 1, adjust: str = None, k: int = None) -> Tuple[np.ndarray, dict]:
    """
    De-correlate & whiten a set of time series.

    Transforms a matrix of time series (in columns) into time series such that they have zero mean,
    unit variance, and are mutually uncorrelated.
    The resulting time series factors are sorted from left to right in decreasing eigenvalues.
    The time series in the first column is the strongest factor, and the time series in the last column is the smallest factor.

    Parameters
    ----------
    x: np.array
        A matrix with time series stored in columns.
    ddof: int, optional, default=1
        Delta degrees of freedom used in the variance estimate.
    adjust: str, optional, default=None
        Indicates how to adjust the decorrelated factors. Options are 'denoise' or 'pca'.
    k: int, optional, default=None
        When specified, denoise or PCA assuming that the top-k largest eigenvectors are the signal.
        When omitted, determine the optimal k by fitting the Marchenko–Pastur distribution.

    Returns
    -------
    Decorrelated series, a dict with statistical information.

    """
    assert adjust in [None, "denoise", "pca"], 'Invalid adjustment string. Valid values are "denoise" or "pca"'

    stats = {}

    mean = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False, ddof=ddof)
    cor, std = _cov_to_cor(cov)
    x_ = (x - mean.reshape(1, -1)) / std.reshape(1, -1)

    eigen_val, eigen_vec = _cor_to_eig(cor)

    # decorelate
    ans = np.matmul(x_, eigen_vec)

    # adjust eigenvalues if needed
    if adjust is not None:

        # for pca, we set the k smalest eigenvalues to zero.
        if adjust == "pca":
            if not k:
                # use eigen_values_denoise to estimate "k", the number of non-noisy eigenvalues
                eigen_val, denoise_stats = _eigen_values_denoise(eigen_val, x.shape[0], k)
                stats.update(denoise_stats)
                k = denoise_stats["k"]

            # for pca we set the last eigenvalues to zero
            if k < len(eigen_val):
                eigen_val[k:] = 0

            # rescale each column to set the std to 1.
            # Divide the first k columns with the eigenvalues, the eigenvalues are zero
            ans[:, :k] = ans[:, :k] / eigen_val.reshape(1, -1)[:, :k] ** 0.5

        elif adjust == "denoise":
            # use eigen_values_denoise to (potentialy) estimate k and adjust the smallest
            # eigenvalues to their mean
            eigen_val, denoise_stats = _eigen_values_denoise(eigen_val, x.shape[0], k)
            stats.update(denoise_stats)

            ans = ans / eigen_val.reshape(1, -1) ** 0.5

    stats.update({"mean": mean, "std": std, "eigen_val": eigen_val, "eigen_vec": eigen_vec})
    return ans, stats


def _recorrelate(
    e: np.ndarray, mean: np.ndarray, std: np.ndarray, eigen_val: np.ndarray, eigen_vec: np.ndarray, **kwargs
):
    """
    Re-correlate previously de-correlated time series.

    This function is the inverse of the decorrelate function.

    Parameters
    ----------
    e: np.ndarray
        A matrix with de-correlated time series stored in columns.
    mean: np.ndarray
        Means for each time series.
    std: np.ndarray
        Standard deviation for each time series.
    eigen_val: np.ndarray
        Eigenvalues of the target correlation matrix.
    eigen_vec: np.ndarray
        Eigenvectors of the target correlation matrix.

    Returns
    -------
    The re-correlated time series.

    """
    e = e * eigen_val.reshape(1, -1) ** 0.5
    x = np.matmul(e, eigen_vec.T)
    x = x * std.reshape(1, -1) + mean.reshape(1, -1)
    return x


if __name__ == "__main__":
    ...
