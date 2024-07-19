"""Functions for evaluating clustering results."""

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import silhouette_samples

from quantfinlib.util import (
    DataFrameOrArray,
    SeriesOrArray,
    validate_frame_or_2Darray,
    validate_series_or_1Darray,
    to_numpy,
)


def _check_labels_distance_consistency(dist: DataFrameOrArray, labels: SeriesOrArray) -> None:
    """Check if the distance matrix and labels are consistent."""
    assert len(dist) == len(labels), "The number of labels must match the distance matrix size."


@validate_series_or_1Darray("labels")
@validate_frame_or_2Darray("dist")
def gap_statistic(
    dist: DataFrameOrArray, labels: SeriesOrArray, nb: int = 10, random_state: Optional[int] = None
) -> Tuple[float, float]:
    r"""
    Calculate the gap statistic for a given distance matrix and cluster labels.

    .. math::
        GAP(k) = E(\log\left(W_{k}\right)) - \log\left(W_{k}\right)


    where :math:`W_{k}` is the sum of within-cluster dispersion for :math:`k` clusters, and
    :math:`E(\log(W_{k}))` is the expected value of :math:`\log(W_{k})` for randomized distances.

    Parameters
    ----------
    dist : np.ndarray
        Distance matrix.
    labels : np.ndarray
        Cluster labels.
    nb : int, optional
        Number of bootstrap samples. Default is 10.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    gap: float
        Gap statistic.
    gap_std: float
        Standard deviation of the gap statistic.
    """
    _check_labels_distance_consistency(dist, labels)
    dist, labels = to_numpy(dist), to_numpy(labels)
    n = len(labels)
    k = len(np.unique(labels))  # number of clusters
    # Create a boolean matrix indicating whether two points are in the same cluster
    within_clstr_identifier = labels[:, np.newaxis] == labels[np.newaxis, :]
    # Calculate within-cluster dispersion (Wk)
    # Additional division by 2 to correct for double summing of pair distances
    Wk = np.sum(within_clstr_identifier * dist**2) / (4 * k)
    # Calculate the expected value of log(Wk) under a null reference distribution by randomizing the distance matrix
    E_log_Wk = 0
    log_wk_boots = []  # Store gap statistics for each bootstrap sample
    if random_state is not None:
        seed = random_state
    else:
        seed = 0
    for b in range(nb):
        # Create a reference distance matrix by drawing random samples from the distance matrix
        np.random.seed(seed + b)
        dist_ref = np.random.uniform(0, np.max(dist), size=(n, n))
        # Make the reference distance matrix symmetric
        dist_ref = np.triu(dist_ref) + np.triu(dist_ref).T
        # Make sure the diagonal elements of reference distance matrix are zero
        np.fill_diagonal(dist_ref, 0)
        # Calculate the within-cluster dispersion for the reference distance matrix
        Wk_boot = np.sum(within_clstr_identifier * dist_ref**2) / (4 * k)
        log_Wk_boot = np.log(Wk_boot) if Wk_boot > 0 else 0
        log_wk_boots.append(log_Wk_boot)
        E_log_Wk += log_Wk_boot
    E_log_Wk /= nb
    gap_std = np.std(log_wk_boots)
    # Calculate the gap statistic
    gap = E_log_Wk - np.log(Wk)
    return gap, gap_std


@validate_series_or_1Darray("labels")
@validate_frame_or_2Darray("dist")
def silhouette_tstat(dist: DataFrameOrArray, labels: SeriesOrArray) -> float:
    """
    Calculate the silhouette score for a given distance matrix and cluster labels.

    Parameters
    ----------
    dist : np.ndarray
        Distance matrix.
    labels : np.ndarray
        Cluster labels.

    Returns
    -------
    silh_tstat : float
        T-statistic of the silhouette score.
    """
    _check_labels_distance_consistency(dist, labels)
    dist, labels = to_numpy(dist), to_numpy(labels)
    silh_samples = silhouette_samples(X=dist, labels=labels, metric="precomputed")
    silh_tstat = np.mean(silh_samples) / np.std(silh_samples)
    return silh_tstat
