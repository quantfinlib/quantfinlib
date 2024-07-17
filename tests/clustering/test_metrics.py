from quantfinlib.clustering.metrics import gap_statistic, silhouette_tstat

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_dist():
    return np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])


@pytest.fixture
def simple_labels():
    return np.array([0, 0, 1])


def test_gap_statistic(simple_dist, simple_labels):
    gap, gap_std = gap_statistic(simple_dist, simple_labels, nb=10)
    assert isinstance(gap, float)
    assert isinstance(gap_std, float)


def test_silhouette_tstat(simple_dist, simple_labels):
    dist, labels = simple_dist, simple_labels
    tstat = silhouette_tstat(dist, labels)
    assert isinstance(tstat, float)


@pytest.mark.parametrize("nb", range(2, 10))
def test_gap_statistic_nb(simple_dist, simple_labels, nb):
    gap, gap_std = gap_statistic(simple_dist, simple_labels, nb=nb)
    assert isinstance(gap, float)
    assert isinstance(gap_std, float)


def _calculate_gap_without_vectorization(dist, labels, nb, random_state=None):
    n = len(labels)
    k = len(np.unique(labels))

    wk = 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                wk += dist[i, j] ** 2
    wk /= 2 * k
    E_log_Wk = 0
    if random_state is not None:
        seed = random_state
    else:
        seed = 0
    for b in range(nb):
        wk_boot = 0
        np.random.seed(seed + b)
        dist_ref = np.random.uniform(0, np.max(dist), size=(n, n))
        # Make the reference distance matrix symmetric
        dist_ref = np.triu(dist_ref) + np.triu(dist_ref).T
        np.fill_diagonal(dist_ref, 0)
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    wk_boot += dist_ref[i, j] ** 2
        wk_boot /= 2 * k
        log_wk_boot = np.log(wk_boot) if wk_boot > 0 else 0
        E_log_Wk += log_wk_boot
    E_log_Wk /= nb
    gap = E_log_Wk - np.log(wk)
    return gap


def test_calculate_gap_without_vectorization(simple_dist, simple_labels):
    expected_gap, _ = gap_statistic(simple_dist, simple_labels, nb=1000)
    bruteforce_gap = _calculate_gap_without_vectorization(simple_dist, simple_labels, nb=1000)
    assert np.isclose(expected_gap, bruteforce_gap, atol=1e-5)


def test_pandas_gap_stats(simple_dist, simple_labels):
    gap_np, gap_std_np = gap_statistic(simple_dist, simple_labels, nb=10)
    dist = pd.DataFrame(simple_dist)
    labels = pd.Series(simple_labels)
    gap_pd, gap_std_pd = gap_statistic(dist, labels, nb=10)
    assert isinstance(gap_pd, float)
    assert isinstance(gap_std_pd, float)
    assert np.isclose(gap_pd, gap_np, atol=1e-5)
    assert np.isclose(gap_std_pd, gap_std_np, atol=1e-5)


def test_pandas_silh_tstats(simple_dist, simple_labels):
    tstat_np = silhouette_tstat(simple_dist, simple_labels)
    dist = pd.DataFrame(simple_dist)
    labels = pd.Series(simple_labels)
    tstat_pd = silhouette_tstat(dist, labels)
    assert isinstance(tstat_pd, float)
    assert np.isclose(tstat_pd, tstat_np, atol=1e-5)
