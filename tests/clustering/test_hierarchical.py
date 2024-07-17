from itertools import product
from typing import Union


from quantfinlib.clustering.hierarchical import HC
from quantfinlib.distance.distance_matrix import (
    corr_to_dist,
    get_corr_distance_matrix,
    get_info_distance_matrix,
    _calculate_correlation,
)

import numpy as np
import pandas as pd
import unittest
import pytest

from quantfinlib.datasets import load_equity_indices


class TestHC(unittest.TestCase):

    def setUp(self):
        self.X = load_equity_indices().astype(float)
        self.X = self.X.iloc[:, :25].drop(columns=["VIX"]).pct_change(periods=21 * 12, fill_method="pad").dropna()
        print(self.X.shape)
        self.corr_pearson = _calculate_correlation(self.X, corr_method="pearson")
        self.corr_spearman = _calculate_correlation(self.X, corr_method="spearman")
        self.dist_pearson = get_corr_distance_matrix(self.X, corr_method="pearson")
        self.dist_spearman = get_corr_distance_matrix(self.X, corr_method="spearman")
        self.dist_info = get_info_distance_matrix(self.X.fillna(0), method="var_info")

    def test_x_pearson(self):
        hc = HC(X=self.X, codependence_method="pearson-correlation")
        self.assertEqual(hc.codependence_method, "pearson-correlation")
        self.assertEqual(hc.corr_to_dist_method, "angular")
        self.assertEqual(hc.linkage_method, "ward")
        np.testing.assert_array_equal(hc.dist, self.dist_pearson)
        linkage = hc.linkage
        self.assertIsInstance(linkage, np.ndarray)
        self.assertEqual(linkage.shape[0], self.X.shape[1] - 1)
        self.assertEqual(linkage.shape[1], 4)

    def test_x_spearman(self):
        hc = HC(X=self.X, codependence_method="spearman-correlation")
        self.assertEqual(hc.codependence_method, "spearman-correlation")
        np.testing.assert_array_equal(hc.dist, self.dist_spearman)
        linkage = hc.linkage
        self.assertIsInstance(linkage, np.ndarray)
        self.assertEqual(linkage.shape[0], self.X.shape[1] - 1)
        self.assertEqual(linkage.shape[1], 4)

    def test_corr_pearson(self):
        hc = HC(corr=self.corr_pearson, codependence_method="pearson-correlation")
        self.assertEqual(hc.codependence_method, "pearson-correlation")
        np.testing.assert_array_equal(hc.dist, self.dist_pearson)
        linkage = hc.linkage
        self.assertIsInstance(linkage, np.ndarray)
        self.assertEqual(linkage.shape[0], self.X.shape[1] - 1)
        self.assertEqual(linkage.shape[1], 4)

    def test_x_var_info(self):
        hc = HC(X=self.X, codependence_method="var_info")
        self.assertEqual(hc.codependence_method, "var_info")
        np.testing.assert_array_equal(hc.dist, self.dist_info)
        linkage = hc.linkage
        self.assertIsInstance(linkage, np.ndarray)
        self.assertEqual(linkage.shape[0], self.X.shape[1] - 1)
        self.assertEqual(linkage.shape[1], 4)


def cor_block_diagonal(
    block_sizes: Union[list, np.ndarray] = [60, 60, 60, 60],
    block_cors: Union[list, np.ndarray] = [0.99, 0.99, 0.99, 0.99],
) -> np.ndarray:
    N = np.sum(block_sizes)
    cor = np.zeros((N, N))
    i = 0
    for block_size, block_cor in zip(block_sizes, block_cors):
        cor[i : i + block_size, i : i + block_size] = block_cor
        i += block_size
    np.fill_diagonal(cor, 1.0)
    return cor


@pytest.mark.parametrize(
    "corr_to_dist_method, metric", list(product(["angular", "squared_angular", "abs_angular"], ["gap", "silhouette"]))
)
def test_block_diagonal(corr_to_dist_method, metric):
    corr = cor_block_diagonal()
    hc = HC(corr=corr, codependence_method="pearson-correlation", corr_to_dist_method=corr_to_dist_method)
    dist_directly_calculated = corr_to_dist(corr, corr_to_dist_method=corr_to_dist_method)
    np.testing.assert_array_almost_equal(hc.dist, dist_directly_calculated, decimal=1)
    optimal_nclust = hc.get_optimal_nclusters(max_clust=10, metric=metric, nb=1000)
    assert isinstance(optimal_nclust, int)
    assert optimal_nclust > 0 and optimal_nclust <= 10
    assert optimal_nclust == 4
