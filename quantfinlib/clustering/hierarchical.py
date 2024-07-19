"""Hierarchical clustering module."""

from typing import Optional

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


from quantfinlib.distance.distance_matrix import get_corr_distance_matrix, get_info_distance_matrix, corr_to_dist
from quantfinlib.clustering.metrics import gap_statistic, silhouette_tstat
from quantfinlib.util import DataFrameOrArray


class HC:
    """Hierarchical clustering class.

    Attributes
    ----------
    X : np.ndarray or pd.DataFrame
        Input data.
    corr : np.ndarray or pd.DataFrame
        Correlation matrix.
    dist : np.ndarray or pd.DataFrame
        Distance matrix.
    codependence_method : str
        The codependence method, default is 'pearson'.
        Options are 'pearson-correlation', 'spearman-correlation', 'var_info'.
    corr_to_dist_method : str
        The correlation to distance conversion method, default is 'angular'.
        Options are 'angular', 'abs_angular', 'squared_angular'.
    linkage_method : str
        The linkage method, default is 'ward'.
        Options are 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'.
    _linkage : np.ndarray
        The linkage matrix.

    Methods
    -------
    _compute_distance_matrix()
        Compute the distance matrix.
    linkage
        Return the linkage matrix.
    get_clusters(n_clusters)
        Get the clusters based on the threshold.
    get_optimal_nclusters(max_clust, method, B)
        Get the optimal number of clusters.
    get_dendrogram(**kwargs)
        Get the dendrogram.

    Properties
    ----------
    linkage : np.ndarray
        The linkage matrix.

    Raises
    ------
    ValueError
        If X is None and corr is None in case
        codependence method is either `pearson` or `spearman`.

    """

    def __init__(
        self,
        X: Optional[DataFrameOrArray] = None,
        corr: Optional[DataFrameOrArray] = None,
        dist: Optional[DataFrameOrArray] = None,
        codependence_method: str = "pearson-correlation",
        corr_to_dist_method: str = "angular",
        linkage_method: str = "ward",
        optimal_ordering: bool = True,
    ) -> None:

        self.X = X
        self.corr = corr
        self.codependence_method = codependence_method
        self.corr_to_dist_method = corr_to_dist_method
        self.linkage_method = linkage_method
        self.optimal_ordering = optimal_ordering
        if dist is None:
            self._compute_distance_matrix()
        else:
            self.dist = dist
        self._linkage = None

    def _compute_distance_matrix(self):
        """Compute the distance matrix."""
        if self.codependence_method in ["pearson-correlation", "spearman-correlation"]:
            if self.corr is None:
                if self.X is None:
                    raise ValueError("Either X or corr must be provided to compute the distance matrix.")
                else:
                    self.dist = get_corr_distance_matrix(
                        X=self.X,
                        corr_method=self.codependence_method.split("-")[0],
                        corr_to_dist_method=self.corr_to_dist_method,
                    )
            else:
                self.dist = corr_to_dist(corr=self.corr, corr_to_dist_method=self.corr_to_dist_method)
        elif self.codependence_method == "var_info":
            if self.X is None:
                raise ValueError("X must be provided to compute the distance matrix.")
            else:
                self.dist: DataFrameOrArray = get_info_distance_matrix(X=self.X, method="var_info")
        else:
            error_message = (
                f"Codependence method {self.codependence_method} is not supported. "
                "Expected 'pearson-correlation','spearman-correlation', 'var_info'."
            )
            raise ValueError(error_message)

    @property
    def linkage(self) -> np.ndarray:
        """Return the linkage matrix."""
        if self._linkage is None:
            self._linkage = hierarchy.linkage(
                y=squareform(self.dist),
                method=self.linkage_method,
                optimal_ordering=self.optimal_ordering,
            )
        return self._linkage

    def get_clusters(self, n_clusters: int) -> np.ndarray:
        """Get the clusters based on the threshold.

        Parameters
        ----------
        n_clusters : int
            The number of clusters.

        Returns
        -------
        np.ndarray
            The cluster labels.
        """
        return hierarchy.cut_tree(Z=self.linkage, n_clusters=n_clusters).flatten()

    def get_optimal_nclusters(self, max_clust: int = 20, metric: str = "gap", nb: int = 10) -> int:
        """Get the optimal number of clusters."""
        return optimize_n_clusters(linkage=self.linkage, dist=self.dist, max_clust=max_clust, metric=metric, nb=nb)


def optimize_n_clusters(
    linkage: np.ndarray, dist: np.ndarray, max_clust: int = 20, metric: str = "gap", nb: int = 10
) -> int:
    """Optimize the number of clusters based on a given method.

    Parameters
    ----------
    linkage : np.ndarray
        The linkage matrix.
    dist : np.ndarray
        The distance matrix.
    max_clust : int, optional
        Maximum number of clusters. Default is 20.
    metric : str, optional
        The optimization method. Default is 'gap'.
        Options are 'gap' and 'silhouette'.
    nb : int, optional
        Number of bootstrap samples. Default is 10.
        Only used if method is 'gap'. Otherwise, it is ignored.

    Returns
    -------
    n_clusters : int
        The optimal number of clusters.
    """
    nclust_range = range(2, max_clust + 1)
    if metric == "gap":
        gap_values = []
        gap_std_values = []
        for k in nclust_range:
            labels = hierarchy.cut_tree(linkage, n_clusters=k).flatten()
            gap, gap_std = gap_statistic(dist=dist, labels=labels, nb=nb)
            gap_values.append(gap)
            gap_std_values.append(gap_std)
        gap_values, gap_std_values = np.array(gap_values), np.array(gap_std_values)
        k = np.where(gap_values[:-1] - gap_values[1:] + gap_std_values[1:] > 0)[0][0]
        return int(k + 2)
    elif metric == "silhouette":
        silh_values = []
        for k in nclust_range:
            labels = hierarchy.cut_tree(linkage, n_clusters=k).flatten()
            silh_values.append(silhouette_tstat(dist=dist, labels=labels))
        return int(np.argmax(silh_values) + 2)
    else:
        raise ValueError(f"Method {metric} is not supported.")
