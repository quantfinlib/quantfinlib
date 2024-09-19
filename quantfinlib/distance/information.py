"""Functions to compute information-theoretical measures of codependence betwween two variables.
These include mutual information, variation of information, and Kullback-Leibler divergence between two variables.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.stats import entropy

from quantfinlib.util import SeriesOrArray, validate_series_or_1Darray


def _get_nb_bins_from_xy(x: np.ndarray, y: Optional[np.ndarray] = None) -> int:
    """Calculate number of bins for discretizing a pair of variables x & y."""
    if y is None:
        return _get_optimal_nb_bins(x.shape[0], None)
    else:
        corr = np.corrcoef(x, y)[0, 1]
        if corr == 1 or abs(corr - 1) < 1e-5:
            return _get_optimal_nb_bins(x.shape[0], None)
        else:
            return _get_optimal_nb_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])


def _get_optimal_nb_bins(n_obs: int, corr: Optional[float] = None) -> int:
    """Get optimal number of bins for calculating mutual (variation of) information.

    This function calculates the number of bins needed
    (according to Hacine-Gharbi and Ravier's method) for calculation of
    variation of (mutual) information between two variables in discrete version.


    Parameters
    ----------
    n_obs : int
        Number of observations.
    corr : float, optional
        Correlation coefficient between two variables.

    Returns
    -------
    int
        Number of bins.
    """
    # Univariate case
    if corr is None:
        z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs**2) ** 0.5) ** (1 / 3.0)
        b = round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3)
    # Bivariate case
    else:
        b = round(2**-0.5 * (1 + (1 + 24 * n_obs / (1.0 - corr**2)) ** 0.5) ** 0.5)
    return int(b)


@validate_series_or_1Darray("x", "y")
def compute_entropies(x: SeriesOrArray, y: SeriesOrArray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute entropies of two variables x and y.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        First input variable.
    y : pd.Series or np.ndarray
        Second input variable.
    bins : int
        Number of bins for histogram calculation.

    Returns
    -------
    np.ndarray , np.ndarray, np.ndarray
        Entropy of x, Entropy of y, Joint entropy of x and y.
    """
    # Compute joint histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins)
    hist_x, _ = np.histogram(x, bins)
    hist_y, _ = np.histogram(y, bins)
    # Compute entropies
    h_x = entropy(hist_x / hist_x.sum())
    h_y = entropy(hist_y / hist_y.sum())
    h_xy = entropy((hist_xy / hist_xy.sum()).flatten())
    return h_x, h_y, h_xy


@validate_series_or_1Darray("x", "y")
def mutual_info(x: SeriesOrArray, y: SeriesOrArray, bins: Optional[int] = None, norm: bool = True) -> float:
    r"""
    Calculate mutual information between two variables.

    Mutual information is calculated using the formula:

    .. math::
        I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)

    where \( p(x,y) \) is the joint probability distribution function of X and Y,
    and \( p(x) \) and \( p(y) \) are the marginal probability distribution functions of X and Y respectively.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        First input variable.
    y : pd.Series or np.ndarray
        Second input variable.
    bins : int, optional
        Number of bins for histogram calculation. If None, optimal bins are computed.
    norm : bool, default=True
        Whether to normalize the mutual information.

    Returns
    -------
    float
        Mutual information between x and y.

    Examples
    --------
    .. exec_code::

        import numpy as np
        from quantfinlib.distance.information import mutual_info

        np.random.seed(1234)
        x = np.random.normal(0, 1, 100)
        noise = np.random.normal(0, .5, 100)
        y_rnd = np.random.normal(0, 1, 100)
        y_linear = 10 * x + noise
        y_nonlinear = 10 * x ** 2. + noise

        for y in [y_rnd, y_linear, y_nonlinear]:
            mi_xy = mutual_info(x, y)
            print(f"Mutual information = {mi_xy}")
    """
    if bins is None:
        bins = _get_nb_bins_from_xy(x, y)
    h_x, h_y, h_xy = compute_entropies(x, y, bins)
    # Compute mutual information
    mi_xy = h_x + h_y - h_xy
    if norm:
        mi_xy /= np.sqrt(h_x * h_y)  # Normalized mutual information
    return mi_xy


@validate_series_or_1Darray("x", "y")
def var_info(x: SeriesOrArray, y: SeriesOrArray, bins: Optional[int] = None, norm: bool = True) -> float:
    """Calculate variation of information between two variables.

    Implementation of variation of information adopted from "Machine Learning for Asset Managers by Lopez de Prado".

    Parameters
    ----------
    x : pd.Series, np.ndarray
        First argument of var_info.
    y : pd.Series, np.ndarray
        Second argument of var_info.
    bins : int, optional
        When no value provided, the function get_optimal_nb_bins
        will be called to calculate the optimal number of
        bins for calculation of variation of information.
    norm : bool, default = True
        Whether to normalize the variation of information or not.

    Returns
    -------
    float
        Mutual information between x and y.

    Examples
    --------
    .. exec_code::

        import numpy as np
        from quantfinlib.distance.information import var_info

        np.random.seed(1234)
        x = np.random.normal(0, 1, 100)
        noise = np.random.normal(0, .5, 100)
        y_rnd = np.random.normal(0, 1, 100)
        y_linear = 10 * x + noise
        y_nonlinear = 10 * x ** 2. + noise

        for y in [y_rnd, y_linear, y_nonlinear]:
            vi_xy = var_info(x, y)
            print(f"variation of information = {vi_xy}")
    """
    if bins is None:
        bins = _get_nb_bins_from_xy(x, y)
    h_x, h_y, h_xy = compute_entropies(x, y, bins)  # Compute entropies
    # Compute variation of information
    var_info_xy = 2 * h_xy - h_x - h_y
    if norm:
        var_info_xy /= h_xy  # Normalized variation of information
    return var_info_xy


@validate_series_or_1Darray("x", "y")
def kl_divergence_xy(x: SeriesOrArray, y: SeriesOrArray, bins: Optional[int] = None) -> float:
    """
    Calculate the Kullback-Leibler divergence between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable.
    y : np.ndarray
        Second variable.
    bins : int, optional
        Number of bins for histogram calculation. If None, optimal bins are computed.

    Returns
    -------
    float
        Kullback-Leibler divergence between x and y.
    """
    if bins is None:
        bins = _get_nb_bins_from_xy(x, y)
    # Discretize the variables
    hist_x, _ = np.histogram(x, bins=bins)
    hist_y, _ = np.histogram(y, bins=bins)

    # Compute the probability distributions
    p_x = hist_x / hist_x.sum()
    p_y = hist_y / hist_y.sum()

    # Compute the KL divergence
    kl_div = np.sum(np.where((p_x != 0) & (p_y != 0), p_x * np.log(p_x / p_y), 0))
    return kl_div
