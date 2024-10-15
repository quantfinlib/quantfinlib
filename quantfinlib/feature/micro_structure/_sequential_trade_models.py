"""Implementations of Volume-synchronized probability of informed trading."""

from quantfinlib.feature.indicators._base import numpy_io_support

import pandas as pd
from scipy.stats import t


@numpy_io_support
def get_vpin(volume: pd.Series, buy_volume: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Volume-synchronized Probability of Informed Trading (VPIN).

    Parameters
    ----------
    volume : pd.Series
        The volume.
    buy_volume : pd.Series
        The buy volume.
    window : int
        The window size.

    Returns
    -------
    pd.Series
        The VPIN.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3345183
    """
    sell_volume = volume - buy_volume
    volume_imbalance = (buy_volume - sell_volume).abs() / volume
    return volume_imbalance.rolling(window).mean()


@numpy_io_support
def estimate_buy_volume(p_close: pd.Series, volume: pd.Series, sigma_p: pd.Series) -> pd.Series:
    r"""Estimate the buy volume with the Bulk Classification Volume method of Eagley et al. (2016).

    .. math::

        \hat{V}_{\tau}^{B} = V_{\tau} \cdot t\left( \frac{P_{\tau} - P_{\tau-1}}{\sigma \Delta P}, \text{df} \right)

    where:
    - :math:`V_{\tau}` is the volume at time :math:`\tau`
    - :math:`\hat{V}_{\tau}^{B}` is the estimate of buy volume at time :math:`\tau`
    - :math:`P_{\tau}` is the price at time :math:`\tau`
    - :math:`\sigma \Delta P` is the standard deviation of the price series between two bars: `\tau` and `\tau-1`
    - :math:`\text{df}` is the degrees of freedom.
    - :math:`t` The cumulative distribution function (CDF) of Student's t distribution with `df` degrees of freedom.

    Eagle et al. (2016) suggest setting the parameter `df` to 0.25 to account for the
    fat tails in the underlying distribution.

    Parameters
    ----------
    p_close : pd.Series
        The close price series.
    volume : pd.Series
        The volume series.
    sigma_p : pd.Series
        The standard deviation of the price series between two bars.
        For instance, if the price series `p_close` are daily, then `sigma_p` is the
        standard deviation of intraday price changes.

    Returns
    -------
    pd.Series
        The estimated buy volume.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3345183

    """
    normalized_price_change = p_close.diff() / sigma_p
    buy_volume = volume * t.cdf(x=normalized_price_change, df=0.25)
    return buy_volume
