"""Implementations of Kyle lambda, Amihud lambda, and Hasbrouck lambda."""

from quantfinlib.feature.indicators._base import numpy_io_support

import numpy as np
import pandas as pd


@numpy_io_support
def get_kyle_lambda(p_close: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    r"""Calculate the Kyle lambda.

    The Kyle lambda is defined as:

    .. math::

            \lambda_{t} = \frac{\Delta p_{t}}{v_{t} b_{t}}

    where:

    - :math:`\Delta p_{t}` is the price change at time :math:`t`
    - :math:`v_{t}` is the traded volume at time :math:`t`
    - :math:`b_{t}` is the sign of the price change at time :math:`t`

    Parameters
    ----------
    p_close : pd.Series
        The close price series.
    volume : pd.Series
        The trade volume series.
    window : int, optional
        The window size for calculating the Kyle lambda, by default 10.

    Returns
    -------
    pd.Series
        The Kyle lambda series.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3345183
    """
    delta_p = p_close.diff()
    delta_p_sign = np.sign(delta_p)
    ratio = delta_p / (delta_p_sign * volume)
    return ratio.rolling(window).mean()


@numpy_io_support
def get_amihud_lambda(p_close: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    r"""Calculate the Amihud lambda.

    The Amihud lambda is defined as:

    .. math::

                \lambda_{t} = \frac{|\Delta p_{t}|}{v_{t}}

    where:
    - :math:`\Delta p_{t}` is the price percentage change at time :math:`t`
    - :math:`v_{t}` is the dollar volume at time :math:`t`

    Parameters
    ----------
    p_close : pd.Series
        The close price series.
    volume : pd.Series
        The traded volume series.
    window : int, optional
        The window size for calculating the Amihud lambda, by default 10.

    Returns
    -------
    pd.Series
        The Amihud lambda series.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3345183
    """
    price_pct_change = p_close.pct_change()
    ratio = price_pct_change.abs() / (volume * p_close)
    return ratio.rolling(window).mean()


@numpy_io_support
def get_hasbrouck_lambda(p_close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    r"""Calculate Habrouck's lambda.

    The Habrouck lambda is defined as:

    .. math::

        \lambda_{t} = \frac{\Delta (\log(p_{t}))}{b_{t}\sqrt{p_{t}.v_{t}}}

    where:

    - :math:`\Delta (\log(p_{t}))` is the log return at time :math:`t`
    - :math:`b_{t}` is the sign of the log return at time :math:`t`
    - :math:`p_{t}` is the price at time :math:`t`
    - :math:`v_{t}` is the trade volume at time :math:`t`

    Parameters
    ----------
    p_close : pd.Series
        The close price series.
    volume : pd.Series
        The trade volume series.
    window : int, optional
        The window size for calculating the Habrouck lambda, by default 20.

    Returns
    -------
    pd.Series
        The Habrouck lambda series.

    References
    ----------
    page 289 of Advances in Financial Machine Learning by Marcos Lopez de Prado.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3270269
    """
    log_p_diff = np.log(p_close / p_close.shift(1))
    log_p_diff_sign = np.sign(log_p_diff)
    ratio = log_p_diff / (log_p_diff_sign * np.sqrt(volume * p_close))
    return ratio.rolling(window).mean()
