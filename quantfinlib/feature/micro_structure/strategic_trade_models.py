import numpy as np 

from quantfinlib.feature.transform.moving_window_func import moving_average


def kyle_lambda(price: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    r"""
    Calculate the Kyle lambda.

    The Kyle lambda is defined as:

    .. math::
    
            \lambda_{t} = \frac{\Delta p_{t}}{v_{t} b_{t}}
    
    where:
    
    - :math:`\Delta p_{t}` is the price change at time :math:`t`
    - :math:`v_{t}` is the volume at time :math:`t`
    - :math:`b_{t}` is the sign of the price change at time :math:`t`

    Parameters
    ----------
    price : np.ndarray
        The price.
    volume : np.ndarray
        The volume.
    window : int
        The window size.

    Returns
    -------
    np.ndarray
        The Kyle lambda.
    """
    delta_p = np.diff(price)
    diff_sign = np.sign(delta_p)
    ratio = delta_p / (diff_sign * volume)
    kyle_lambda = moving_average(ratio , window_size=window)
    return kyle_lambda


def amihud_lambda(price: np.ndarray, dollar_volume: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Amihud lambda.

    Parameters
    ----------
    price : np.ndarray
        The price.
    dollar_volume : np.ndarray
        The dollar volume.
    window : int
        The window size.
    
    Returns
    -------
    np.ndarray
        The Amihud lambda
    """
    price_pct_change = price.copy()
    price_pct_change[1:] = np.diff(price) / price[:-1]
    price_pct_change[0] = 0
    ratio = np.abs(price_pct_change) / dollar_volume
    amihud_lambda = moving_average(ratio , window_size=window)
    return amihud_lambda


def vpin(volume: np.ndarray, buy_volume: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Volume-synchronized Probability of Informed Trading (VPIN).

    Parameters
    ----------
    volume : np.ndarray
        The volume.
    buy_volume : np.ndarray
        The buy volume.
    window : int
        The window size.
    
    Returns
    -------
    np.ndarray
        The VPIN.
    """
    sell_volume = volume - buy_volume
    volume_imbalance = np.abs(buy_volume - sell_volume)
    vpin = moving_average(volume_imbalance, window_size=window)/volume
    return vpin
