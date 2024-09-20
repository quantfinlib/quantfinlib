"""Functions to calculate the Bollinger Bands, Keltner Channels, and Donchian Channels."""

import numpy as np
import pandas as pd

from typing import Union

from ._base import rolling_mean, rolling_std, rolling_max, rolling_min, average_true_range, ewm_mean, ewm_std


_generic_docstring = """
    Attributes
    ----------
    ts : Union[pd.Series, np.ndarray]
        The time series data.
    rolling_mean : Union[pd.Series, np.ndarray]
        The rolling mean of the time series data.
    rolling_std : Union[pd.Series, np.ndarray]
        The rolling standard deviation of the time series data.
    multiplier : int
        The multiplier used to calculate the upper and lower bands.
    basename : str
        The base name used for renaming the series.

    Methods
    -------
    upper() -> Union[pd.Series, np.ndarray]
        Calculate the upper band.
    lower() -> Union[pd.Series, np.ndarray]
        Calculate the lower band.
    middle() -> Union[pd.Series, np.ndarray]
        Calculate the middle band.
    bandwidth() -> Union[pd.Series, np.ndarray]
        Calculate the bandwidth.
    percent_b() -> Union[pd.Series, np.ndarray]
        Calculate the percent b.
"""


class GenericBands:
    """
    Generic bands for time series data.
    """
    def __init__(
        self,
        ts: Union[pd.Series, np.ndarray],
        rolling_mean: Union[pd.Series, np.ndarray],
        rolling_std: Union[pd.Series, np.ndarray],
        multiplier: int,
        basename: str = ""
    ):
        self.ts = ts
        self.rolling_mean = rolling_mean
        self.rolling_std = rolling_std
        self.multiplier = multiplier
        if isinstance(ts, pd.Series) or isinstance(rolling_mean, pd.Series) or isinstance(rolling_std, pd.Series):
            self._rename = self._rename_pd
        else:
            self._rename = lambda x, _: x
        self.basename = basename

    def _rename_pd(self, s: pd.Series, name: str) -> pd.Series:
        return s.rename(name + self.basename)

    def upper(self) -> Union[pd.Series, np.ndarray]:
        """
        Calculate the upper band of the indicator.
        Returns:
            Union[pd.Series, np.ndarray]: The upper band values.
        """
        return self._rename(self.rolling_mean + (self.rolling_std * self.multiplier), "Upper")

    def lower(self) -> Union[pd.Series, np.ndarray]:
        """
        Calculate the lower band of the indicator.
        Returns:
            Union[pd.Series, np.ndarray]: The lower band values.
        """
        return self._rename(self.rolling_mean - (self.rolling_std * self.multiplier), "Lower")

    def middle(self) -> Union[pd.Series, np.ndarray]:
        """
        Calculate the middle band of the indicator.
        Returns:
            Union[pd.Series, np.ndarray]: The middle band values.
        """
        return self._rename(self.rolling_mean, "Middle")

    def bandwidth(self) -> Union[pd.Series, np.ndarray]:
        """
        Calculate the bandwidth of the indicator.
        Returns:
            Union[pd.Series, np.ndarray]: The bandwidth values.
        """
        return self._rename(2 * self.rolling_std * self.multiplier / self.middle(), "Bandwidth")

    def percent_b(self) -> Union[pd.Series, np.ndarray]:
        """
        Calculate the percent b of the indicator.
        Returns:
            Union[pd.Series, np.ndarray]: The percent b values.
        """
        return self._rename((self.ts - self.lower()) / (2 * self.rolling_std * self.multiplier), "%B")


GenericBands.__doc__ += _generic_docstring


class BollingerBands(GenericBands):
    """
    Bollinger Bands for time series data.

    Parameters:
    -----------
    ts : Union[pd.Series, np.ndarray]
        The time series data.
    window : int, optional
        The window size for calculating the rolling mean and standard deviation, by default 20.
    multiplier : int, optional
        The multiplier for calculating the upper and lower bands, by default 2.
    """
    def __init__(
        self,
        ts: Union[pd.Series, np.ndarray],
        window: int = 20,
        multiplier: int = 2
    ):
        super().__init__(
            ts,
            rolling_mean(ts, window=window),
            rolling_std(ts, window=window),
            multiplier,
            basename=" Bollinger"
        )


BollingerBands.__doc__ += _generic_docstring


class EwmBollingerBands(GenericBands):
    """
    Exponential Weighted Moving Average Bollinger Bands for time series data.

    Parameters:
    -----------
    ts : Union[pd.Series, np.ndarray]
        The time series data.
    window : int, optional
        The window size for calculating the exponential weighted moving average and standard deviation, by default 20.
    multiplier : int, optional
        The multiplier for calculating the upper and lower bands, by default 2.
    """
    def __init__(
        self,
        ts: Union[pd.Series, np.ndarray],
        window: int = 20,
        multiplier: int = 2
    ):
        super().__init__(
            ts,
            ewm_mean(ts, span=window),
            ewm_std(ts, span=window),
            multiplier,
            basename=" EwmBollinger"
        )


EwmBollingerBands.__doc__ += _generic_docstring


class KeltnerBands(GenericBands):
    """
    Keltner Channels for time series data.

    Parameters:
    -----------
    high : Union[pd.Series, np.ndarray]
        The high prices.
    low : Union[pd.Series, np.ndarray]
        The low prices.
    close : Union[pd.Series, np.ndarray]
        The close prices.
    window_atr : int, optional
        The window size for calculating the average true range, by default 10.
    window : int, optional
        The window size for calculating the rolling mean, by default 20.
    multiplier : int, optional
        The multiplier for calculating the upper and lower bands, by default 2.
    """
    def __init__(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        window_atr: int = 10,
        window: int = 20,
        multiplier: int = 2
    ):
        super().__init__(
            close,
            ewm_mean(close, span=window),
            average_true_range(high, low, close, window=window_atr),
            multiplier,
            basename=" Keltner"
        )


KeltnerBands.__doc__ += _generic_docstring


class DonchianBands(GenericBands):
    """
    Donchian Channels for time series data.

    Parameters:
    -----------
    ts : Union[pd.Series, np.ndarray]
        The time series data.
    window : int, optional
        The window size for calculating the rolling maximum and minimum, by default 20.
    """
    def __init__(
        self,
        ts: Union[pd.Series, np.ndarray],
        window: int = 20
    ):
        ub = rolling_max(ts, window=window)
        lb = rolling_min(ts, window=window)
        super().__init__(
            ts,
            (ub + lb) / 2,
            (ub - lb) / 2,
            1,
            basename=" Donchian"
        )


DonchianBands.__doc__ += _generic_docstring
