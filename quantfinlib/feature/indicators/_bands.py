"""Functions to calculate the Bollinger Bands of a stock."""

import numpy as np
import pandas as pd

from typing import Union

from ._base import rolling_mean, rolling_std, rolling_max, rolling_min, average_true_range, ewm_mean, ewm_std


class GenericBands:
    def __init__(
        self,
        ts: Union[pd.Series, np.ndarray],
        rolling_mean: Union[pd.Series, np.ndarray],
        rolling_std: Union[pd.Series, np.ndarray],
        multiplier: int
    ):
        self.ts = ts
        self.rolling_mean = rolling_mean
        self.rolling_std = rolling_std
        self.multiplier = multiplier
        if isinstance(ts, pd.Series) or isinstance(rolling_mean, pd.Series) or isinstance(rolling_std, pd.Series):
            self.rename = self._rename_pd
        else:
            self.rename = lambda x, _: x
        self.basename = ""

    def _rename_pd(self, s: pd.Series, name: str) -> pd.Series:
        return s.rename(name + self.basename)

    def upper(self) -> Union[pd.Series, np.ndarray]:
        return self.rename(self.rolling_mean + (self.rolling_std * self.multiplier), "Upper")

    def lower(self) -> Union[pd.Series, np.ndarray]:
        return self.rename(self.rolling_mean - (self.rolling_std * self.multiplier), "Lower")

    def middle(self) -> Union[pd.Series, np.ndarray]:
        return self.rename(self.rolling_mean, "Middle")

    def bandwidth(self) -> Union[pd.Series, np.ndarray]:
        return self.rename(2 * self.rolling_std * self.multiplier / self.middle(), "Bandwidth")

    def percent_b(self) -> Union[pd.Series, np.ndarray]:
        return self.rename((self.ts - self.lower()) / (2 * self.rolling_std * self.multiplier), "%B")


class BollingerBands(GenericBands):
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
            multiplier
        )
        self.basename = " Bollinger"


class EwmBollingerBands(GenericBands):
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
            multiplier
        )
        self.basename = " EwmBollinger"


class KeltnerBands(GenericBands):
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
            multiplier
        )
        self.basename = " Keltner"


class DonchianBands(GenericBands):
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
            1
        )
        self.basename = " Donchian"
