"""
Technical Indicator Module.
=================

This module contains functions for calculating various technical indicators from time series data.


.. currentmodule:: quantfinlib.feature.indicators

.. autosummary::
    :toctree: _autosummary

    rsi          Calculate the Relative Strength Index (RSI) of a time series.
    ewm_rsi      Calculate the Exponential Weighted Moving Average (EWMA) of the RSI.
    macd         Calculate the Moving Average Convergence Divergence (MACD) of a time series.
    macd_signal  Calculate the Signal Line of the MACD.
    BollingerBands  Calculate the Bollinger Bands of a time series.
    EwmBollingerBands  Calculate the Exponential Weighted Moving Average (EWMA) of the Bollinger Bands.
    KeltnerBands  Calculate the Keltner Bands of a time series.
    DonchianBands  Calculate the Donchian Bands of a time series.
    rolling_mean  Calculate the rolling mean of a time series.
    rolling_std   Calculate the rolling standard deviation of a time series.
    rolling_max   Calculate the rolling maximum of a time series.
    rolling_min   Calculate the rolling minimum of a time series.
    ewm_mean      Calculate the Exponential Weighted Moving Average (EWMA) of a time series.
    ewm_std       Calculate the Exponential Weighted Moving Standard Deviation of a time series.
    average_true_range  Calculate the Average True Range of a time series.
    rolling_mom   Calculate the rolling momentum of a time series.
    ewm_mom      Calculate the Exponential Weighted Moving Average (EWMA) of the momentum.

"""

from ._bands import BollingerBands, EwmBollingerBands, KeltnerBands, DonchianBands
from ._base import (
    rolling_mean, rolling_std, rolling_max, rolling_min, ewm_mean, ewm_std, average_true_range,
    rolling_mom, ewm_mom
)
from ._macd import macd, macd_signal
from ._rsi import rsi, ewm_rsi
