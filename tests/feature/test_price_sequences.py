from quantfinlib.feature.micro_structure._price_sequences import (
    _edge_spread,
    get_close_close_volatility,
    get_edge_spread,
    get_high_low_volatility,
    get_garman_klass_volatility,
    get_becker_parkinson_volatility,
    get_yang_zhang_volatility,
    get_rogers_satchell_volatility,
    get_cowrin_schultz_spread,
    get_roll_measure,
    get_roll_impact,
)

import numpy as np
import pandas as pd
import pytest

from quantfinlib.sim._gbm import GeometricBrownianMotion

gbm = GeometricBrownianMotion(drift=0.05, vol=0.30)

# Defining global variables for testing

P_HIGH = gbm.path_sample(x0=100, label_start="2020-01-01", label_freq="B", num_steps=252, num_paths=1)
P_LOW = P_HIGH.copy() * 0.90
P_OPEN = P_HIGH.copy() - 0.5 * (P_HIGH - P_LOW)
P_CLOSE = P_HIGH.copy() - 0.25 * (P_HIGH - P_LOW)
VOLUME = pd.Series(data=np.random.randint(100, 1000, len(P_HIGH)), index=P_HIGH.index)


FUNC = [
    get_close_close_volatility,
    get_high_low_volatility,
    get_garman_klass_volatility,
    get_becker_parkinson_volatility,
    get_yang_zhang_volatility,
    get_rogers_satchell_volatility,
    get_edge_spread,
    get_cowrin_schultz_spread,
    get_roll_measure,
    get_roll_impact,
]
WINDOWS = [10, 25]
INPUTS = [(func, window) for func in FUNC for window in WINDOWS]


@pytest.mark.parametrize("func, window", INPUTS)
def test_func_numpy_input(func, window):
    if func in [get_close_close_volatility]:
        out = func(P_CLOSE.values, window=window)
    if func in [get_high_low_volatility, get_cowrin_schultz_spread, get_becker_parkinson_volatility]:
        out = func(P_HIGH.values, P_LOW.values, window=window)
    elif func in [get_roll_measure]:
        out = func(P_HIGH.values, window=window)
    elif func in [get_roll_impact]:
        out = func(P_CLOSE.values, VOLUME.values, window=window)
    elif func in [
        get_garman_klass_volatility,
        get_yang_zhang_volatility,
        get_rogers_satchell_volatility,
        get_edge_spread,
    ]:
        out = func(
            p_close=P_CLOSE.values, p_high=P_HIGH.values, p_low=P_LOW.values, p_open=P_OPEN.values, window=window
        )
    assert len(out) == len(P_HIGH)  # output has the same length as input
    assert np.all(out[np.isfinite(out)] > 0)  # all values are positive


@pytest.mark.parametrize("func, window", INPUTS)
def test_func_pd_input(func, window):
    if func in [get_close_close_volatility]:
        out = func(P_CLOSE, window=window)
    if func in [get_high_low_volatility, get_cowrin_schultz_spread, get_becker_parkinson_volatility]:
        out = func(P_HIGH, P_LOW, window=window)
    elif func in [get_roll_measure]:
        out = func(P_HIGH, window=window)
    elif func in [get_roll_impact]:
        out = func(P_CLOSE, VOLUME, window=window)
    elif func in [
        get_garman_klass_volatility,
        get_yang_zhang_volatility,
        get_rogers_satchell_volatility,
        get_edge_spread,
    ]:
        out = func(
            p_close=P_CLOSE, p_high=P_HIGH, p_low=P_LOW, p_open=P_OPEN, window=window
        )
    assert len(out) == len(P_HIGH)  # output has the same length as input
    assert np.all(out[np.isfinite(out)] > 0)  # all values are positive


def test_edge():

    df = pd.read_csv("https://raw.githubusercontent.com/eguidotti/bidask/main/pseudocode/ohlc.csv")

    estimate = _edge_spread(p_high=df.High, p_low=df.Low, p_open=df.Open, p_close=df.Close)
    assert estimate == pytest.approx(0.0101849034905478)

    estimate = _edge_spread(
        p_high=df.High[0:10], p_low=df.Low[0:10], p_open=df.Open[0:10], p_close=df.Close[0:10], sign=True
    )
    assert estimate == pytest.approx(-0.016889917516422)

    assert np.isnan(
        _edge_spread(
            pd.Series([18.21, 17.61, 17.61]),
            pd.Series([18.21, 17.61, 17.61]),
            pd.Series([17.61, 17.61, 17.61]),
            pd.Series([17.61, 17.61, 17.61]),
        )
    )
