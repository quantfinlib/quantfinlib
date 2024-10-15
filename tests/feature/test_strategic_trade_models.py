import numpy as np
import pandas as pd
import pytest

from quantfinlib.sim._gbm import GeometricBrownianMotion
from quantfinlib.feature.micro_structure import get_amihud_lambda, get_kyle_lambda, get_hasbrouck_lambda

# Defining global variables for testing
gbm = GeometricBrownianMotion(drift=0.05, vol=0.30)

# Defining global variables for testing

P_CLOSE = gbm.path_sample(x0=100, label_start="2020-01-01", label_freq="B", num_steps=252, num_paths=1)

VOLUME = pd.Series(data=np.random.randint(100, 1000, len(P_CLOSE)), index=P_CLOSE.index)

# BUY_VOLUME = pd.Series(
#    data=np.random.rand(len(P_CLOSE)) * VOLUME.values,
#    index=P_HIGH.index
# )

WINDOWS = [10, 25]
INP_TYPE = [pd.Series, np.ndarray]
INPUTS = [(window, inp_type) for window in WINDOWS for inp_type in INP_TYPE]


@pytest.mark.parametrize("window, inp_type", INPUTS)
def test_kyle_lambda(window, inp_type):
    if inp_type == pd.Series:
        out = get_kyle_lambda(P_CLOSE, VOLUME, window=window)
        assert isinstance(out, pd.Series), "output is not a pandas Series"
    else:
        out = get_kyle_lambda(P_CLOSE.values, VOLUME.values, window=window)
        assert isinstance(out, np.ndarray), "output is not a numpy array"
    assert len(out) == len(P_CLOSE), "length mismatch between input array and output array"
    assert np.all(out[np.isfinite(out)] >= 0), "all finite values must be greater than or equal to 0"
    return None


def test_kyle_lambda_scaling_relations():
    out1 = get_kyle_lambda(P_CLOSE, VOLUME, window=10)
    out2 = get_kyle_lambda(P_CLOSE * 2.0, VOLUME, window=10)
    out3 = get_kyle_lambda(P_CLOSE, VOLUME * 2.0, window=10)
    out4 = get_kyle_lambda(P_CLOSE * 0, VOLUME, window=10)
    assert np.all(2.0 * out1.dropna() == out2.dropna())
    assert np.all(0.5 * out1.dropna() == out3.dropna())
    assert np.all(out4.dropna() == 0)
    return None


@pytest.mark.parametrize("window, inp_type", INPUTS)
def test_amihud_lambda(window, inp_type):
    if inp_type == pd.Series:
        out = get_amihud_lambda(P_CLOSE, VOLUME, window=window)
        assert isinstance(out, pd.Series), "output is not a pandas Series"
    else:
        out = get_amihud_lambda(P_CLOSE.values, VOLUME.values, window=window)
        assert isinstance(out, np.ndarray), "output is not a numpy array"
    assert len(out) == len(P_CLOSE), "length mismatch between input array and output array"
    assert np.all(out[np.isfinite(out)] >= 0), "all finite values must be greater than or equal to 0"
    return None


def test_amihud_lambda_scaling_relations():
    out1 = get_amihud_lambda(P_CLOSE, VOLUME, window=10)
    out2 = get_amihud_lambda(P_CLOSE * 2.0, VOLUME, window=10)
    out3 = get_amihud_lambda(P_CLOSE, VOLUME * 2.0, window=10)
    out4 = get_amihud_lambda(pd.Series(np.ones_like(P_CLOSE), index=P_CLOSE.index), VOLUME, window=10)
    assert np.all(0.5 * out1.dropna() == out2.dropna())
    assert np.all(0.5 * out1.dropna() == out3.dropna())
    assert np.all(out4.dropna() == 0)
    return None


@pytest.mark.parametrize("window, inp_type", INPUTS)
def test_hasbrouck_lambda(window, inp_type):
    if inp_type == pd.Series:
        out = get_hasbrouck_lambda(P_CLOSE, VOLUME, window=window)
        assert isinstance(out, pd.Series), "output is not a pandas Series"
    else:
        out = get_hasbrouck_lambda(P_CLOSE.values, VOLUME.values, window=window)
        assert isinstance(out, np.ndarray), "output is not a numpy array"
    assert len(out) == len(P_CLOSE), "length mismatch between input array and output array"
    assert np.all(out[np.isfinite(out)] >= 0), "all finite values must be greater than or equal to 0"
    return None


def test_hasbrouck_lambda_scaling_relations():
    out = get_hasbrouck_lambda(p_close=P_CLOSE, volume=VOLUME, window=10)
    out2 = get_hasbrouck_lambda(p_close=P_CLOSE * 2, volume=VOLUME, window=10)
    out3 = get_hasbrouck_lambda(P_CLOSE, VOLUME * 2, window=10)
    out4 = get_hasbrouck_lambda(pd.Series(np.ones_like(P_CLOSE), index=P_CLOSE.index), VOLUME, window=10)
    assert np.allclose(out.dropna() / np.sqrt(2) , out2.dropna())
    assert np.allclose(out.dropna() / np.sqrt(2) , out3.dropna())
    assert np.allclose(out4.dropna(), 0)
