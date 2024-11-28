import numpy as np
import pandas as pd
import pytest

from quantfinlib.feature.indicators._base import numpy_io_support
from quantfinlib.feature.indicators import (
    rolling_mean, rolling_std, rolling_max, rolling_min, ewm_mean, ewm_std, average_true_range,
    rolling_mom, ewm_mom,
    BollingerBands, EwmBollingerBands, KeltnerBands, DonchianBands,
    macd, macd_signal, rsi, ewm_rsi
)


def test_numpy_io_support_1():
    @numpy_io_support
    def test_func(a: pd.Series, b: int = 5) -> pd.Series:
        assert isinstance(a, pd.Series)
        assert isinstance(b, int)
        return a + b
    # Numpy array as input and output
    result = test_func(np.array([1, 2, 3]))
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([6, 7, 8]))
    # Pandas Series as input and output
    result = test_func(pd.Series([1, 2, 3]))
    assert isinstance(result, pd.Series)
    # Numpy array as input with keyword argument
    result = test_func(np.array([1, 2, 3]), b=4)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([5, 6, 7]))
    # Test exception
    with pytest.raises(TypeError) as e:
        test_func([1, 2, 3])
    assert str(e.value) == "Argument 'a' must be either numpy array or pandas Series."
    # Test positional argument as keyword argument
    result = test_func(a=np.array([1, 2, 3]), b=4)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([5, 6, 7]))


def test_numpy_io_support_2():
    @numpy_io_support
    def test_func(a: pd.Series, b: pd.Series, c: int = 5) -> pd.Series:
        assert isinstance(a, pd.Series)
        assert isinstance(b, pd.Series)
        assert isinstance(c, int)
        return a + b + c
    # Numpy arrays as input and output
    result = test_func(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert isinstance(result, np.ndarray)
    # Mixed inputs give TypeError
    with pytest.raises(TypeError) as e:
        test_func(np.array([1, 2, 3]), pd.Series([4, 5, 6]))
    assert str(e.value) == "Cannot mix numpy arrays with pandas Series in input."
    with pytest.raises(TypeError) as e:
        test_func(pd.Series([4, 5, 6]), np.array([1, 2, 3]))
    assert str(e.value) == "Cannot mix numpy arrays with pandas Series in input."
    # Pandas Series as input and output
    result = test_func(pd.Series([1, 2, 3]), pd.Series([4, 5, 6]))
    assert isinstance(result, pd.Series)
    assert np.array_equal(result, np.array([10, 12, 14]))
    # Numpy arrays as input with keyword argument
    result = test_func(np.array([1, 2, 3]), np.array([4, 5, 6]), c=7)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([12, 14, 16]))
    # Numpy arrays as input with positional argument
    result = test_func(np.array([1, 2, 3]), np.array([4, 5, 6]), 7)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([12, 14, 16]))


def test_numpy_io_support_3():
    @numpy_io_support
    def test_func(a: pd.Series, b: pd.Series, c: int = 5, d: int = 4) -> pd.Series:
        assert isinstance(a, pd.Series)
        assert isinstance(b, pd.Series)
        assert isinstance(c, int)
        assert isinstance(d, int)
        return a + b + c + d
    result = test_func(np.array([1, 2, 3]), np.array([4, 5, 6]), 7, 8)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([20, 22, 24]))
    result = test_func(np.array([1, 2, 3]), np.array([4, 5, 6]), c=7, d=8)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([20, 22, 24]))
    result = test_func(np.array([1, 2, 3]), np.array([4, 5, 6]), 7, d=8)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([20, 22, 24]))
    result = test_func(np.array([1, 2, 3]), b=np.array([4, 5, 6]), c=7, d=8)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([20, 22, 24]))


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_rolling_mean(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rm = rolling_mean(ts, window=3)
    assert isinstance(rm, dtype)
    np.testing.assert_array_equal(rm, [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert rolling_mean.__name__ == "rolling_mean"
    assert rolling_mean.__doc__.strip().startswith("Calculate the rolling mean of a time series.")


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_rolling_std(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rs = rolling_std(ts, window=3)
    assert isinstance(rs, dtype)
    np.testing.assert_array_equal(rs, [np.nan] * 2 + [1.0] * 8)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_rolling_max(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rm = rolling_max(ts, window=3)
    assert isinstance(rm, dtype)
    np.testing.assert_array_equal(rm, [np.nan] * 2 + [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_rolling_min(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rm = rolling_min(ts, window=3)
    assert isinstance(rm, dtype)
    np.testing.assert_array_equal(rm, [np.nan] * 2 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_ewm_mean(init, dtype):
    ts = init([1, 2, 3, 4])
    em = ewm_mean(ts, span=3)
    assert isinstance(em, dtype)
    expected = [1.0, 2.5 / 1.5, 4.25 / 1.75, 6.125 / 1.875]
    np.testing.assert_array_equal(em, expected)
    ts = init([1, 1, 1, 1])
    em = ewm_mean(ts, span=3)
    assert isinstance(em, dtype)
    expected = [1.0, 1.0, 1.0, 1.0]
    np.testing.assert_array_equal(em, expected)


@pytest.fixture
def expected_ewmstd():
    weights = np.array([0.125, 0.25, 0.5, 1])
    expected = [np.nan]
    for i in range(2, 5):
        v = np.arange(1, i + 1)
        w = weights[-i:]
        sw = sum(w)
        ewma = sum(w * v) / sw
        ewmstd = np.sqrt(sw / (sw**2 - sum(w**2)) * sum(w * (v - ewma)**2))
        expected.append(ewmstd)
    return expected


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_ewm_std(init, dtype, expected_ewmstd):
    ts = init([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    es = ewm_std(ts, span=3)
    assert isinstance(es, dtype)
    expected = init([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(es, expected)

    ts = init([1, 2, 3, 4])
    es = ewm_std(ts, span=3)
    assert isinstance(es, dtype)
    np.testing.assert_allclose(es, expected_ewmstd, rtol=1e-9)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_average_true_range(init, dtype):
    high = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    low = init([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    close = high
    atr = average_true_range(high, low, close, window=2)
    assert isinstance(atr, dtype)
    expected = init([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(atr, expected)
    low = init([-0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    close = high - 0.5
    atr = average_true_range(high, low, close, window=2)
    assert isinstance(atr, dtype)
    expected = init([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_array_equal(atr, expected)
    close = high + 1.5
    atr = average_true_range(high, low, close, window=2)
    assert isinstance(atr, dtype)
    expected = init([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_array_equal(atr, expected)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_rolling_mom(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rm = rolling_mom(ts, window=3)
    assert isinstance(rm, dtype)
    np.testing.assert_array_equal(rm, [np.nan] * 3 + [1.0] * 7)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_ewm_mom(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    em = ewm_mom(ts, span=3, min_periods=3)
    assert isinstance(em, dtype)
    np.testing.assert_array_equal(em, [np.nan] * 3 + [1.0] * 7)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_bollinger_bands(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bb = BollingerBands(ts, window=3, multiplier=2)
    assert isinstance(bb.middle(), dtype)
    assert isinstance(bb.upper(), dtype)
    assert isinstance(bb.lower(), dtype)
    assert isinstance(bb.bandwidth(), dtype)
    assert isinstance(bb.percent_b(), dtype)
    if dtype == pd.Series:
        assert bb.middle().name == "Middle Bollinger"
        assert bb.upper().name == "Upper Bollinger"
        assert bb.lower().name == "Lower Bollinger"
        assert bb.bandwidth().name == "Bandwidth Bollinger"
        assert bb.percent_b().name == "%B Bollinger"
    expected_middle = init([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    expected_upper = init([np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    expected_lower = init([np.nan, np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    expected_bandwidth = 4 / expected_middle
    expected_percent_b = init([np.nan, np.nan] + [3 / 4] * 8)
    np.testing.assert_array_equal(bb.middle(), expected_middle)
    np.testing.assert_array_equal(bb.upper(), expected_upper)
    np.testing.assert_array_equal(bb.lower(), expected_lower)
    np.testing.assert_array_equal(bb.bandwidth(), expected_bandwidth)
    np.testing.assert_array_equal(bb.percent_b(), expected_percent_b)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_ewm_bollinger_bands(init, dtype, expected_ewmstd):
    ts = init([1, 2, 3, 4])
    bb = EwmBollingerBands(ts, window=3, multiplier=2)
    assert isinstance(bb.middle(), dtype)
    assert isinstance(bb.upper(), dtype)
    assert isinstance(bb.lower(), dtype)
    assert isinstance(bb.bandwidth(), dtype)
    assert isinstance(bb.percent_b(), dtype)
    if dtype == pd.Series:
        assert bb.middle().name == "Middle EwmBollinger"
        assert bb.upper().name == "Upper EwmBollinger"
        assert bb.lower().name == "Lower EwmBollinger"
        assert bb.bandwidth().name == "Bandwidth EwmBollinger"
        assert bb.percent_b().name == "%B EwmBollinger"
    expected_middle = init([1.0, 2.5 / 1.5, 4.25 / 1.75, 6.125 / 1.875])
    expected_ewmstd = init(expected_ewmstd)
    expected_upper = expected_middle + 2 * expected_ewmstd
    expected_lower = expected_middle - 2 * expected_ewmstd
    expected_bandwidth = 4 * expected_ewmstd / expected_middle
    expected_percent_b = (ts - expected_lower) / (4 * expected_ewmstd)
    np.testing.assert_array_equal(bb.middle(), expected_middle)
    np.testing.assert_allclose(bb.upper(), expected_upper, rtol=1e-9)
    np.testing.assert_allclose(bb.lower(), expected_lower, rtol=1e-9)
    np.testing.assert_allclose(bb.bandwidth(), expected_bandwidth, rtol=1e-9)
    np.testing.assert_allclose(bb.percent_b(), expected_percent_b, rtol=1e-9)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_keltner_bands(init, dtype):
    ts = init([1, 2, 3, 4])
    kb = KeltnerBands(high=ts - 0.5, low=ts - 1.5, close=ts, window_atr=3, window=3, multiplier=2)
    assert isinstance(kb.middle(), dtype)
    assert isinstance(kb.upper(), dtype)
    assert isinstance(kb.lower(), dtype)
    assert isinstance(kb.bandwidth(), dtype)
    assert isinstance(kb.percent_b(), dtype)
    if dtype == pd.Series:
        assert kb.middle().name == "Middle Keltner"
        assert kb.upper().name == "Upper Keltner"
        assert kb.lower().name == "Lower Keltner"
        assert kb.bandwidth().name == "Bandwidth Keltner"
        assert kb.percent_b().name == "%B Keltner"
    expected_middle = init([1.0, 2.5 / 1.5, 4.25 / 1.75, 6.125 / 1.875])
    expected_upper = expected_middle + 2
    expected_lower = expected_middle - 2
    expected_bandwidth = 4 / expected_middle
    expected_percent_b = (ts - expected_lower) / 4
    np.testing.assert_array_equal(kb.middle(), expected_middle)
    np.testing.assert_array_equal(kb.upper(), expected_upper)
    np.testing.assert_array_equal(kb.lower(), expected_lower)
    np.testing.assert_array_equal(kb.bandwidth(), expected_bandwidth)
    np.testing.assert_array_equal(kb.percent_b(), expected_percent_b)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_donchian_bands(init, dtype):
    ts = init([1, 2, 3, 4])
    db = DonchianBands(ts, window=3)
    assert isinstance(db.upper(), dtype)
    assert isinstance(db.lower(), dtype)
    assert isinstance(db.middle(), dtype)
    assert isinstance(db.bandwidth(), dtype)
    assert isinstance(db.percent_b(), dtype)
    if dtype == pd.Series:
        assert db.middle().name == "Middle Donchian"
        assert db.upper().name == "Upper Donchian"
        assert db.lower().name == "Lower Donchian"
        assert db.bandwidth().name == "Bandwidth Donchian"
        assert db.percent_b().name == "%B Donchian"
    expected_middle = init([np.nan, np.nan, 2.0, 3.0])
    expected_upper = init([np.nan, np.nan, 3.0, 4.0])
    expected_lower = init([np.nan, np.nan, 1.0, 2.0])
    expected_bandwidth = 2 / expected_middle
    expected_percent_b = (ts - expected_lower) / 2
    np.testing.assert_array_equal(db.middle(), expected_middle)
    np.testing.assert_array_equal(db.upper(), expected_upper)
    np.testing.assert_array_equal(db.lower(), expected_lower)
    np.testing.assert_array_equal(db.bandwidth(), expected_bandwidth)
    np.testing.assert_array_equal(db.percent_b(), expected_percent_b)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_macd(init, dtype):
    ts = init([1, 2, 3, 4])
    m = macd(ts, fast=1, slow=3)
    assert isinstance(m, dtype)
    expected = ts - init([1.0, 2.5 / 1.5, 4.25 / 1.75, 6.125 / 1.875])
    np.testing.assert_array_equal(m, expected)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_macd_signal(init, dtype):
    ts = init([1, 2, 3, 4])
    ms = macd_signal(ts, fast=1, slow=3, signal=2)
    assert isinstance(ms, dtype)
    expected = ts - init([1.0, 2.5 / 1.5, 4.25 / 1.75, 6.125 / 1.875])
    expected = expected - init(pd.Series(expected).ewm(span=2).mean())
    np.testing.assert_array_equal(ms, expected)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_rsi(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    r = rsi(ts, period=3)
    assert isinstance(r, dtype)
    expected = init([np.nan, np.nan, 100, 100, 100, 100, 100, 100, 100, 100])
    np.testing.assert_array_equal(r, expected)
    ts = init([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    r = rsi(ts, period=3)
    assert isinstance(r, dtype)
    expected = init([np.nan, np.nan, 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(r, expected)


@pytest.mark.parametrize("init, dtype", [(pd.Series, pd.Series), (np.array, np.ndarray)])
def test_ewm_rsi(init, dtype):
    ts = init([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    r = ewm_rsi(ts, period=3)
    assert isinstance(r, dtype)
    expected = init([np.nan, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    np.testing.assert_array_equal(r, expected)
    ts = init([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    r = ewm_rsi(ts, period=3)
    assert isinstance(r, dtype)
    expected = init([np.nan, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(r, expected)
