import numpy as np
import pandas as pd

from quantfinlib.feature.micro_structure._sequential_trade_models import estimate_buy_volume, get_vpin


def test_estimate_buy_volume_non_informative():
    price = pd.Series([100, 101, 102, 103, 104])
    volume = pd.Series([100, 200, 300, 400, 500])
    sigma_p = pd.Series(np.zeros_like(price))
    buy_volume = estimate_buy_volume(price, volume, sigma_p)
    assert np.allclose((buy_volume - volume).dropna() / 2, 0)


def test_estimate_buy_volume_general():
    price = pd.Series([100, 101, 102, 103, 104])
    volume = pd.Series([100, 200, 300, 400, 500])
    sigma_p = pd.Series([1, 1, 1, 1, 1])
    buy_volume = estimate_buy_volume(price, volume, sigma_p)
    assert len(buy_volume) == len(price)
    assert np.all(buy_volume[np.isfinite(buy_volume)] >= 0)

    buy_volume_np = estimate_buy_volume(price.values, volume.values, sigma_p.values)
    assert len(buy_volume_np) == len(price)
    assert np.all(buy_volume_np[np.isfinite(buy_volume_np)] >= 0)
    assert np.allclose(buy_volume.dropna() - buy_volume_np[np.isfinite(buy_volume_np)], 0)


def test_get_vpin_non_informative():
    volume = pd.Series([100, 200, 300, 400, 500])
    buy_volume = volume / 2
    window = 3
    vpin = get_vpin(volume, buy_volume, window)
    assert np.allclose(vpin.dropna(), 0.0)


def test_get_vpin_general():
    volume = pd.Series(np.random.randint(100, 1000, 100))
    buy_volume = pd.Series(np.random.randint(0, 1, 100)) * volume
    window = 10
    vpin = get_vpin(volume, buy_volume, window)
    assert len(vpin) == len(volume)
    assert np.all(vpin[np.isfinite(vpin)] >= 0)
    assert np.all(vpin[np.isfinite(vpin)] <= 1)
    vpin_np = get_vpin(volume.values, buy_volume.values, window)
    assert len(vpin_np) == len(volume)
    assert np.all(vpin_np[np.isfinite(vpin_np)] >= 0)
    assert np.all(vpin_np[np.isfinite(vpin_np)] <= 1)
    assert np.allclose(vpin.dropna(), vpin_np[np.isfinite(vpin_np)])
