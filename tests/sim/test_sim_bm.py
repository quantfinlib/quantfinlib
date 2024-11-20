import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim import BrownianMotion


def test_BrownianMotion_init():
    b = BrownianMotion()
    assert b.drift == 0.0
    assert b.vol == 0.1

    b = BrownianMotion(0.05)
    assert b.drift == 0.05

    b = BrownianMotion(drift=0.05)
    assert b.drift == 0.05

    b = BrownianMotion(0.05, 0.2)
    assert b.drift == 0.05
    assert b.vol == 0.2

    b = BrownianMotion(drift=0.05, vol=0.2)
    assert b.drift == 0.05
    assert b.vol == 0.2

    b = BrownianMotion(vol=0.2)
    assert b.vol == 0.2

    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])
    assert np.all(b.L_ == np.array([[1, 0], [0.4, np.sqrt(1 - 0.4**2)]]))


def test_BrownianMotion_fit():
    b = BrownianMotion()
    x = 3.14 * np.ones(shape=(5, 1))
    b.fit(x, 0.1)
    assert_allclose(b.drift, np.array([0.0]).reshape(1, -1))
    assert_allclose(b.vol, np.array([0.0]).reshape(1, -1))


def test_BrownianMotion_fit_mv_err():
    b = BrownianMotion()
    x = 3.14 * np.ones(shape=(5, 2))
    with pytest.raises(ValueError):
        b.fit(x, 0.1)


def test_BrownianMotion_fit_mv_ok():
    b = BrownianMotion()
    x = 3.14 * np.random.normal(size=(5, 2))
    b.fit(x, 0.1)
    assert b.L_ is not None


def test_BrownianMotion_path_sample():
    b = BrownianMotion(drift=1.0, vol=0.01)
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=12, num_paths=1, random_state=42)
    assert p.shape == (13, 1)
    expected = np.linspace(1, 2, 13).reshape(-1, 1)
    assert_allclose(p, expected, atol=0.1)


def test_BrownianMotion_path_sample_mp():
    b = BrownianMotion(drift=1.0, vol=0.01)
    p = b.path_sample(x0=[1, 2, 3], dt=1 / 12, num_steps=12, num_paths=3, random_state=42)
    assert p.shape == (13, 3)
    expected = np.linspace(1, 2, 13)
    for i in range(3):
        assert_allclose(p[:, i], expected + i, atol=0.1)


def test_BrownianMotion_path_sample_mv():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.01, 0.1], cor=[[1, 0.4], [0.4, 1]])
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=12, num_paths=1, random_state=42)
    assert p.shape == (13, 2)
    expected = np.linspace(1, 1.05, 13)
    assert_allclose(p[:, 0], expected, atol=0.1)
    assert_allclose(p[:, 1], expected, atol=0.1)
    dedrifted = p - expected.reshape(-1, 1)
    realized_vol = np.sqrt(12) * np.std(np.diff(dedrifted, axis=0), axis=0)
    assert_allclose(realized_vol, [0.01, 0.1], rtol=0.3)


def test_BrownianMotion_path_sample_mv_mp():
    b = BrownianMotion(drift=[0.005, 0.005], vol=[0.001, 0.01], cor=[[1, 0.4], [0.4, 1]])
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=120, num_paths=3, random_state=42)
    assert p.shape == (121, 6)
    expected = np.linspace(1, 1.05, 121)
    for i in range(3):
        assert_allclose(p[:, 2 * i], expected, atol=0.01)
        assert_allclose(p[:, 2 * i + 1], expected, atol=0.1)
    dedrifted = p - expected.reshape(-1, 1)
    realized_vol = np.sqrt(12) * np.std(np.diff(dedrifted, axis=0), axis=0)
    assert_allclose(realized_vol, [0.001, 0.01] * 3, rtol=0.3)


def test_BrownianMotion_path_sample_pd():
    b = BrownianMotion(drift=1.0, vol=0.01)
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=12, num_paths=1, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (13,)
    assert isinstance(p, pd.Series)
    assert isinstance(p.index, pd.DatetimeIndex)
    expected = np.linspace(1, 2, 13)
    assert_allclose(p.values, expected, atol=0.1)


def test_BrownianMotion_path_sample_pd_mp():
    b = BrownianMotion(drift=1.0, vol=0.01)
    p = b.path_sample(x0=[1, 2, 3], dt=1 / 12, num_steps=12, num_paths=3, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (13, 3)
    assert isinstance(p, pd.DataFrame)
    assert isinstance(p.index, pd.DatetimeIndex)
    assert p.columns.tolist() == [f"S_{i}" for i in range(3)]
    expected = np.linspace(1, 2, 13)
    for i in range(3):
        assert_allclose(p[f"S_{i}"], expected + i, atol=0.1)


def test_BrownianMotion_path_sample_pd_mv():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.01, 0.1], cor=[[1, 0.4], [0.4, 1]])
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=12, num_paths=1, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (13, 2)
    assert isinstance(p, pd.DataFrame)
    assert isinstance(p.index, pd.DatetimeIndex)
    assert p.columns.tolist() == ["S0", "S1"]
    expected = np.linspace(1, 1.05, 13)
    assert_allclose(p["S0"], expected, atol=0.1)
    assert_allclose(p["S1"], expected, atol=0.1)
    dedrifted = p - expected.reshape(-1, 1)
    realized_vol = np.sqrt(12) * dedrifted.diff().std()
    assert_allclose(realized_vol, [0.01, 0.1], rtol=0.3)


def test_BrownianMotion_path_sample_pd_mv_mp():
    b = BrownianMotion(drift=[0.005, 0.005], vol=[0.001, 0.01], cor=[[1, 0.4], [0.4, 1]])
    p = b.path_sample(x0=[1, 2], dt=1 / 12, num_steps=120, num_paths=3, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (121, 6)
    assert isinstance(p, pd.DataFrame)
    assert isinstance(p.index, pd.DatetimeIndex)
    assert p.columns.tolist() == [f"S{j}_{i}" for i in range(3) for j in range(2)]
    expected = np.linspace(1, 1.05, 121)
    for i in range(3):
        assert_allclose(p[f"S0_{i}"], expected, atol=0.01)
        assert_allclose(p[f"S1_{i}"], expected + 1, atol=0.1)
    dedrifted = p - expected.reshape(-1, 1)
    realized_vol = np.sqrt(12) * dedrifted.diff().std()
    assert_allclose(realized_vol, [0.001, 0.01] * 3, rtol=0.3)


def test_BrownianMotion_nll_1d():
    b = BrownianMotion(drift=0.05, vol=0.1)
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=10, num_paths=1)
    _ = b.nll(p)


def test_BrownianMotion_nll_2d():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=10, num_paths=1)
    _ = b.nll(p)


def test_BrownianMotion_aic_1d():
    b = BrownianMotion(drift=0.05, vol=0.1)
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=10, num_paths=1)
    _ = b.aic(p)


def test_BrownianMotion_aic_2d():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=10, num_paths=1)
    _ = b.aic(p)


def test_BrownianMotion_bic_1d():
    b = BrownianMotion(drift=0.05, vol=0.1)
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=10, num_paths=1)
    _ = b.bic(p)


def test_BrownianMotion_bic_2d():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=10, num_paths=1)
    _ = b.bic(p)
