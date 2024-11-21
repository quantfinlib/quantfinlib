import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim import QuasiBrownianMotion


def tech_ind_testfunc(prices: np.ndarray) -> np.ndarray:
    return np.hstack([-np.diff(prices), np.nan])


def test_QuasiBrownianMotion_init():
    b = QuasiBrownianMotion(tech_ind_testfunc)
    assert b.drift == 0.0
    assert b.vol == 0.1
    assert b.f_signal_vol == 0.1
    assert b.tech_ind_func == tech_ind_testfunc

    b = QuasiBrownianMotion(tech_ind_testfunc, 0.05)
    assert b.drift == 0.05

    b = QuasiBrownianMotion(tech_ind_testfunc, drift=0.05)
    assert b.drift == 0.05

    b = QuasiBrownianMotion(tech_ind_testfunc, 0.05, 0.2)
    assert b.drift == 0.05
    assert b.vol == 0.2

    b = QuasiBrownianMotion(tech_ind_testfunc, drift=0.05, vol=0.2)
    assert b.drift == 0.05
    assert b.vol == 0.2

    b = QuasiBrownianMotion(tech_ind_testfunc, vol=0.2)
    assert b.vol == 0.2

    b = QuasiBrownianMotion(tech_ind_testfunc, drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])
    assert np.all(b.L_ == np.array([[1, 0], [0.4, np.sqrt(1 - 0.4**2)]]))


def test_QuasiBrownianMotion_fit_error():
    b = QuasiBrownianMotion(tech_ind_testfunc)
    x = np.ones(shape=(5, 1))
    with pytest.raises(NotImplementedError):
        b.fit(x, 0.1)


def test_QuasiBrownianMotion_path_sample():
    qb = QuasiBrownianMotion(tech_ind_testfunc, drift=0.0, vol=0.1, f_signal_vol=1.0)
    p = qb.path_sample(x0=1, dt=1 / 12, num_steps=1200, num_paths=1, random_state=42)
    assert p.shape == (1201, 1)
    assert_allclose(p.T, np.ones((1, 1201)), atol=0.1)


def test_QuasiBrownianMotion_path_sample_mp():
    b = QuasiBrownianMotion(tech_ind_testfunc, drift=0.0, vol=0.1, f_signal_vol=1.0)
    p = b.path_sample(x0=[1, 2, 3], dt=1 / 12, num_steps=1200, num_paths=3, random_state=42)
    assert p.shape == (1201, 3)
    for i in range(3):
        assert_allclose(p[:, i], i + np.ones(1201), atol=0.1)


def test_QuasiBrownianMotion_path_sample_mv():
    b = QuasiBrownianMotion(tech_ind_testfunc, drift=[0.0, 0.0], vol=[0.01, 0.1],
                            cor=[[1, 0.4], [0.4, 1]], f_signal_vol=1.0)
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=1200, num_paths=1, random_state=42)
    assert p.shape == (1201, 2)
    assert_allclose(p[:, 0], np.ones(1201), atol=0.01)
    assert_allclose(p[:, 1], np.ones(1201), atol=0.1)
    realized_vol = np.sqrt(12) * np.std(np.diff(p, axis=0), axis=0)
    assert_allclose(realized_vol, [0.0, 0.0], atol=0.01)


def test_QuasiBrownianMotion_path_sample_mv_mp():
    b = QuasiBrownianMotion(tech_ind_testfunc, drift=[0.0, 0.0], vol=[0.001, 0.01],
                            cor=[[1, 0.4], [0.4, 1]], f_signal_vol=1.0)
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=1200, num_paths=3, random_state=42)
    assert p.shape == (1201, 6)
    for i in range(3):
        assert_allclose(p[:, 2 * i], np.ones(1201), atol=0.01)
        assert_allclose(p[:, 2 * i + 1], np.ones(1201), atol=0.1)
    realized_vol = np.sqrt(12) * np.std(np.diff(p, axis=0), axis=0)
    assert_allclose(realized_vol, [0.0] * 6, atol=0.01)


def test_QuasiBrownianMotion_path_sample_pd():
    b = QuasiBrownianMotion(tech_ind_testfunc, drift=1.0, vol=0.01)
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=12, num_paths=1, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (13,)
    assert isinstance(p, pd.Series)
    assert isinstance(p.index, pd.DatetimeIndex)
    expected = np.linspace(1, 2, 13)
    assert_allclose(p.values, expected, atol=0.1)


def test_QuasiBrownianMotion_path_sample_pd_mp():
    b = QuasiBrownianMotion(tech_ind_testfunc, drift=0.0, vol=0.1, f_signal_vol=1.0)
    p = b.path_sample(x0=[1, 2, 3], dt=1 / 12, num_steps=12, num_paths=3, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (13, 3)
    assert isinstance(p, pd.DataFrame)
    assert isinstance(p.index, pd.DatetimeIndex)
    assert p.columns.tolist() == [f"S_{i}" for i in range(3)]
    for i in range(3):
        assert_allclose(p[f"S_{i}"], np.ones(13) + i, atol=0.1)


def test_QuasiBrownianMotion_path_sample_pd_mv():
    b = QuasiBrownianMotion(tech_ind_testfunc, drift=[0.0, 0.0], vol=[0.01, 0.1],
                            cor=[[1, 0.4], [0.4, 1]], f_signal_vol=1.0)
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=12, num_paths=1, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (13, 2)
    assert isinstance(p, pd.DataFrame)
    assert isinstance(p.index, pd.DatetimeIndex)
    assert p.columns.tolist() == ["S0", "S1"]
    assert_allclose(p["S0"], np.ones(13), atol=0.01)
    assert_allclose(p["S1"], np.ones(13), atol=0.1)
    realized_vol = np.sqrt(12) * p.diff().std()
    assert_allclose(realized_vol, [0.0, 0.0], atol=0.05)


def test_QuasiBrownianMotion_path_sample_pd_mv_mp():
    b = QuasiBrownianMotion(tech_ind_testfunc, drift=[0.0, 0.0], vol=[0.01, 0.1],
                            cor=[[1, 0.4], [0.4, 1]], f_signal_vol=1.0)
    p = b.path_sample(x0=[1, 2], dt=1 / 12, num_steps=1200, num_paths=3, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (1201, 6)
    assert isinstance(p, pd.DataFrame)
    assert isinstance(p.index, pd.DatetimeIndex)
    assert p.columns.tolist() == [f"S{j}_{i}" for i in range(3) for j in range(2)]
    for i in range(3):
        assert_allclose(p[f"S0_{i}"], np.ones(1201), atol=0.01)
        assert_allclose(p[f"S1_{i}"], np.ones(1201) + 1, atol=0.1)
    realized_vol = np.sqrt(12) * p.diff().std()
    assert_allclose(realized_vol, [0.0] * 6, atol=0.01)
