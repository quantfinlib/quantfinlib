import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim import BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck, QuasiRandom


def tech_ind_testfunc(prices: np.ndarray) -> np.ndarray:
    return np.hstack([-np.diff(prices), np.nan])


def test_QuasiRandom_init():
    b = QuasiRandom(tech_ind_func=tech_ind_testfunc)
    assert str(b.base_model) == str(BrownianMotion())
    assert b.tech_ind_func == tech_ind_testfunc
    assert b.f_signal_vol == 0.1
    assert b.base_model.drift == 0.0
    assert b.base_model.vol == 0.1

    b = QuasiRandom(tech_ind_func=tech_ind_testfunc, base_model=BrownianMotion(drift=0.05))
    assert b.base_model.drift == 0.05

    b = QuasiRandom(tech_ind_func=tech_ind_testfunc, base_model=BrownianMotion(drift=0.05, vol=0.2))
    assert b.base_model.drift == 0.05
    assert b.base_model.vol == 0.2

    base_model = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])
    b = QuasiRandom(tech_ind_testfunc, base_model=base_model)
    assert np.all(b.base_model.L_ == np.array([[1, 0], [0.4, np.sqrt(1 - 0.4**2)]]))

    base_model = GeometricBrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])
    b = QuasiRandom(tech_ind_testfunc, base_model=base_model)
    assert np.all(b.base_model.L_ == np.array([[1, 0], [0.4, np.sqrt(1 - 0.4**2)]]))
    assert np.all(b.base_model.drift == np.array([0.05, 0.05]))
    assert np.all(b.base_model.vol == np.array([0.1, 0.1]))

    base_model = OrnsteinUhlenbeck(mean=[0.05, 0.05], mrr=[0.1, 0.1], vol=[0.1, 0.1])
    b = QuasiRandom(tech_ind_testfunc, base_model=base_model)
    assert np.all(b.base_model.mean == np.array([0.05, 0.05]))
    assert np.all(b.base_model.mrr == np.array([0.1, 0.1]))
    assert np.all(b.base_model.vol == np.array([0.1, 0.1]))


def test_QuasiRandom_repr():
    b = QuasiRandom(tech_ind_testfunc)
    assert str(b) == f"QuasiRandom(tech_ind_func={tech_ind_testfunc}, f_signal_vol=0.1, base_model={BrownianMotion()})"


def test_QuasiRandom_fit_error():
    b = QuasiRandom(tech_ind_testfunc)
    x = np.ones(shape=(5, 1))
    with pytest.raises(NotImplementedError):
        b.fit(x, 0.1)


def test_QuasiRandom_path_sample():
    base_model = BrownianMotion(drift=0.0, vol=0.1)
    qb = QuasiRandom(tech_ind_testfunc, f_signal_vol=1.0, base_model=base_model)
    p = qb.path_sample(x0=1, dt=1 / 12, num_steps=1200, num_paths=1, random_state=42)
    assert p.shape == (1201, 1)
    assert_allclose(p.T, np.ones((1, 1201)), atol=0.1)


def test_QuasiRandom_path_sample_mp():
    base_model = BrownianMotion(drift=0.0, vol=0.1)
    b = QuasiRandom(tech_ind_testfunc, f_signal_vol=1.0, base_model=base_model)
    p = b.path_sample(x0=[1, 2, 3], dt=1 / 12, num_steps=1200, num_paths=3, random_state=42)
    assert p.shape == (1201, 3)
    for i in range(3):
        assert_allclose(p[:, i], i + np.ones(1201), atol=0.1)


def test_QuasiRandom_path_sample_mv():
    base_model = BrownianMotion(drift=[0.0, 0.0], vol=[0.01, 0.1], cor=[[1, 0.4], [0.4, 1]])
    b = QuasiRandom(tech_ind_testfunc, f_signal_vol=1.0, base_model=base_model)
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=1200, num_paths=1, random_state=42)
    assert p.shape == (1201, 2)
    assert_allclose(p[:, 0], np.ones(1201), atol=0.01)
    assert_allclose(p[:, 1], np.ones(1201), atol=0.1)
    realized_vol = np.sqrt(12) * np.std(np.diff(p, axis=0), axis=0)
    assert_allclose(realized_vol, [0.0, 0.0], atol=0.01)


def test_QuasiRandom_path_sample_mv_mp():
    base_model = BrownianMotion(drift=[0.0, 0.0], vol=[0.01, 0.1], cor=[[1, 0.4], [0.4, 1]])
    b = QuasiRandom(tech_ind_testfunc, f_signal_vol=1.0, base_model=base_model)
    p = b.path_sample(x0=[1, 1], dt=1 / 12, num_steps=1200, num_paths=3, random_state=42)
    assert p.shape == (1201, 6)
    for i in range(3):
        assert_allclose(p[:, 2 * i], np.ones(1201), atol=0.01)
        assert_allclose(p[:, 2 * i + 1], np.ones(1201), atol=0.1)
    realized_vol = np.sqrt(12) * np.std(np.diff(p, axis=0), axis=0)
    assert_allclose(realized_vol, [0.0] * 6, atol=0.01)


def test_QuasiRandom_path_sample_pd():
    base_model = BrownianMotion(drift=1.0, vol=0.01)
    b = QuasiRandom(tech_ind_testfunc, base_model=base_model)
    p = b.path_sample(x0=1, dt=1 / 12, num_steps=12, num_paths=1, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (13,)
    assert isinstance(p, pd.Series)
    assert isinstance(p.index, pd.DatetimeIndex)
    expected = np.linspace(1, 2, 13)
    assert_allclose(p.values, expected, atol=0.1)


def test_QuasiRandom_path_sample_pd_mp():
    base_model = BrownianMotion(drift=0.0, vol=0.1)
    b = QuasiRandom(tech_ind_testfunc, f_signal_vol=1.0, base_model=base_model)
    p = b.path_sample(x0=[1, 2, 3], dt=1 / 12, num_steps=12, num_paths=3, random_state=42,
                      label_start='2020-01-01', label_freq='B')
    assert p.shape == (13, 3)
    assert isinstance(p, pd.DataFrame)
    assert isinstance(p.index, pd.DatetimeIndex)
    assert p.columns.tolist() == [f"S_{i}" for i in range(3)]
    for i in range(3):
        assert_allclose(p[f"S_{i}"], np.ones(13) + i, atol=0.1)


def test_QuasiRandom_path_sample_pd_mv():
    base_model = BrownianMotion(drift=[0.0, 0.0], vol=[0.01, 0.1], cor=[[1, 0.4], [0.4, 1]])
    b = QuasiRandom(tech_ind_testfunc, f_signal_vol=1.0, base_model=base_model)
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


def test_QuasiRandom_path_sample_pd_mv_mp():
    base_model = BrownianMotion(drift=[0.0, 0.0], vol=[0.01, 0.1], cor=[[1, 0.4], [0.4, 1]])
    b = QuasiRandom(tech_ind_testfunc, f_signal_vol=1.0, base_model=base_model)
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
