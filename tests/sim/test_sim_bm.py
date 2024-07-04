import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim import BrownianMotion


def test_BrownianMotion_init():
    b = BrownianMotion()

    b = BrownianMotion(0.05)
    assert b.drift == 0.05

    b = BrownianMotion(drift=0.05)    
    assert b.drift == 0.05

    b = BrownianMotion(0.05, 0.1)
    assert b.drift == 0.05
    assert b.vol == 0.1

    b = BrownianMotion(drift=0.05, vol=0.1)    
    assert b.drift == 0.05
    assert b.vol == 0.1

    b = BrownianMotion(vol=0.1)    
    assert b.vol == 0.1

    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    assert b.L_ is not None


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
    b = BrownianMotion()
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    assert p.shape[0] == 11
    assert p.shape[1] == 1

def test_BrownianMotion_path_sample_mv():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    p = b.path_sample(x0=[1, 1], dt=1/12, num_steps=10, num_paths=1)
    assert p.shape[0] == 11
    assert p.shape[1] == 2

def test_BrownianMotion_nll_1d():
    b = BrownianMotion(drift=0.05, vol=0.1)    
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    nll = b.nll(p)

def test_BrownianMotion_nll_2d():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    p = b.path_sample(x0=[1, 1], dt=1/12, num_steps=10, num_paths=1)
    nll = b.nll(p)

def test_BrownianMotion_aic_1d():
    b = BrownianMotion(drift=0.05, vol=0.1)    
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    nll = b.aic(p)

def test_BrownianMotion_aic_2d():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    p = b.path_sample(x0=[1, 1], dt=1/12, num_steps=10, num_paths=1)
    nll = b.aic(p)

def test_BrownianMotion_bic_1d():
    b = BrownianMotion(drift=0.05, vol=0.1)    
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    nll = b.bic(p)

def test_BrownianMotion_bic_2d():
    b = BrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    p = b.path_sample(x0=[1, 1], dt=1/12, num_steps=10, num_paths=1)
    nll = b.bic(p)

