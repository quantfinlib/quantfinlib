import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim import GeometricBrownianMotion


def test_GeometricBrownianMotion_init():
    b = GeometricBrownianMotion()

    b = GeometricBrownianMotion(0.05)
    assert b.drift == 0.05

    b = GeometricBrownianMotion(drift=0.05)    
    assert b.drift == 0.05

    b = GeometricBrownianMotion(0.05, 0.1)
    assert b.drift == 0.05
    assert b.vol == 0.1

    b = GeometricBrownianMotion(drift=0.05, vol=0.1)    
    assert b.drift == 0.05
    assert b.vol == 0.1

    b = GeometricBrownianMotion(vol=0.1)    
    assert b.vol == 0.1

    b = GeometricBrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    assert b.L_ is not None


def test_GeometricBrownianMotion_fit():
    b = GeometricBrownianMotion()
    x = 3.14 * np.ones(shape=(5, 1))
    b.fit(x, 0.1)
    assert_allclose(b.drift, np.array([0.0]).reshape(1, -1))
    assert_allclose(b.vol, np.array([0.0]).reshape(1, -1))

def test_GeometricBrownianMotion_fit_mv_err():
    b = GeometricBrownianMotion()
    x = 3.14 * np.ones(shape=(5, 2))
    with pytest.raises(ValueError):
        b.fit(x, 0.1)

def test_GeometricBrownianMotion_fit_mv_ok():
    b = GeometricBrownianMotion()
    x = np.exp(3.14 * np.random.normal(size=(5, 2)))
    b.fit(x, 0.1)
    assert b.L_ is not None

def test_GeometricBrownianMotion_path_sample():
    b = GeometricBrownianMotion()
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    assert p.shape[0] == 11
    assert p.shape[1] == 1

def test_GeometricBrownianMotion_path_sample_mv():
    b = GeometricBrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    p = b.path_sample(x0=[1, 1], dt=1/12, num_steps=10, num_paths=1)
    assert p.shape[0] == 11
    assert p.shape[1] == 2

def test_GeometricBrownianMotion_nll_1d():
    b = GeometricBrownianMotion(drift=0.05, vol=0.1)    
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    nll = b.nll(p, None)

def test_GeometricBrownianMotion_nll_2d():
    b = GeometricBrownianMotion(drift=[0.05, 0.05], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    p = b.path_sample(x0=[1, 1], dt=1/12, num_steps=10, num_paths=1)
    nll = b.nll(p)


def test_GeometricBrownianMotion_repr():
    b = GeometricBrownianMotion()
    assert str(b) == f"GeometricBrownianMotion(drift=[[0.]], vol=[[0.1]], cor=None)"
    b = GeometricBrownianMotion(0.05)
    assert str(b) == f"GeometricBrownianMotion(drift=[[0.05]], vol=[[0.1]], cor=None)"

