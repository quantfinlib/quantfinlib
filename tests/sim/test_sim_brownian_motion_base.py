import pandas as pd
import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim._bm import BrownianMotionBase

def test_BrownianMotionBase_init():
    b = BrownianMotionBase()

    b = BrownianMotionBase(0.05)
    assert b.drift == 0.05

    b = BrownianMotionBase(drift=0.05)    
    assert b.drift == 0.05

    b = BrownianMotionBase(0.05, 0.1)
    assert b.drift == 0.05
    assert b.vol == 0.1

    b = BrownianMotionBase(drift=0.05, vol=0.1)    
    assert b.drift == 0.05
    assert b.vol == 0.1

    b = BrownianMotionBase(vol=0.1)    
    assert b.vol == 0.1


def test_BrownianMotionBase_init_mv():
    b = BrownianMotionBase([0.05, 0.05], [0.1, 0.1])
    assert_allclose(b.drift, np.array([0.05, 0.05]).reshape(1, -1))
    assert b.L_ is None


def test_BrownianMotionBase_init_cor():
    b = BrownianMotionBase([0.05, 0.05], [0.1, 0.1], [[1.0, 0.8], [0.8, 1.0]])
    assert_allclose(b.drift, np.array([0.05, 0.05]).reshape(1, -1))
    assert b.L_ is not None


def test_BrownianMotionBase_fit():
    b = BrownianMotionBase()
    x = 3.14 * np.ones(shape=(5, 1))
    b.fit(x, 0.1)
    assert_allclose(b.drift, np.array([0.0]).reshape(1, -1))
    assert_allclose(b.vol, np.array([0.0]).reshape(1, -1))

def test_BrownianMotionBase_fit_with_drift():
    b = BrownianMotionBase()
    x = np.arange(10)
    b.fit(x, 0.25)
    assert_allclose(b.drift, np.array([4.0]).reshape(1, -1))
    assert_allclose(b.vol, np.array([0.0]).reshape(1, -1))


def test_BrownianMotionBase_fit_mv():
    b = BrownianMotionBase()
    x = np.random.normal(size=(10, 3))
    b.fit(x, 0.25)
    assert b.cor is not None
    assert b.cor.shape[0] == 3
    assert b.cor.shape[1] == 3

    assert b.L_ is not None
    assert b.L_.shape[0] == 3
    assert b.L_.shape[1] == 3

def test_BrownianMotionBase_path_sample_1p():
    b = BrownianMotionBase()
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    assert p.shape[0] == 11
    assert p.shape[1] == 1
    
def test_BrownianMotionBase_path_sample_3p():
    b = BrownianMotionBase()
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=3)
    assert p.shape[0] == 11
    assert p.shape[1] == 3

def test_BrownianMotionBase_path_sample_random_state():
    b = BrownianMotionBase()
    p0 = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=3)
    p1 = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=3, random_state=42)
    p2 = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=3, random_state=42)

    assert p0[1,0] != p1[1,0] 

    assert_allclose(p1, p2)


def test_BrownianMotionBase_path_sample_2d_cor():
    b = BrownianMotionBase(drift=[0.1, 0.2], vol= [0.2, 0.2], cor=[[1.0, 0.3], [0.3, 1.0]])
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    assert b.L_ is not None
    