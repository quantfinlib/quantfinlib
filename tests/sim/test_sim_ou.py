import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim import OrnsteinUhlenbeck


#  mean=0.0, mrr=1.0, vol=0.1, cor=None
def test_OrnsteinUhlenbeck_init():
    b = OrnsteinUhlenbeck()

    b = OrnsteinUhlenbeck(mean=0.0, mrr=1.0, vol=0.1, cor=None)
    assert b.mean == 0.0
    assert b.mrr == 1.0
    assert b.vol == 0.1
    assert b.cor == None
    assert b.L_ is None

    b = OrnsteinUhlenbeck(mean=[0.0, 0.0], mrr=[1.0, 1.0], vol=[0.1, 0.1], cor=[[1.0, 0.4], [0.4, 1.0]])
    assert_allclose(b.mean, np.array([0.0, 0.0]).reshape(1, -1))
    assert_allclose(b.mrr, np.array([1.0, 1.0]).reshape(1, -1))
    assert b.cor is not None
    assert b.L_ is not None


def test_OrnsteinUhlenbeck_fit():
    b = OrnsteinUhlenbeck()
    x = 3.14 * np.ones(shape=(5, 1))
    b.fit(x, 0.1)
    assert_allclose(b.mean, np.array([3.14]).reshape(1, -1))
    assert_allclose(b.vol, np.array([0.0]).reshape(1, -1))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_OrnsteinUhlenbeck_fit_mv_ok():
    b = OrnsteinUhlenbeck()
    x = 3.14 * np.random.normal(size=(5, 2))
    b.fit(x, 0.1)
    assert b.L_ is not None

def test_OrnsteinUhlenbeck_path_sample():
    b = OrnsteinUhlenbeck()
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    assert p.shape[0] == 11
    assert p.shape[1] == 1

def test_OrnsteinUhlenbeck_path_sample_mv():
    b = OrnsteinUhlenbeck(mean=[0.05, 0.05], mrr=[1.0, 1.0], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    p = b.path_sample(x0=[1, 1], dt=1/12, num_steps=10, num_paths=1)
    assert p.shape[0] == 11
    assert p.shape[1] == 2

def test_OrnsteinUhlenbeck_nll_1d():
    b = OrnsteinUhlenbeck(mean=0.0, mrr=1.0, vol=0.1)
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    nll = b.nll(p, None)

def test_OrnsteinUhlenbeck_nll_2d():
    b = OrnsteinUhlenbeck(mean=[0.05, 0.05], mrr=[1.0, 1.0], vol=[0.1, 0.1], cor=[[1, 0.4], [0.4, 1]])    
    p = b.path_sample(x0=[1, 1], dt=1/12, num_steps=10, num_paths=1)
    nll = b.nll(p)


def test_OrnstienUhlenbeck_repr():
    b = OrnsteinUhlenbeck(mean=0.0, mrr=1.0, vol=0.1)
    assert str(b) == f"OrnsteinUhlenbeck(mean=[[0.]], mrr=[[1.]], vol=[[0.1]], cor=None)"
