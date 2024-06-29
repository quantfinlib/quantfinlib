import pandas as pd
import numpy as np
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


def test_BrownianMotion_fit():
    b = BrownianMotion()
    x = 3.14 * np.ones(shape=(5, 1))
    b.fit(x, 0.1)
    assert_allclose(b.drift, np.array([0.0]).reshape(1, -1))
    assert_allclose(b.vol, np.array([0.0]).reshape(1, -1))


def test_BrownianMotion_path_sample():
    b = BrownianMotion()
    p = b.path_sample(x0=1, dt=1/12, num_steps=10, num_paths=1)
    assert p.shape[0] == 11
    assert p.shape[1] == 1