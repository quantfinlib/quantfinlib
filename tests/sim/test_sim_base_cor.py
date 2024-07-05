import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim._base import (_fill_with_correlated_noise,
                                   _make_cor_from_upper_tri, _to_numpy,
                                   _triangular_index)


def test_triangular_index_good():
    for n in range(1, 10):
        T_n = int(n * (n+1) / 2)
        assert _triangular_index(T_n) == n


def test_triangular_index_bad():
    for T_n in [2, 4, 5, 7, 8 , 9]:
        with pytest.raises(ValueError):
            i = _triangular_index(T_n)


def test_fill_with_correlated_noise():
    ans = np.empty(shape=(5,3))
    
    _fill_with_correlated_noise(ans)
    _fill_with_correlated_noise(ans, loc=[1,1,1])


def test_make_cor_from_upper_tri():
    ans = _make_cor_from_upper_tri(0.6)
    assert_allclose(ans, np.array([[1.0, 0.6], [0.6, 1.0]]))

    u = [0.6, 0.5, 0.3]
    ans = _make_cor_from_upper_tri(u)
    assert_allclose(ans, np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.3], [0.5, 0.3, 1.0]]))

    u = _to_numpy([0.6, 0.5, 0.3]).reshape(1, -1)
    ans = _make_cor_from_upper_tri(u)
    assert_allclose(ans, np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.3], [0.5, 0.3, 1.0]]))

    u = _to_numpy([0.6, 0.5, 0.3]).reshape(-1, 1)
    ans = _make_cor_from_upper_tri(u)
    assert_allclose(ans, np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.3], [0.5, 0.3, 1.0]]))    


def test_make_cor_from_upper_tri_bad():
    with pytest.raises(ValueError):
        ans = _make_cor_from_upper_tri(np.zeros(shape=(2,2)))
