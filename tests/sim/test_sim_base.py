from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim._base import (SimBase,
                                   _get_column_names,
                                   _SimFitDataType, _to_numpy)

from .sample_type_instances import *


# ------------------------------------------------------------------------------
# _to_numpy()
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("x", [
        x_int(),
        x_float(),
        x_list(),
        x_nparray1d(),
        x_nparray2d(),
        x_series(),
        x_series_b(),
        x_series_d(),
        x_series_m(),
        x_dataframe1(),
        x_dataframe1_b(),
        x_dataframe1_d(),
        x_dataframe1_m(),
        x_dataframe2(),
        x_dataframe2_b(),
        x_dataframe2_d(),
        x_dataframe2_m()
    ])
def test_sim_base_to_numpy(x):
    assert _to_numpy(x) is not None

def test_sim_base_to_numpy_negative():
    with pytest.raises(ValueError):
        tmp = _to_numpy(None)
    with pytest.raises(ValueError):
        tmp = _to_numpy("I'm not a number")

# ------------------------------------------------------------------------------
# _get_column_names()
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("x", [
        x_series(),
        x_series_b(),
        x_series_d(),
        x_series_m(),
        x_dataframe1(),
        x_dataframe1_b(),
        x_dataframe1_d(),
        x_dataframe1_m(),
        x_dataframe2(),
        x_dataframe2_b(),
        x_dataframe2_d(),
        x_dataframe2_m()
    ])
def test_sim_base_get_column_names(x):
    assert isinstance(_get_column_names(x), list)


@pytest.mark.parametrize("x", [
        x_int(),
        x_float(),
        x_list(),
        x_nparray1d(),
        x_nparray2d()
    ])
def test_sim_base_get_column_names_negative(x):
    assert _get_column_names(x) is None



# ------------------------------------------------------------------------------
# SimBaseTest
# ------------------------------------------------------------------------------
class SimBaseTest(SimBase):
    def _path_sample_np(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        return np.empty(shape=(1,))
    
    def _fit_np(self, x: np.ndarray, dt: float):
        pass


# ------------------------------------------------------------------------------
# SimBase._make_columns_names()
# ------------------------------------------------------------------------------

def test_make_column_names():
    num_target_columns = 1
    num_paths = 1
    columns = None
    b = SimBaseTest()
    ans =  b._make_columns_names(num_target_columns, num_paths, columns)
    assert ans[0] == 'S'

def test_make_column_names_1_3():
    num_target_columns = 3
    num_paths = 3
    columns = None
    b = SimBaseTest()
    ans =  b._make_columns_names(num_target_columns, num_paths, columns)
    assert ans[0] == 'S_0'
    assert ans[1] == 'S_1'
    assert ans[2] == 'S_2'

def test_make_column_names_6():
    num_target_columns = 6
    num_paths = 1
    columns = None
    b = SimBaseTest()
    ans =  b._make_columns_names(num_target_columns, num_paths, columns)
    assert ans[0] == 'S0'
    assert ans[1] == 'S1'
    assert ans[2] == 'S2'
    assert ans[3] == 'S3'
    assert ans[4] == 'S4'
    assert ans[5] == 'S5'

def test_make_column_names_2p():
    num_target_columns = 6
    num_paths = 2
    columns = None
    b = SimBaseTest()
    ans =  b._make_columns_names(num_target_columns, num_paths, columns)
    assert ans[0] == 'S0_0'
    assert ans[1] == 'S1_0'
    assert ans[2] == 'S2_0'
    assert ans[3] == 'S0_1'
    assert ans[4] == 'S1_1'
    assert ans[5] == 'S2_1'

def test_make_column_names_2p_names():
    num_target_columns = 6
    num_paths = 2
    columns = ['A', 'BB', 'CCC']
    b = SimBaseTest()
    ans =  b._make_columns_names(num_target_columns, num_paths, columns)
    assert ans[0] == 'A_0'
    assert ans[1] == 'BB_0'
    assert ans[2] == 'CCC_0'
    assert ans[3] == 'A_1'
    assert ans[4] == 'BB_1'
    assert ans[5] == 'CCC_1'



def test_overload():
    b = SimBaseTest()
    x0 = np.zeros(4)
    dt = 0.1
    num_steps = 4
    num_paths = 5
    random_state = 6
    ans = b._path_sample_np(x0, dt, num_steps, num_paths, random_state)
