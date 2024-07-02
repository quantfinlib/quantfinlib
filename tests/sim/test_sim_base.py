from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim._base import (SimBase, _DateTimeIndexInfo,
                                   _get_column_names, _get_date_time_index,
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
# _DateTimeIndexInfo()
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("x", [
        x_series_b(),
        x_series_d(),
        x_series_m(),
        x_dataframe1_b(),
        x_dataframe1_d(),
        x_dataframe1_m(),
        x_dataframe2_b(),
        x_dataframe2_d(),
        x_dataframe2_m()
    ])
def test_DateTimeIndexInfo(x):
    info = _DateTimeIndexInfo(x)
    assert info.min_ is not None


@pytest.mark.parametrize("x", [
        x_int(),
        x_float(),
        x_list(),
        x_nparray1d(),
        x_nparray2d(),
        x_dataframe1(),
        x_dataframe2(),
    ])
def test_DateTimeIndexInfo_negative(x):
    info = _DateTimeIndexInfo(x)
    assert info.min_ is None

# ------------------------------------------------------------------------------
# SimBase._fit_preprocess_x_and_dt()
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

@pytest.mark.parametrize("x", [
    x_series_b(),
    x_series_d(),
    x_series_m(),
    x_dataframe1_b(),
    x_dataframe1_d(),
    x_dataframe1_m(),
    x_dataframe2_b(),
    x_dataframe2_d(),
    x_dataframe2_m()
])
def test_SimBase_fit_preprocess_x_and_dt_no_dt_positive(x):
    b = SimBaseTest()
    x_new, dt_new = b._preprocess_fit_x_and_dt(x, None)
    assert isinstance(x_new, np.ndarray)
    assert dt_new is not None


@pytest.mark.parametrize("x", [
        x_dataframe1_2d(),
])
def test_SimBase_fit_preprocess_x_and_dt_no_dt_negative(x):
    b = SimBaseTest()
    with pytest.raises(ValueError):
        x_new, dt_new = b._preprocess_fit_x_and_dt(x, None)



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
def test_SimBase_fit_preprocess_x_and_dt__dt_set(x):
    b = SimBaseTest()
    x_new, dt_new = b._preprocess_fit_x_and_dt(x, 42.0)
    assert isinstance(x_new, np.ndarray)
    assert dt_new == 42.0


# ------------------------------------------------------------------------------
# SimBase._preprocess_sim_path_args()
# ------------------------------------------------------------------------------
def test_SimBase_preprocess_sim_path_args_not_fitted():
    b = SimBaseTest()
    x, x0, dt, label_start, label_freq, x0_default = None, None, None, None, None, 0.0
    
    x = x_nparray1d()

    x0, dt, label_start, label_freq = b._preprocess_sim_path_args(
        x0, dt, label_start, label_freq, x0_default)

def test_SimBase_preprocess_sim_path_args_not_fitted_2():
    x, x0, dt, label_start, label_freq, x0_default = None, None, None, None, None, 0.0
    
    x = x_nparray1d()
    for x0 in ["first", "last", "hello"]:
        b = SimBaseTest()
        with pytest.raises(ValueError):
            x0, dt, label_start, label_freq = b._preprocess_sim_path_args(
                x0, dt, label_start, label_freq, x0_default)


def test_SimBase_preprocess_sim_path_args_fitted():
    b = SimBaseTest()
    x, x0, dt, label_start, label_freq, x0_default = None, None, None, None, None, 0.0
    
    x = x_nparray1d()

    x, dt = b._preprocess_fit_x_and_dt(x, dt)
    x0, dt, label_start, label_freq = b._preprocess_sim_path_args(
        x0, dt, label_start, label_freq, x0_default)


def test_SimBase_preprocess_sim_path_args_fitted_1():
    b = SimBaseTest()
    x, x0, dt, label_start, label_freq, x0_default = None, None, None, None, None, 0.0
    
    x = x_nparray1d()

    x, _ = b._preprocess_fit_x_and_dt(x, dt)
    x0, dt, label_start, label_freq = b._preprocess_sim_path_args(
        x0, dt, label_start, label_freq, x0_default)


def test_SimBase_preprocess_sim_path_args_fitted_2():
    b = SimBaseTest()
    x, x0, dt, label_start, label_freq, x0_default = None, None, None, None, None, 0.0
    
    x = x_nparray1d()
    label_freq = "D"

    x, dt = b._preprocess_fit_x_and_dt(x, dt)
    x0, dt, label_start, label_freq = b._preprocess_sim_path_args(
        x0, dt, label_start, label_freq, x0_default)


def test_SimBase_preprocess_sim_path_args_fitted_3():        
    for x0 in ["first", "last"]:
        x, dt, label_start, label_freq, x0_default = None, None, None, None, 0.0
        b = SimBaseTest()
        x = x_nparray1d()
        x, dt = b._preprocess_fit_x_and_dt(x, dt)
        x0, dt, label_start, label_freq = b._preprocess_sim_path_args(
            x0, dt, label_start, label_freq, x0_default)
        assert x0 is not None
        assert label_start is None

def test_SimBase_preprocess_sim_path_args_fitted_4():    
    for x0 in ["first", "last"]:
        x, dt, label_start, label_freq, x0_default = None, None, None, None, 0.0
        b = SimBaseTest()
        x = x_dataframe1_b()
        x, dt = b._preprocess_fit_x_and_dt(x, dt)
        print('case 4', x0)
        x0, dt, label_start, label_freq = b._preprocess_sim_path_args(
            x0, dt, label_start, label_freq, x0_default)
        assert x0 is not None
        assert label_start is not None


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

def test_make_column_names_fitted():
    num_target_columns = 6
    num_paths = 3
    columns = None
    b = SimBaseTest()
    x = x_dataframe2_m()
    dt = None
    x, dt = b._preprocess_fit_x_and_dt(x, dt)
    ans =  b._make_columns_names(num_target_columns, num_paths, columns)
    assert ans[0] == 'a_0'
    assert ans[1] == 'b_0'
    assert ans[2] == 'a_1'
    assert ans[3] == 'b_1'
    assert ans[4] == 'a_2'
    assert ans[5] == 'b_2'

def test_make_column_names_fitted_names():
    num_target_columns = 6
    num_paths = 3
    columns = None
    b = SimBaseTest()
    x = x_dataframe2_m()
    dt = None
    x, dt = b._preprocess_fit_x_and_dt(x, dt)
    columns = ['A', 'BB']
    ans =  b._make_columns_names(num_target_columns, num_paths, columns)
    assert ans[0] == 'A_0'
    assert ans[1] == 'BB_0'
    assert ans[2] == 'A_1'
    assert ans[3] == 'BB_1'
    assert ans[4] == 'A_2'
    assert ans[5] == 'BB_2'

# ------------------------------------------------------------------------------
# SimBase._make_columns_names()
# ------------------------------------------------------------------------------
def test_make_date_time_index():
    b = SimBaseTest()
    num_rows = 4
    label_start = '2000-03-31'
    label_freq = 'D'
    ans = b._make_date_time_index(num_rows, label_start, label_freq)

def test_make_date_time_index_fitted():
    b = SimBaseTest()
    x = x_dataframe2_m()
    dt = None
    x, dt = b._preprocess_fit_x_and_dt(x, dt)    
    num_rows = 4
    label_start = '2000-03-31'
    label_freq = None
    ans = b._make_date_time_index(num_rows, label_start, label_freq)    

def test_make_date_time_index_fitted_2():
    b = SimBaseTest()
    x = x_dataframe2_m()
    dt = None
    x, dt = b._preprocess_fit_x_and_dt(x, dt)    
    num_rows = 4
    label_start = None
    label_freq = None
    ans = b._make_date_time_index(num_rows, label_start, label_freq)        

def test_make_date_time_index_fitted_3():
    b = SimBaseTest()
    x = x_dataframe2_m()
    dt = None
    x, dt = b._preprocess_fit_x_and_dt(x, dt)    
    num_rows = 4
    label_start = None
    label_freq = 'W'
    ans = b._make_date_time_index(num_rows, label_start, label_freq)

# ------------------------------------------------------------------------------
# SimBase._make_columns_names()
# ------------------------------------------------------------------------------
def test_format_ans():
    b = SimBaseTest()
    x = np.zeros(shape=(10, 6))
    label_start = None
    label_freq = None
    columns = None
    include_x0 = True
    num_paths = 6
    ans = b._format_ans(x, label_start, label_freq, columns, include_x0, num_paths)
    assert isinstance(ans, np.ndarray)

def test_format_ans_series():
    b = SimBaseTest()
    x = np.zeros(shape=(10, 1))
    label_start = '2000-01-01'
    label_freq = 'D'
    columns = None
    include_x0 = True
    num_paths = 1
    ans = b._format_ans(x, label_start, label_freq, columns, include_x0, num_paths)
    assert isinstance(ans, pd.Series)

def test_format_ans_df():
    b = SimBaseTest()
    x = np.zeros(shape=(10, 6))
    label_start = '2000-01-01'
    label_freq = 'D'
    columns = None
    include_x0 = True
    num_paths = 6
    ans = b._format_ans(x, label_start, label_freq, columns, include_x0, num_paths)
    assert isinstance(ans, pd.DataFrame)

def test_format_ans_df_no_x0_df():
    b = SimBaseTest()
    x = np.zeros(shape=(10, 6))
    label_start = '2000-01-01'
    label_freq = 'D'
    columns = None
    include_x0 = False
    num_paths = 6
    ans = b._format_ans(x, label_start, label_freq, columns, include_x0, num_paths)
    assert ans.shape[0] == 9   

def test_format_ans_df_no_x0():
    b = SimBaseTest()
    x = np.zeros(shape=(10, 6))
    label_start = None
    label_freq = None
    columns = None
    include_x0 = False
    num_paths = 6
    ans = b._format_ans(x, label_start, label_freq, columns, include_x0, num_paths)
    assert ans.shape[0] == 9       


def test_overload():
    b = SimBaseTest()
    x0 = np.zeros(4)
    dt = 0.1
    num_steps = 4
    num_paths = 5
    random_state = 6
    ans = b._path_sample_np(x0, dt, num_steps, num_paths, random_state)
