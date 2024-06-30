import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantfinlib.sim._base import SimHelperBase


def index_b():
    return pd.bdate_range('2020-9-12', periods=3)

def index_d():
    return pd.date_range('2020-9-12', periods=3, freq='B')

def index_m():
    return pd.date_range('2020-9-12', periods=3, freq='ME')

def index_2d():
    return pd.date_range('2020-9-12', periods=3, freq='2D')


def x_list():
    return [0.5, -0.4, 1.0]

def x_nparray1d():
    return np.array(x_list())

def x_nparray2d():
    return np.array(x_list()).reshape(-1, 1)

def x_series():
    return pd.Series(data=x_list(), name='a')

def x_series_b():
    return pd.Series(data=x_list(), index=index_b(), name='a')

def x_series_d():
    return pd.Series(data=x_list(), index=index_d(), name='a')

def x_series_m():
    return pd.Series(data=x_list(), index=index_m(), name='a')

def x_dataframe1():
    return pd.DataFrame(data=x_list(), columns=['a'])

def x_dataframe1_b():
    return pd.DataFrame(data=x_list(), columns=['a'], index=index_b())

def x_dataframe1_d():
    return pd.DataFrame(data=x_list(), columns=['a'], index=index_d())

def x_dataframe1_2d():
    return pd.DataFrame(data=x_list(), columns=['a'], index=index_2d())


def x_dataframe1_m():
    return pd.DataFrame(data=x_list(), columns=['a'], index=index_m())

def x_dataframe2():
    return pd.DataFrame(data={'a':x_list(), 'b':x_list()})

def x_dataframe2_b():
    return pd.DataFrame(data={'a':x_list(), 'b':x_list()}, index=index_b())

def x_dataframe2_d():
    return pd.DataFrame(data={'a':x_list(), 'b':x_list()}, index=index_d())

def x_dataframe2_m():
    return pd.DataFrame(data={'a':x_list(), 'b':x_list()}, index=index_m())

@pytest.mark.parametrize("x", [
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
        x_dataframe2_m(),
    ])
def test_sim_base_fit_x_types(x):
    b = SimHelperBase()
    dt = 0.1
    b.inspect_and_normalize_fit_args(x, dt)

    assert b.fit_num_rows_ == 3

    if isinstance(x, np.ndarray):
        num_cols = 1 if (x.ndim == 1) else x.shape[1]
        assert b.fit_num_cols_ == num_cols
        assert b.fit_container_dtype_ == np.ndarray

    if isinstance(x, pd.Series):
        assert b.fit_num_cols_ == 1
        assert b.fit_container_dtype_ == pd.Series

    if isinstance(x, pd.DataFrame):
        assert b.fit_num_cols_ == x.shape[1]
        assert b.fit_container_dtype_ == pd.DataFrame




@pytest.mark.parametrize("x", [
        x_series_d(),
        x_dataframe1_d(),
        x_dataframe2_d(),
    ])
def test_sim_base_fit_dt_inference_d(x):
    b = SimHelperBase()
    dt = None
    _, dt = b.inspect_and_normalize_fit_args(x, dt)
    assert dt == 1/365


@pytest.mark.parametrize("x", [
        x_series_m(),
        x_dataframe1_m(),
        x_dataframe2_m(),
    ])
def test_sim_base_fit_dt_inference_m(x):
    b = SimHelperBase()
    dt = None
    _, dt = b.inspect_and_normalize_fit_args(x, dt)
    assert dt == 1/12


def test_sim_base_value_error_dt_1():
    with pytest.raises(ValueError):
        b = SimHelperBase()
        _, dt = b.inspect_and_normalize_fit_args(x_dataframe1(), None)

def test_sim_base_value_error_dt_2():
    with pytest.raises(ValueError):
        b = SimHelperBase()
        _, dt = b.inspect_and_normalize_fit_args(x_dataframe1_2d(), None)


def test_normalize_sim_path_args_1():
    b = SimHelperBase()
    x0 = None
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)
    assert_allclose(x0, np.array([[3.12]]), rtol=1e-5, atol=1e-8)
    assert dt == 1/252
    assert label_start is None
    assert label_freq is None


def test_normalize_sim_path_args_1e1():
    b = SimHelperBase()
    x0 = "first"
    dt = None
    label_start = None
    label_freq = None
    with pytest.raises(ValueError):
        x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)

def test_normalize_sim_path_args_1e2():
    b = SimHelperBase()
    x0 = "last"
    dt = None
    label_start = None
    label_freq = None
    with pytest.raises(ValueError):
        x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)


def test_normalize_sim_path_args_2():
    b = SimHelperBase()
    x = x_dataframe1()
    b.inspect_and_normalize_fit_args(x, 1/252)
    x0 = None
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq)
    assert_allclose(x0, x.values[0,:].reshape(1,-1), rtol=1e-5, atol=1e-8)
    assert dt == 1/252
    assert label_start is None
    assert label_freq is None    

def test_normalize_sim_path_args_3():
    b = SimHelperBase()
    x = x_dataframe1_d()
    b.inspect_and_normalize_fit_args(x, None)
    x0 = None
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq)
    assert_allclose(x0, x.values[0,:].reshape(1,-1), rtol=1e-5, atol=1e-8)
    assert dt == 1/365
    assert label_start == x.index[0]
    assert label_freq == "D"  

def test_normalize_sim_path_args_4():
    b = SimHelperBase()
    x = x_dataframe1()
    b.inspect_and_normalize_fit_args(x, 1/252)
    x0 = "first"
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq)
    assert_allclose(x0, x.values[0,:].reshape(1,-1), rtol=1e-5, atol=1e-8)
    assert dt == 1/252
    assert label_start is None
    assert label_freq  is None    


def test_normalize_sim_path_args_5():
    b = SimHelperBase()
    x = x_dataframe1()
    b.inspect_and_normalize_fit_args(x, 1/252)
    x0 = "last"
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq)
    assert_allclose(x0, x.values[-1,:].reshape(1,-1), rtol=1e-5, atol=1e-8)
    assert dt == 1/252
    assert label_start is None
    assert label_freq  is None    

def test_normalize_sim_path_args_4t():
    b = SimHelperBase()
    x = x_dataframe1_d()
    b.inspect_and_normalize_fit_args(x, None)
    x0 = "first"
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq)
    assert_allclose(x0, x.values[0,:].reshape(1,-1), rtol=1e-5, atol=1e-8)
    assert dt == 1/365
    assert label_start == x.index[0]
    assert label_freq == "D"      


def test_normalize_sim_path_args_5t():
    b = SimHelperBase()
    x = x_dataframe1_d()
    b.inspect_and_normalize_fit_args(x, None)
    x0 = "last"
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq)
    assert_allclose(x0, x.values[-1,:].reshape(1,-1), rtol=1e-5, atol=1e-8)
    assert dt == 1/365
    assert label_start == x.index[-1]
    assert label_freq == "D"


def test_normalize_sim_path_args_value_error_1():
    b = SimHelperBase()
    x0 = "first"
    dt = None
    label_start = None
    label_freq = None
    with pytest.raises(ValueError):
        x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq)

def test_normalize_sim_path_args_value_error_2():
    b = SimHelperBase()
    x0 = "lAST"
    dt = None
    label_start = None
    label_freq = None
    with pytest.raises(ValueError):
        x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq)


def test_normalize_sim_path_args_x0_default():
    b = SimHelperBase()
    x0 = None
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)
    assert_allclose(x0, np.array([[3.12]]), rtol=1e-5, atol=1e-8)

def test_normalize_sim_path_args_x0_float():
    b = SimHelperBase()
    x0 = 3.14
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)
    assert_allclose(x0, np.array([[3.14]]), rtol=1e-5, atol=1e-8)

def test_normalize_sim_path_args_x0_int():
    b = SimHelperBase()
    x0 = 42
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)
    assert_allclose(x0, np.array([[42]]), rtol=1e-5, atol=1e-8)

def test_normalize_sim_path_args_x0_list():
    b = SimHelperBase()
    x0 = [13]
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)
    assert_allclose(x0, np.array([[13]]), rtol=1e-5, atol=1e-8)

def test_normalize_sim_path_args_x0_ndarray():
    b = SimHelperBase()
    x0 = np.array([2])
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)
    assert_allclose(x0, np.array([[2]]), rtol=1e-5, atol=1e-8)    

def test_normalize_sim_path_args_x0_series():
    b = SimHelperBase()
    x0 = pd.Series([5])
    dt = None
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)
    assert_allclose(x0, np.array([[5]]), rtol=1e-5, atol=1e-8)    


def test_normalize_sim_path_args_dt_set():
    b = SimHelperBase()
    x0 = pd.Series([5])
    dt = 1/100
    label_start = None
    label_freq = None
    x0, dt, label_start, label_freq = b.normalize_sim_path_args(x0, dt, label_start, label_freq, 3.12)


def test_format_ans_ndarray_to_ndarray():
    b = SimHelperBase()
    x = np.arange(10).reshape(-1, 1)
    y = b.format_ans(x, None, None)
    assert isinstance(y, np.ndarray)

def test_format_ans_ndarray_to_series_fit_labels_1():
    b = SimHelperBase()
    x, dt = b.inspect_and_normalize_fit_args(x_series_d(), None)
    x = np.arange(10).reshape(-1, 1)
    y = b.format_ans(x, '2000-01-01', None)
    assert isinstance(y, pd.Series)

def test_format_ans_ndarray_to_series_fit_labels_2():
    b = SimHelperBase()
    x, dt = b.inspect_and_normalize_fit_args(x_series_d(), None)
    x = np.arange(10).reshape(-1, 1)
    y = b.format_ans(x, None, 'D')
    assert isinstance(y, pd.Series)

def test_format_ans_ndarray_to_series_labels():
    b = SimHelperBase()
    x = np.arange(10).reshape(-1, 1)
    y = b.format_ans(x, '2000-01-01', 'D')
    assert isinstance(y, pd.Series)

def test_format_ans_ndarray_to_series_fit():
    b = SimHelperBase()
    x, dt = b.inspect_and_normalize_fit_args(x_series_d(), None)
    x_out = np.arange(10).reshape(-1, 1)
    y = b.format_ans(x_out, None, None)
    assert isinstance(y, pd.Series)

# 2d
def test_format_ans_ndarray_2D_to_series_labels():
    b = SimHelperBase()
    x = np.arange(12).reshape(-1, 2)
    y = b.format_ans(x, '2000-01-01', 'D')
    assert isinstance(y, pd.DataFrame)

def test_format_ans_ndarray_2D_to_dataframe_labels_2paths():
    b = SimHelperBase()
    x = np.arange(12).reshape(-1, 2)
    y = b.format_ans(x, '2000-01-01', 'D', num_paths=2)
    assert isinstance(y, pd.DataFrame)

def test_format_ans_ndarray_2D_to_dataframe_2paths_2cols():
    b = SimHelperBase()
    x = np.arange(12).reshape(-1, 4)
    y = b.format_ans(x, '2000-01-01', 'D', num_paths=2)
    assert isinstance(y, pd.DataFrame)

def test_format_ans_ndarray_2D_to_dataframe_exclude_x0():
    b = SimHelperBase()
    x = np.arange(12).reshape(-1, 4)
    y = b.format_ans(x, '2000-01-01', 'D', num_paths=2, include_x0=False)
    assert isinstance(y, pd.DataFrame)
    assert y.shape[0] == 2

def test_format_ans_ndarray_2D_to_array_exclude_x0():
    b = SimHelperBase()
    x = np.arange(12).reshape(-1, 4)
    y = b.format_ans(x,None, None, num_paths=2, include_x0=False)
    assert isinstance(y, np.ndarray)
    assert y.shape[0] == 2

def test_make_columns_namest():
    b = SimHelperBase()
    c = b._make_columns_names( num_target_columns=6, num_paths=3)
    assert c[0] == 'S0_0'
    assert c[5] == 'S1_2'

def test_make_columns_names_set():
    b = SimHelperBase()
    c = b._make_columns_names( num_target_columns=6, num_paths=3, columns=['AA', 'BB'])
    assert c[0] == 'AA_0'
    assert c[5] == 'BB_2'

def test_set_x0():
    ans = np.zeros(shape=(4,12))
    SimHelperBase.set_x0(ans, [10,20,30])
    assert ans[0,0] == 10
    assert ans[0,1] == 20
    assert ans[0,2] == 30

    assert ans[0,3] == 10
    assert ans[0,4] == 20
    assert ans[0,5] == 30
    
    assert ans[0,6] == 10