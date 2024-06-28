import pandas as pd
import numpy as np
import pytest

from quantfinlib.sim._base import SimHelperBase

def index_b():
    return pd.bdate_range('2020-9-12', periods=3)

def index_d():
    return pd.date_range('2020-9-12', periods=3, freq='B')

def index_m():
    return pd.date_range('2020-9-12', periods=3, freq='M')

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
