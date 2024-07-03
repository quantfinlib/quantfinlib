import numpy as np
import pandas as pd


# Various DatetimeIndex
def index_b():
    return pd.bdate_range('2020-9-12', periods=3)

def index_d():
    return pd.date_range('2020-9-12', periods=3, freq='B')

def index_m():
    return pd.date_range('2020-9-12', periods=3, freq='ME')

def index_2d():
    return pd.date_range('2020-9-12', periods=3, freq='2D')

def x_int():
    return 42

def x_float():
    return 3.14

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