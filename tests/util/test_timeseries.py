import pandas as pd
import numpy as np
import pytest

from quantfinlib.util.timeseries import infer_time_series_resolution

def _make_df(freq):  
    date_rng = pd.date_range(start='2020-01-12', end='2024-12-31', freq=freq)
    df = pd.DataFrame(date_rng, columns=['date'])
    df['data'] = np.random.randn(len(date_rng))
    df = df.set_index('date')
    return df
  
def test_infer_time_series_resolution():
    
    # Test non-pandas type
    assert infer_time_series_resolution([1,2,3]) == None
    assert infer_time_series_resolution(pd.DataFrame([1,2,3])) == None

    # Test common types
    for freq in ['D', 'B', 'W', 'ME', 'MS', 'QE', 'QS', 'YE', 'YS']:
        df = _make_df(freq)
        infered_freq = infer_time_series_resolution(df)
        assert infered_freq == freq

    # Test pandas dataframe with too little rows
    df = _make_df('D')
    assert infer_time_series_resolution(df.iloc[:2,:]) == None

    # Test a frequency we dont support (every 2 days)
    df = _make_df('D')
    assert infer_time_series_resolution(df.iloc[::2,:]) == None