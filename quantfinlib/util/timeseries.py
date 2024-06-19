from typing import Optional
import pandas as pd


def infer_time_series_resolution(ts_obj) -> Optional[str]:
    """
    Determine the resolution of a time series.

    This function analyzes the index of a pandas DataFrame or Series to determine
    if the time series has a daily resolution of 7 days a week, a working-day
    resolution of roughly 5 days a week, weekly, monthly, quarterly, or yearly 
    resolution. Additionally, it can distinguish month-end, month-start, quarter-end, 
    quarter-start, year-end, and year-start resolutions.

    Parameters
    ----------
    ts_obj : pandas.DataFrame or pandas.Series
        The time series object to be analyzed. It must have a DateTimeIndex.

    Returns
    -------
    Optional[str]
        A string representing the resolution of the time series:

        * 'D'  : Daily resolution (7 days a week)
        * 'B'  : Working-day resolution (roughly 5 days a week)
        * 'W'  : Weekly resolution (7 days)
        * 'ME' : Month-end resolution
        * 'MS' : Month-start resolution
        * 'QE' : Quarter-end resolution
        * 'QS' : Quarter-start resolution
        * 'YE' : Year-end resolution
        * 'YS' : Year-start resolution
        * None : if the resolution is irregular or cannot be determined.

    Notes
    -----
    The function requires at least three time points to determine the resolution.
    It checks the frequency of time deltas and specific characteristics of the 
    time series index to classify its resolution.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> date_rng = pd.date_range(start='2022-01-01', end='2022-01-15', freq='D')
    >>> df = pd.DataFrame(date_rng, columns=['date'])
    >>> df['data'] = np.random.randn(len(date_rng))
    >>> df = df.set_index('date')
    >>> infer_time_series_resolution(df)
    'D'

    >>> date_rng = pd.bdate_range(start='2022-01-01', end='2022-01-15')
    >>> df = pd.DataFrame(date_rng, columns=['date'])
    >>> df['data'] = np.random.randn(len(date_rng))
    >>> df = df.set_index('date')
    >>> infer_time_series_resolution(df)
    'B'
    """
    if not isinstance(ts_obj, (pd.DataFrame, pd.Series)):
        return None  # "The provided object is not a pandas DataFrame or Series."

    if not isinstance(ts_obj.index, pd.DatetimeIndex):
        return None  # "The provided object does not have a DateTimeIndex."

    # Calculate the time deltas between consecutive time points
    time_deltas = ts_obj.index.to_series().diff().dropna()

    # Ensure there are enough data points to make a determination
    if len(time_deltas) < 2:
        return None  # "The time series index must contain more than one unique time point to determine resolution."

    # Count the occurrences of each time delta
    delta_counts = time_deltas.value_counts()

      
    # Check if timestamps are year-ends
    year_ends = ts_obj.index.is_year_end.sum()
    if (year_ends >= 2) and (len(ts_obj.index) - year_ends <= 2):
        return "YE"
    
    # Check if timestamps are year-starts
    year_starts = ts_obj.index.is_year_start.sum()
    if (year_starts >= 2) and (len(ts_obj.index) - year_starts <= 2):
        return "YS"


    # Check if timestamps are quarter-ends
    quarter_ends = ts_obj.index.is_quarter_end.sum()
    if (quarter_ends >= 2) and (len(ts_obj.index) - quarter_ends <= 2):
        return "QE"
    
    # Check if timestamps are quarter-starts
    quarter_starts = ts_obj.index.is_quarter_start.sum()
    if (quarter_starts >= 2) and (len(ts_obj.index) - quarter_starts <= 2):
        return "QS"

    # Check if timestamps are month-ends
    print('is_month_end', ts_obj.index.is_month_end)
    month_ends = ts_obj.index.is_month_end.sum()
    if (month_ends >= 2) and (len(ts_obj.index) - month_ends <= 2):
        return "ME"
    
    # Check if timestamps are month-starts
    month_starts = ts_obj.index.is_month_start.sum()
    if (month_starts >= 2) and (len(ts_obj.index) - month_starts <= 2):
        return "MS"
    
    # Check for daily resolution (7 days a week)
    daily_count = delta_counts.get(pd.Timedelta(days=1), 0)
    daily_resolution = len(time_deltas) - daily_count <= 2
    if daily_resolution:
        return "D"

    # Check for working-day resolution (5 days a week)
    weekday_deltas = time_deltas[time_deltas.dt.days.isin([1, 3])]
    weekday_resolution = (len(weekday_deltas) / len(time_deltas)) >= 0.9
    if weekday_resolution:
        return "B"

    # Check for weekly resolution (timesteps of 7 days)
    weekly_count = delta_counts.get(pd.Timedelta(days=7), 0)
    if (weekly_count >= 2) and (len(time_deltas) - weekly_count <= 2):
        return "W"
    

    return None

