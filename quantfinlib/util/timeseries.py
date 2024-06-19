import pandas as pd


def is_time_series(obj):
    """Check if an object is a pandas DataFrame or Series with a DateTimeIndex."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if isinstance(obj.index, pd.DatetimeIndex):
            return True
    return False


def check_time_series_resolution(ts_obj):
    """Determine if the time series has a daily resolution of 7 days a week or roughly 5 days a week."""
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

    minimal_fraction = 0.8

    # Check for daily resolution (7 days a week)
    daily_count = delta_counts[pd.Timedelta(days=1)]
    daily_resolution = (daily_count / len(time_deltas)) >= minimal_fraction
    if daily_resolution:
        return "D"

    # Check for working-day resolution (5 days a week)
    weekday_deltas = time_deltas[time_deltas.dt.days.isin([1, 3])]
    weekday_resolution = (len(weekday_deltas) / len(time_deltas)) >= minimal_fraction
    if weekday_resolution:
        return "B"

    # Check for weekly resolution (timesteps of 7 days)
    weekly_count = delta_counts[pd.Timedelta(days=7)]
    weekly_resolution = (weekly_count / len(time_deltas)) >= minimal_fraction
    if weekly_resolution:
        return "W"

    # Check for monthly resolution
    monthly_deltas = time_deltas[time_deltas.dt.days.isin([28, 29, 30, 31])]
    monthly_resolution = (len(monthly_deltas) / len(time_deltas)) >= minimal_fraction

    if monthly_resolution:

        # Check if timestamps are month-ends
        month_ends = ts_obj.index.is_month_end
        month_end_resolution = (
            month_ends.sum() / len(ts_obj.index)
        ) >= minimal_fraction
        if month_end_resolution:
            return "ME"

        # Check if timestamps are month-starts
        month_starts = ts_obj.index.is_month_start
        month_start_resolution = (
            month_starts.sum() / len(ts_obj.index)
        ) >= minimal_fraction
        if month_start_resolution:
            return "MS"

        # Else it's just monthly
        return "M"

    return None


def freq_to_index(freq: str, t0: str, length: int) -> pd.DatetimeIndex:
    """Create a Pandas date-time index with equally spaced dates.

    Example
    -------

    .. exec_code::

        from samcoml.util import freq_to_index

        ndx = freq_to_index('B', t0='2020-1-6', length=10)
        print(ndx)


    Parameters
    ----------
    freq: str
        Frequency code string.
        * 'B' Business days, dt = 1 / 252
        * 'D' Regular days, dt = 1 / 365
        * 'W' Weekly dy = 7 / 365
        * 'M' or "MS': Monthly, dt = 1 / 12
    t0: str
        Start date string. String or datetime-like.
    length: int
        Length of the index.

    Returns
    -------

    """
    if freq == "B":
        return pd.bdate_range(start=t0, periods=length)
    elif freq == "D":
        return pd.date_range(start=t0, periods=length, freq="D")
    elif freq == "M":
        return pd.date_range(start=t0, periods=length, freq="M")
    elif freq == "W":
        return pd.date_range(start=t0, periods=length, freq="W")
    elif freq == "MS":
        return pd.date_range(start=t0, periods=length, freq="MS")
    raise ValueError("Invalid freq argument. Valid values are 'D', 'B', 'M' or 'MS'.")
