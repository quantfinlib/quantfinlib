from typing import Union, Optional
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from quantfinlib.util.inspect import num_rows, num_cols


def is_time_series(obj):
    """Check if an object is a pandas DataFrame or Series with a DateTimeIndex."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if isinstance(obj.index, pd.DatetimeIndex):
            return True
    return False


def estimate_time_step_size(ts_obj):
    """Estimate the time-step size of a pandas DataFrame or Series with a DateTimeIndex."""
    if is_time_series(ts_obj):
        time_deltas = ts_obj.index.to_series().diff().dropna()
        if len(time_deltas) > 0:
            return time_deltas.mode()[0]  # Most common time delta
        else:
            raise ValueError(
                "The time series index must contain more than one unique time point to estimate the time-step size."
            )
    else:
        raise TypeError(
            "The provided object is not a pandas DataFrame or Series with a DateTimeIndex."
        )


def check_time_series_resolution(ts_obj):
    """Determine if the time series has a daily resolution of 7 days a week or roughly 5 days a week."""
    if not isinstance(ts_obj, (pd.DataFrame, pd.Series)):
        raise TypeError("The provided object is not a pandas DataFrame or Series.")

    if not isinstance(ts_obj.index, pd.DatetimeIndex):
        raise TypeError("The provided object does not have a DateTimeIndex.")

    # Calculate the time deltas between consecutive time points
    time_deltas = ts_obj.index.to_series().diff().dropna()

    # Ensure there are enough data points to make a determination
    if len(time_deltas) < 2:
        raise ValueError(
            "The time series index must contain more than one unique time point to determine resolution."
        )

    # Determine the mode of the time deltas
    mode_delta = time_deltas.mode()[0]

    # Count the occurrences of each time delta
    delta_counts = time_deltas.value_counts()

    # Check for daily resolution (7 days a week)
    daily_count = delta_counts[pd.Timedelta(days=1)]
    daily_resolution = (daily_count / len(time_deltas)) >= 0.7

    # Check for working-day resolution (5 days a week)
    weekday_deltas = time_deltas[time_deltas.dt.days.isin([1, 3])]
    weekday_resolution = (len(weekday_deltas) / len(time_deltas)) >= 0.7

    if daily_resolution:
        return "Daily resolution (7 days a week)"
    elif weekday_resolution:
        return "Working-day resolution (roughly 5 days a week)"
    else:
        return "Irregular resolution or different frequency"


def _guess_dt(x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> float:
    return 1 / 252.0
    if isinstance(x, pd.Series):
        return 1
    elif isinstance(x, pd.DataFrame):
        return len(x.columns)
    elif isinstance(x, np.ndarray):
        return 1.0 / 252.0
    return 1.0 / 252.0


class BrownianMotion(BaseEstimator):
    """Brownian Motion simulation"""

    def __init__(self, drift=0.0, vol=0.1, cor=None):
        # Parameters
        self.drift = drift
        self.vol = vol
        self.cor = None

        # Private attributes
        self.x0_ = 0
        self.dt_ = 1.0 / 252.0
        self.num_samples_ = 253

        if self.cor is None:
            self.L_ = None
        else:
            self.L_ = np.linalg.cholesky(self.cor)

    def fit(
        self,
        x: Union[np.ndarray, pd.DataFrame, pd.Series],
        dt: Optional[float],
        **kwargs
    ):

        # Make sure that x is a 2d array
        x_ = np.asarray(x)
        if x_.ndim == 1:
            x_ = x_.reshape(-1, 1)

        # Initial value
        self.dt_ = dt
        self.x0_ = x_[0, :].reshape(1, -1)
        self.num_samples_ = x_.shape[0]

        # changes from one row to the next
        dx = np.diff(x_, axis=0)

        # Mean and standard deviation of the changes
        self.drift = np.mean(dx, axis=0, keepdims=True) / self.dt_
        self.vol = np.std(dx, axis=0, ddof=1, keepdims=True) / np.sqr(self.dt_)

        # Optionally correlations if we have multiple columns
        if x.shape[1] > 1:
            self.cor = np.corrcoef(dx, rowvar=False)
            self.L_ = np.linalg.cholesky(self.cor)
        else:
            self.cor = None
            self.L_ = None

        return self

    def sample(
        self,
        x0: Union[float, np.ndarray, pd.DataFrame, pd.Series, None],
        dt: Optional[float],
        num_steps: Optional[int],
        random_state: Optional[int] = None,
        include_x0: bool = True,
    ) -> np.array:

        # Handle default values
        if x0 is None:
            x0 = self.x0_
        if dt is None:
            dt = self.dt_
        if num_steps is None:
            num_steps = self.num_steps_

        # Answer
        ans = np.zeros(shape=(num_steps + 1, self.mean.shape[1]))

        # set the initial value
        ans[0, :] = self.x0

        # Create a Generator instance with the seed
        rng = np.random.default_rng(random_state)

        # fill in Normal noise
        ans[1 : num_steps + 1, :] = rng.normal(size=(num_steps, self.drift.shape[1]))

        # Optionally correlate the noise
        if self.L_ is not None:
            ans[1 : num_steps + 1, :] = ans[1 : num_steps + 1, :] @ self.L_.T

        # Translate the noise with mean and variance
        ans[1 : num_steps + 1, :] = (
            ans[1 : num_steps + 1, :] * self.vol * dt**0.5 + self.drift * dt
        )

        # compound
        ans = np.cumsum(ans, axis=0)

        if include_x0:
            return ans
        else:
            return ans[1:, :]

        return ans
