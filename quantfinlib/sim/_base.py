from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quantfinlib._datatypes.timeseries import time_series_freq_to_duration

_SimFitDataType = Union[int, float, list, np.ndarray, pd.DataFrame, pd.Series]


def _to_numpy(x: _SimFitDataType) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    elif isinstance(x, list):
        return np.array(x).reshape(1, -1)
    elif isinstance(x, int):
        return np.array(x, dtype=float).reshape(1, 1)
    elif isinstance(x, float):
        return np.array(x).reshape(1, 1)
    else:
        raise ValueError(f"Cannot convert {x} to a numpy array.")


def _get_column_names(x: _SimFitDataType) -> Optional[List[str]]:
    if isinstance(x, pd.DataFrame):
        return x.columns.values.tolist()
    elif isinstance(x, pd.Series):
        return [x.name]
    else:
        return None


def _get_date_time_index(x: Optional[Any]) -> Optional[pd.DatetimeIndex]:
    if not isinstance(x, (pd.DataFrame, pd.Series)):
        return None
    if not isinstance(x.index, pd.DatetimeIndex):
        return None
    return x.index

def _fill_with_correlated_noise(
    ans: np.ndarray, 
    loc: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    random_state: Optional[int] = None):

    # Create a Generator instance with the seed
    rng = np.random.default_rng(random_state)

    # fill with standard normal Nose
    ans[:, :] = rng.normal(size=ans.shape)
    if L is not None:
        ans[:, :] = ans[:, :] @ L.T
    if scale is not None:
        ans[:, :] *= scale
    if loc is not None:
        ans[:, :] += loc


def _create_prepared_sim_ans(
    x0: np.ndarray, num_steps: int, num_cols: int, num_paths: int, random_state: int, L: Optional[np.ndarray] = None
) -> np.ndarray:
    """Create an array filled with collelated random sample and x0 on the first row
    """
    # Allocate storage for the simulation
    ans = np.zeros(shape=(num_steps + 1, num_cols * num_paths))

    # set the initial value of the simulation
    SimBase.set_x0(ans, x0)

    # create a view on ans, skippin the first row x0
    dx = ans[1 : num_steps + 1, :]

    # fill in Normal noise
    dx[:, :] = rng.normal(size=dx.shape)

    # Optionally correlate the noise
    if L is not None:
        # reshape
        dx = dx.reshape(-1, num_paths, num_cols)
        # correlate
        dx = dx @ L.T
        # reshape back
        dx = dx.reshape(-1, num_paths * num_cols)

    return ans


class _DateTimeIndexInfo:
    def __init__(self, x: Optional[Any] = None):
        index = _get_date_time_index(x)
        if index is None:
            self.has_index_ = False
            self.name_ = None
            self.min_ = None
            self.max_ = None
            self.freq_ = None
        else:
            self.has_index_ = True
            self.name_ = index.name
            self.min_ = index.min()
            self.max_ = index.max()
            self.freq_ = pd.infer_freq(x.index)


class SimBase(ABC):
    def __init__(self, *args, **kwargs):
        # Info about the data used for fitting
        self.fit_container_dtype_ = None
        self.fit_num_rows_ = None
        self.fit_num_cols_ = None

        self.fit_index_ = _DateTimeIndexInfo()
        self.fit_column_names_ = None

        # Additional info from the fit() call we might need later
        self.fit_dt_ = None
        self.fit_x0_ = None
        self.fit_xn_ = None

        # defaults, sometimes overrules in derived classs
        self.x0_default = 0.0

    def _preprocess_fit_x_and_dt(
        self, x: Union[list, np.ndarray, pd.DataFrame, pd.Series], dt: Optional[Union[float, int]]
    ) -> Tuple[np.ndarray, float]:
        r"""Inspect and normalizer the X and dt value provided to the fit function."""

        self.fit_container_dtype_ = type(x)
        self.fit_column_names_ = _get_column_names(x)
        self.fit_index_ = _DateTimeIndexInfo(x)

        values = _to_numpy(x)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.fit_num_cols_ = values.shape[1]
        self.fit_x0_ = values[0, :].reshape(1, -1)
        self.fit_xn_ = values[-1, :].reshape(1, -1)
        self.fit_num_rows_ = values.shape[0]

        # if dt is not provided then we need to infer it
        if dt is None:
            if self.fit_index_.freq_ is not None:
                dt = time_series_freq_to_duration(self.fit_index_.freq_)
                if dt is None:
                    raise ValueError("Unable to determine dt based on freq", self.fit_index_.freq_)
            else:
                dt = 1 / 252  # Defualt
        self.fit_dt_ = dt

        return values, dt

    def _preprocess_sim_path_args(
        self,
        x0: Optional[Union[float, int, list, np.ndarray, pd.DataFrame, pd.Series, str]] = None,
        dt: Optional[Union[float, int]] = None,
        label_start=None,
        label_freq: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, Any, str]:

        need_datetime_index = (label_start is not None) or (label_freq is not None) or (self.fit_index_.has_index_)

        is_fitted = self.fit_x0_ is not None

        # Defaults for x0
        if x0 is None:
            if is_fitted:
                x0 = self.fit_x0_
                if need_datetime_index and (label_start is None):
                    label_start = self.fit_index_.min_
            else:
                x0 = np.array([[self.x0_default]])
        elif isinstance(x0, str):
            print(x0, is_fitted, need_datetime_index, label_start)
            if x0 == "first":
                if not is_fitted:
                    raise ValueError('x0: "first" can not be used because the model is not fitted.')
                x0 = self.fit_x0_
                if need_datetime_index and (label_start is None):
                    label_start = self.fit_index_.min_
            elif x0 == "last":
                if not is_fitted:
                    raise ValueError('x0: "last" can not be used because the model is not fitted.')
                x0 = self.fit_xn_
                if need_datetime_index and (label_start is None):
                    label_start = self.fit_index_.max_
            else:
                raise ValueError(f'x0: Unknown string value "{x0}", valid string values are "first" or "last".')
        else:
            x0 = _to_numpy(x0)

        x0 = x0.reshape(1, -1)

        # if dt not t is provided then align dt with the fitting()
        if dt is None:
            if is_fitted:
                dt = self.fit_dt_
            else:
                dt = 1.0 / 252.0
        assert dt is not None
        assert dt > 0

        # is freq is missing, use the freq we saw while fitting
        if need_datetime_index and (label_freq is None):
            label_freq = self.fit_index_.freq_

        return x0, dt, label_start, label_freq

    def _make_columns_names(self, num_target_columns: int = 1, num_paths: int = 1, columns: Optional[List[str]] = None):
        num_base_columns = int(num_target_columns // num_paths)

        assert num_target_columns == (num_paths * num_base_columns)

        if columns is not None:
            assert len(columns) == num_base_columns
            base_columns = columns
        elif self.fit_column_names_:
            assert len(self.fit_column_names_) == num_base_columns
            base_columns = self.fit_column_names_
        else:
            if num_base_columns > 1:
                base_columns = [f"S{i}" for i in range(num_base_columns)]
            else:
                base_columns = ["S"]

        if num_paths == 1:
            return base_columns

        return [f"{c}_{i}" for i in range(num_paths) for c in base_columns]

    def _make_date_time_index(self, num_rows: int, label_start: Optional[str], label_freq: Optional[str]):
        if label_start is None:
            label_start = self.fit_index_.min_

        if label_freq is None:
            label_freq = self.fit_index_.freq_

        assert label_start is not None
        assert label_freq is not None

        return pd.date_range(start=label_start, freq=label_freq, periods=num_rows, name=self.fit_index_.name_)

    def _format_ans(
        self,
        ans: np.ndarray,
        label_start: Optional[str],
        label_freq: Optional[str],
        columns: Optional[List[str]] = None,
        include_x0: bool = True,
        num_paths: int = 1,
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:

        need_date_time_index = (self.fit_index_.has_index_) or (label_start is not None) or (label_freq is not None)

        need_columns = (
            need_date_time_index
            or (self.fit_container_dtype_ is pd.Series)
            or (self.fit_container_dtype_ is pd.DataFrame)
            or (columns is not None)
        )

        if need_date_time_index:
            index = self._make_date_time_index(ans.shape[0], label_start, label_freq)

        if need_columns:
            columns = self._make_columns_names(ans.shape[1], num_paths, columns)

        if need_date_time_index or need_columns:
            # Return a Series is we have 1 column and didn't fit()
            # or, if we fitted with a Series
            if (len(columns) == 1) and (
                (self.fit_container_dtype_ is pd.Series) or (self.fit_container_dtype_ is None)
            ):
                ans = pd.Series(data=ans.flatten(), index=index, name=columns[0])
            # in all other cases return a pandas DataFrame
            else:
                ans = pd.DataFrame(data=ans, columns=columns, index=index)

        # Return ans, potentially strip away the first
        if include_x0:
            return ans
        else:
            if isinstance(ans, (pd.DataFrame, pd.Series)):
                return ans.iloc[1:, :]
            else:
                return ans[1:, :]

    @staticmethod
    def set_x0(ans: np.ndarray, x0: np.ndarray):
        x0 = np.asarray(x0).reshape(1, -1)
        ans[0, :] = np.tile(x0, ans.shape[1] // x0.shape[1])

    @abstractmethod
    def _path_sample_np(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def path_sample(
        self,
        x0: Optional[Union[float, np.ndarray, pd.DataFrame, pd.Series, str]] = None,
        dt: Optional[float] = None,
        num_steps: Optional[int] = 252,
        num_paths: Optional[int] = 1,
        label_start=None,
        label_freq: Optional[str] = None,
        columns: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        include_x0: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        r"""Simulates random paths.

        Parameters
        ----------
        x0 : Optional[Union[float, np.ndarray, pd.DataFrame, pd.Series, str]], optional
            The initial value(s) of the paths (default is None). The strings "first" and "last"
            will set x0 to the first or last value of the datasets used in a `fit()` call.
        dt : Optional[float], optional
            The time step between observations (default is None). If **dt** is not specfied a values
            will be picked based on the bollowing fall-backs:
            * if **label_freq** is specfied **dt** will based on that
            * if the model is fitted with **fit()** the value of **dt** used during fitting is used
            * else **dt** will default to 1 / 252.
        num_steps : Optional[int], optional
            The number of time steps to simulate (default is 252).
        num_paths : Optional[int], optional
            The number of paths to simulate (default is 1).
        label_start : optional, date-time like.
            The date-time start label for the simulated paths (default is None). When set, this
            function will return Pandas DataFrame with a DateTime index.
        label_freq : Optional[str], optional
            The frequency for labeling the time steps (default is None).
        columns : Optional[List[str]], optional
            A list of column names.
        random_state : Optional[int], optional
            The seed for the random number generator (default is None).
        include_x0 : bool, optional
            Whether to include the initial value in the simulated paths (default is True).

        Returns
        -------
        Union[np.ndarray, pd.DataFrame, pd.Series]
            The simulated random paths.

        """
        # handle arg defaults
        x0, dt, label_start, label_freq = self._preprocess_sim_path_args(x0, dt, label_start, label_freq)

        # do the sims using the actual implementation in the base class
        ans = self._path_sample_np(x0, dt, num_steps, num_paths, random_state)

        # format the ans
        ans = self._format_ans(ans, label_start, label_freq, columns, include_x0, num_paths)

        return ans

    @abstractmethod
    def _fit_np(self, x: np.ndarray, dt: float):
        raise NotImplementedError

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series], dt: Optional[float], **kwargs):
        r"""Calibrates the model to the given path(s).

        Parameters
        ----------
        x : Union[np.ndarray, pd.DataFrame, pd.Series]
            The input data for calibration.
        dt : Optional[float]
            The time step between observations.
        **kwargs
            Additional arguments for the fit process.

        Returns
        -------
        self : Self
            The fitted model instance.

        """
        values, dt = self._preprocess_fit_x_and_dt(x, dt)
        self._fit_np(values, dt)
        return self
