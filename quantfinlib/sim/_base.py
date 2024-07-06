"""
File: quantfinlib/sim/_base.py

Description:
    Private base classes and functions used in the sim module.

Author:    Thijs van den Berg
Email:     thijs@sitmo.com
Copyright: (c) 2024 Thijs van den Berg
License:   MIT License
"""
from typing import Any, List, Optional, Tuple, Union, Dict
import warnings
from abc import ABC, abstractmethod
import math

import numpy as np
import pandas as pd

from quantfinlib._datatypes.timeseries import time_series_freq_to_duration

_SimFitDataType = Union[int, float, list, np.ndarray, pd.DataFrame, pd.Series]

def _count_non_none_elements(tup: Tuple) -> int:
    return sum(1 for item in tup if item is not None)


def _count_non_none_dict_values(d: Dict) -> int:
    return sum(1 for value in d.values() if value is not None)


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


def _get_datetime_index(x: Optional[Any]) -> Optional[pd.DatetimeIndex]:
    if not isinstance(x, (pd.DataFrame, pd.Series)):
        return None
    if not isinstance(x.index, pd.DatetimeIndex):
        return None
    return x.index

def _make_datetime_index(
    num_rows:int,    
    label_start: Optional[Any] = None,
    label_end: Optional[Any] = None,
    label_freq: Optional[Any] = None,
    fallback_start: Optional[Any] = None,
    fallback_freq: Optional[Any] = None,
    name: Optional[str] = None
) -> Optional[pd.DatetimeIndex]:

    date_range_args = {
        'start': label_start,
        'freq': label_freq,
        'end': label_end
    }

    num_vars_set = _count_non_none_dict_values(date_range_args)

    if num_vars_set == 3:
        raise ValueError('Cannot create a DatetimeIndex when all 3 [label_start, label_end, label_freq] are set.')

    if num_vars_set == 2:
        return pd.date_range(**date_range_args, periods=num_rows, name=name)

    fallbacks = [
        ('freq', fallback_freq),
        ('start', fallback_start),
    ]
    
    for patch_key, patch_value in fallbacks:
        if date_range_args[patch_key] is None:
            date_range_args[patch_key] = patch_value
        num_vars_set = _count_non_none_dict_values(date_range_args)
        if num_vars_set == 2:
            break

    if num_vars_set == 2:
        return pd.date_range(**date_range_args, periods=num_rows, name=name)

    raise ValueError('Cannot create a DatetimeIndex, not enough information.')
    

def _fill_with_correlated_noise(
    ans: np.ndarray,
    loc: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
):

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


def _triangular_index(T_n: int) -> int:
    # Calculate the potential index using the derived formula
    n = (-1 + math.sqrt(1 + 8 * T_n)) / 2

    # Check if n is an integer
    if n.is_integer():
        return int(n)
    else:
        raise ValueError("The given number is not a triangular number")


def _make_cor_from_upper_tri(values: _SimFitDataType) -> np.ndarray:
    print("_make_cor_from_upper_tri", values)

    values = _to_numpy(values)

    # We support 1 list, or a row/col from a matrix, as long as it's effectively 1d
    assert values.ndim == 2
    if (values.shape[0] == 1) or (values.shape[1] == 1):
        values = values.flatten()
    else:
        raise ValueError(f"Expected a 1d list of numbers but got {values.shape}")

    # Compute the dimensions of the output matrix
    num_dims = _triangular_index(len(values)) + 1

    # Initialize an n x n matrix with zeros
    matrix = np.zeros((num_dims, num_dims))

    # Fill the upper triangle (including the diagonal)
    index = 0
    for i in range(num_dims):
        matrix[i, i] = 1.0
        for j in range(i + 1, num_dims):
            matrix[i, j] = values[index]
            matrix[j, i] = values[index]
            index += 1

    return matrix


def _estimate_dt_from_time_intervals(first:pd.Timestamp, last: pd.Timestamp, num_rows: int) -> float:
    # Calculate the total duration between the first and last dates
    total_duration = last - first

    # Calculate the average duration in days
    average_duration_days = total_duration / num_rows

    # Convert the average duration from days to years (considering 365.25 days per year for leap years)
    average_duration_years = average_duration_days / pd.Timedelta(days=365.25)

    # Get the float value
    average_duration_years_float = average_duration_years.total_seconds() / pd.Timedelta(days=365.25).total_seconds()

    return average_duration_years_float


class SimBase(ABC):
    def __init__(self, *args, **kwargs):

        # defaults, sometimes overrules in derived classs
        self.x0_default = 0.0
        self.num_parameters_ = 2

    def _inspect_and_normalize_fit_x(self, x):
        
        # Info about the data used for fitting
        self.fit_container_dtype_ = type(x)

        # Info about the content of x
        np_x = _to_numpy(x)
        self.fit_num_rows_ = np_x.shape[0]
        self.fit_num_cols_ = np_x.shape[1]
        self.fit_x0_ = np_x[0, ...]
        self.fit_xn_ = np_x[-1, ...]

        # info about the potential DatetimeIndex of x
        index = _get_datetime_index(x)
        if index is not None:
            self.fit_has_date_time_index_ = True
            self.fit_index_name_ = index.name
            self.fit_index_min_ = index.min()
            self.fit_index_max_ = index.max()
            self.fit_index_freq_ = pd.infer_freq(index)
            self.fit_index_rows_ = len(index)
            self.fit_index_dt_ = _estimate_dt_from_time_intervals(
                self.fit_index_min_, 
                self.fit_index_max_, 
                self.fit_index_rows_
            )            

        # info about the potential column names of x
        self.fit_column_names_ = _get_column_names(x)

        return np_x

    def _preprocess_path_sample_x0(
        self,
        x0: Optional[Union[float, int, list, np.ndarray, pd.DataFrame, pd.Series, str]] = None
    ) -> Tuple[np.ndarray, Any]:

        # Check if the model was fitted, if so then we can use default values from that
        is_fitted = hasattr(self, 'fit_num_rows_')

        if x0 is None:
            if is_fitted:
                return self.fit_x0_,  self.fit_index_.min_
            else:
                return np.array([[self.x0_default]]), None
        
        if x0 == "first":
            if not is_fitted:
                raise ValueError('x0: "first" can not be used because the model is not fitted.')
            return self.fit_x0_,  self.fit_index_.min_

        if x0 == "last":
            if not is_fitted:
                raise ValueError('x0: "last" can not be used because the model is not fitted.')
            return self.fit_xn_,  self.fit_index_.max_

        # Any other string is wrong
        if isinstance(x0, str):
            raise ValueError(f'x0: Unknown string value "{x0}", valid string values are "first" or "last".')

        # Defaults for x0, all other cases, we have a non-string value, try to convert it to a numpy npdarray       
        return _to_numpy(x0), None


    def _make_columns_names(self, num_target_columns: int = 1, num_paths: int = 1, columns: Optional[List[str]] = None):
        num_base_columns = int(num_target_columns // num_paths)

        assert num_target_columns == (num_paths * num_base_columns)

        if columns is not None:
            assert len(columns) == num_base_columns
            base_columns = columns
        elif hasattr(self, 'fit_column_names_'): 
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

        # Process x0
        x0_clean, x0_clean_label_start = self._preprocess_path_sample_x0(x0)

        # Condition that determines if we need to attach a DatetimeIndex to the return value
        # - when the user specify at lesat one of [label_start, label_start, label_freq]
        # - when we had a DatetimeIndex during fitting
        need_datetime_index = (_count_non_none_elements((label_start, label_start, label_freq)) > 0) or hasattr(self, 'fit_has_date_time_index_')
        
        # Create the index
        if need_datetime_index:

            index = _make_datetime_index(
                num_rows=ans.shape[0],    
                label_start=label_start,
                label_end=label_end,
                label_freq=label_freq,
                fallback_start=x0_clean_label_start,
                fallback_freq=self.fit_index_freq_,
                name=self.get('fit_index_name_', None)
            )
  
            # Compute the average timestep in this index
            index_dt = _estimate_dt_from_time_intervals(first=index.min(), last=index.max(), num_rows=len(index))
            
            # Use the dt of the index if the users hasn't privided a dt
            if dt is None:
                dt = index_dt
            
            # Check that the dt used for the simulation and dt of the index don't deviate too much
            if abs(dt - index_dt) / index_dt > 0.01:
                warnings.warn(
                    f"The simulation timestep {dt} and the index label timestep {index_dt} differ more than 1%",
                    UserWarning,
                )

        # Condition that determines if we need column names
        need_columns = need_datetime_index or (columns is not None)
        if hasattr(self, 'fit_container_dtype_'):
            need_columns = need_columns or (self.fit_container_dtype_ is pd.Series)
            need_columns = need_columns or (self.fit_container_dtype_ is pd.DataFrame)

        # Create the columnn names
        if need_columns:
            columns = self._make_columns_names(ans.shape[1], num_paths, columns)

        # --------------------------------------------------------------------------------------
        # do the actual sims using the implementation in the derived class
        # --------------------------------------------------------------------------------------
        ans = self._path_sample_np(x0_clean, dt, num_steps, num_paths, random_state)

        # Now convert the ans numpy array to a target output container
        if need_datetime_index or need_columns:

            # Return a Series is we have 1 column and didn't fit()
            # or, if we fitted with a Series
            if (len(columns) == 1) and (
                (self.fit_container_dtype_ is pd.Series) or (self.fit_container_dtype_ is None)
            ):
                ans = pd.Series(data=ans.flatten(), index=index, name=columns[0])
            
            # in all other cases return a pandas DataFrame
            else:
                ans = pd.DataFrame(data=ans, columns=columns, index=index)

        # Return ans, potentially strip away the first row
        if include_x0:
            return ans
        else:
            if isinstance(ans, (pd.DataFrame, pd.Series)):
                return ans.iloc[1:, :]
            else:
                return ans[1:, :]


    @abstractmethod
    def _fit_np(self, x: np.ndarray, dt: float):
        raise NotImplementedError

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series], dt: Optional[float] = None, **kwargs):
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

        # Convert x to a 2d numpy array and store collected information about the x container in self.fit_* attributes 
        values = self._inspect_and_normalize_fit_x(x)

        # If dt was not passed then maybe we can use an estimate based on a DatetimeIndex of x?
        if dt is None:
            dt = self.fit_index_dt_
            if dt is None:
                raise ValueError('Unable to fit because we dont know dt. dt ws not provided to the fit() function, and we cant infer dt from a time index of x')
        
        # We save the final dt we use in the fit
        self.fit_dt_ = dt

        # The actual fit
        self._fit_np(values, dt)
        return self


class SimNllMixin:

    def nll(self, x: Union[np.ndarray, pd.DataFrame, pd.Series], dt: Optional[float] = None) -> float:
        r"""Calculate the negative log-likelihood (lower is better) for a given path.

        Parameters
        ----------
        x : Union[np.ndarray, pd.DataFrame, pd.Series]
            The input data for negative log-likelihood calculation.
        dt : Optional[float]
            The time step between observations.

        Returns
        -------
        nll : float
            The computed negative log-likelihood.

        """
        # Convert x to a 2d numpy array and store collected information about the x container in self.fit_* attributes 
        values = self._inspect_and_normalize_fit_x(x)

        # If dt was not passed then maybe we can use an estimate based on a DatetimeIndex of x?
        if dt is None:
            dt = getattr(self, 'fit_index_dt_', None)
            if dt is None:
                raise ValueError('Unable to compute NLL because we dont know dt. dt ws not provided to the fit() function, and we cant infer dt from a time index of x')
        
        return self._nll(values, dt)

    def aic(self, x: Union[np.ndarray, pd.DataFrame, pd.Series], dt: Optional[float] = None) -> float:
        r"""Calculate the Akaike Information Criterion (AIC) for a given path.

        Parameters
        ----------
        x : Union[np.ndarray, pd.DataFrame, pd.Series]
            The input data for AIC calculation.
        dt : Optional[float]
            The time step between observations.

        Returns
        -------
        aic : float
            The computed Akaike Information Criterion (AIC) value.

        """
        # Convert x to a 2d numpy array and store collected information about the x container in self.fit_* attributes 
        values = self._inspect_and_normalize_fit_x(x)

        # If dt was not passed then maybe we can use an estimate based on a DatetimeIndex of x?
        if dt is None:
            dt = getattr(self, 'fit_index_dt_', None)
            if dt is None:
                raise ValueError('Unable to compute NLL because we dont know dt. dt ws not provided to the fit() function, and we cant infer dt from a time index of x')
        

        num_samples = values.shape[0]
        return 2 * self.num_parameters_ + 2 * self._nll(values, dt) * num_samples

    def bic(self, x: Union[np.ndarray, pd.DataFrame, pd.Series], dt: Optional[float] = None) -> float:
        r"""Calculate the Bayesian Information Criterion (BIC) for a given path.

        Parameters
        ----------
        x : Union[np.ndarray, pd.DataFrame, pd.Series]
            The input data for BIC calculation.
        dt : Optional[float]
            The time step between observations.

        Returns
        -------
        bic : float
            The computed Bayesian Information Criterion (BIC) value.
        """
        # Convert x to a 2d numpy array and store collected information about the x container in self.fit_* attributes 
        values = self._inspect_and_normalize_fit_x(x)

        # If dt was not passed then maybe we can use an estimate based on a DatetimeIndex of x?
        if dt is None:
            dt = getattr(self, 'fit_index_dt_', None)
            if dt is None:
                raise ValueError('Unable to compute NLL because we dont know dt. dt ws not provided to the fit() function, and we cant infer dt from a time index of x')
        
        num_samples = values.shape[0]
        return 2 * np.log(num_samples) * self.num_parameters_ + 2 * self._nll(values, dt) * num_samples
