"""
Brownian Motion simulation

Classes in this module:

BrownianMotion()
"""

__all__ = ["BrownianMotion"]

from typing import Union, Optional
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from quantfinlib._datatypes.timeseries import time_series_freq_to_duration


class BrownianMotionBase(BaseEstimator):
    """Brownian Motion base class"""

    def __init__(self, drift=0.0, vol=0.1, cor=None):

        # Parameters
        self.drift = np.asarray(drift).reshape(1, -1)
        self.vol = np.asarray(vol).reshape(1, -1)
        self.cor = None

        # Private attributes
        if self.cor is None:
            self.L_ = None
        else:
            self.L_ = np.linalg.cholesky(self.cor)

        # Info about the data used for fitting        
        self.fit_dt_ = None
        self.fit_num_rows_ = None
        self.fit_num_cols_ = None
        self.fit_x0_ = None
        self.fit_xn_ = None
        
    def fit(self, x: np.ndarray, dt: float):
        
        # Preprocessing arguments
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # validation
        assert x.ndim == 2
        assert x.shape[0] >= 3
        assert dt > 0.0
        
        # Save fitting information
        self.fit_dt_ = dt
        self.fit_num_rows_ = x.shape[0]
        self.fit_num_cols_ = x.shape[1]
        self.fit_x0_ = x[0, :].reshape(1, -1)
        self.fit_xn_ = x[-1, :].reshape(1, -1)
        
        # changes from one row to the next
        dx = np.diff(x, axis=0)

        # Mean and standard deviation of the changes
        self.drift = np.mean(dx, axis=0, keepdims=True) / dt
        self.vol = np.std(dx, axis=0, ddof=1, keepdims=True) / np.sqrt(dt)

        # Optionally correlations if we have multiple columns
        if x.shape[1] > 1:
            self.cor = np.corrcoef(dx, rowvar=False)
            self.L_ = np.linalg.cholesky(self.cor)
        else:
            self.cor = None
            self.L_ = None

        return self

    def sim_path(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None
    ) -> np.ndarray:
            
        # validation
        assert dt > 0
        assert x0.ndim == 2 # a matrix
        assert x0.shape[0] == 1 # a row
        assert x0.shape[1] == self.drift.shape[1]

        # Allocate storage for the simulation
        num_cols = self.drift.shape[1]
        ans = np.zeros(shape=(num_steps + 1, num_cols * num_paths))

        # set the initial value
        if x0.shape[1] == ans.shape[1]:
            ans[0, :] = x0
        elif x0.shape[1] == self.drift.shape[1]:
            ans[0, :] = np.repeat(x0, num_paths, axis=1)
        elif x0.shape[1] == 1:
            ans[0, :] = x0
        else:
            raise ValueError(f'The shape of x0 {x0.shape} is incompatible with the target output {ans.shape}')
        
        # Create a Generator instance with the seed
        rng = np.random.default_rng(random_state)

        # fill in Normal noise
        ans[1 : num_steps + 1, :] = rng.normal(size=(num_steps, ans.shape[1]))

        tmp = ans[1 : num_steps + 1, :]
        tmp = tmp.reshape(-1, num_paths, num_cols)

        # Optionally correlate the noise
        if self.L_ is not None:
            tmp = tmp @ self.L_.T

        # Translate the noise with drift and variance
        tmp = tmp * self.vol * dt**0.5 + self.drift * dt

        # reshape back
        ans[1 : num_steps + 1, :] = tmp.reshape(-1, num_paths * num_cols)

        # compound
        ans = np.cumsum(ans, axis=0)
        
        return ans



class BrownianMotion(BrownianMotionBase):
    """Brownian Motion simulation"""

    def __init__(self, drift=0.0, vol=0.1, cor=None):
        super().__init__(drift, vol, cor)

        self.fit_container_dtype_ = None
        self.fit_index_min_ = None
        self.fit_index_max_ = None
        self.fit_index_freq_ = None
        self.fit_dt_ = None

    def fit(
        self,
        x: Union[np.ndarray, pd.DataFrame, pd.Series],
        dt: Optional[float],
        **kwargs
    ):
        # 
        self.fit_container_dtype_ = type(x)
        
        # Collect and remember DatetimeIndex data if available
        # also make "value" a numpy array
        if isinstance(x, (pd.DataFrame, pd.Series)):
            values = x.to_numpy()
            if isinstance(x.index, pd.DatetimeIndex):
                self.fit_index_min_ = x.index.min()
                self.fit_index_max_ = x.index.max()
                self.fit_index_freq_ = pd.infer_freq(x.index)
        else: # np.ndarray
            values = x
        
        # if dt is not provided then we need to infer it
        if dt is None:
            if self.index_freq_ is not None:
                dt = time_series_freq_to_duration(self.index_freq_)
            else:
                raise ValueError("Unable to determine dt")
        self.dt_ = dt
       
        # do the actual fitting in the base class
        super().fit(values, dt)
        return self

    def sim_path(
        self,
        x0: Optional[Union[float, np.ndarray, pd.DataFrame, pd.Series, str]] = None,
        dt: Optional[float] = None,
        num_steps: Optional[int] = 252,
        num_paths: Optional[int] = 1,
        label_start = None,
        label_freq: Optional[str] = None,
        random_state: Optional[int] = None,
        include_x0: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:

        # Are we going to attempt to return a pandas object with DataTimeIndex?
        # we will try this if 
        # 1) we had a DateTimeIndex in the fit(x) dataset
        # 2) if one of of label_start or label_freq was set in this call
        return_datetime_index = (
            (self.fit_index_min_ is not None) 
            or (label_start is not None)
            or (label_freq is not None)
        )

        is_fitted = (self.fit_x0_ is not None)

        # Defaults for x0
        if x0 is None:
            if is_fitted:
                x0 = self.fit_x0_
                if return_datetime_index and (label_start is None):
                    label_start = self.fit_index_min_
            else:
                x0 = np.array([[0.0]])
        elif x0 == "first":
            if not is_fitted:
                raise ValueError('x0: "first" can not be used because the model is not fitted.')
            x0 = self.fit_x0_
            if return_datetime_index and (label_start is None):
                label_start = self.fit_index_min_
        elif x0 == "last":
            if not is_fitted:
                raise ValueError('x0: "last" can not be used because the model is not fitted.')            
            x0 = self.fit_xn_
            if return_datetime_index and (label_start is None):
                label_start = self.fit_index_max_
        elif isinstance(x0, str):
            raise ValueError(f'x0: Unknown string value "{x0}", valid string values are "first" or "last".')
        elif isinstance(x0, float):
            x0 = np.ndarray([[x0]])
        elif isinstance(xp0, list):
            x0 = np.ndarray(x0)
        elif isinstance(x0, (pd.DataFrame, pd.Series)):
            x0 = x0.to_numpy()
        assert isinstance(x0, np.ndarray)

        x0 = x0.reshape(1, -1)

        # if dt not t is provided then align dt with the fitting()
        if dt is None:
            if is_fitted:
                dt = self.fit_dt_
            else:
                dt = 1.0/252.0
        assert (dt is not None)
        assert (dt > 0)

        # is freq is missing, use the freq we saw while fitting
        if return_datetime_index and (label_freq is None):
            label_freq = self.fit_index_freq_
        
        # do the sims
        ans = super().sim_path(x0, dt, num_steps, num_paths, random_state)

        # configure the return container
        if return_datetime_index:
            assert label_start is not None
            assert label_freq is not None
            index = pd.date_range(start=label_start, freq=label_freq, periods=num_num_steps + 1)

            # should we return a series?
            if isinstance(self.fit_container_dtype_, pd.Series) and (ans.shape[1] == 1):
                ans = pd.Series(
                    data = ans,
                    index = index,
                    name = self.fit_column_names_[0] + '_0'
                )
            else:
                columns = []
                for i in range(num_paths):
                    for c in self.fit_column_names_:
                        columns.append(f'{c}_{i}')
                ans = pd.DataFrame(
                    data = ans,
                    index= index,
                    columns = columns
                )

        # Return ans, potentially strip away the first
        if include_x0:
            return ans
        else:
            if isinstance(ans, (pd.DataFrame, pd.Series)):
                return ans.iloc[1:, :]
            else:
                return ans[1:, :]

        return ans
