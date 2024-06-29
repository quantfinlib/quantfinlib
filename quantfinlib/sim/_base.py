
from typing import Union, Optional
import pandas as pd
import numpy as np
from quantfinlib._datatypes.timeseries import time_series_freq_to_duration


class SimHelperBase:
    def __init__(self, *args, **kwargs):
        # Info about the data used for fitting        
        self.fit_container_dtype_ = None

        self.fit_num_rows_ = None
        self.fit_num_cols_ = None

        self.fit_index_name_ = None
        self.fit_index_min_ = None
        self.fit_index_max_ = None
        self.fit_index_freq_ = None

        self.fit_column_names_ = None
        
        # Additional info from the fit() call we might need later
        self.fit_dt_ = None
        self.fit_x0_ = None
        self.fit_xn_ = None

    def inspect_and_normalize_fit_args(
        self,
        x: Union[list, np.ndarray, pd.DataFrame, pd.Series],
        dt: Optional[Union[float, int]]):

        self.fit_container_dtype_ = type(x)

        # Collect and remember DatetimeIndex data if available
        if isinstance(x, (pd.DataFrame, pd.Series)):
            values = x.to_numpy()
            if isinstance(x, pd.Series):
                self.fit_column_names_ = [x.name]
            else: # isinstance(x, pd.DataFrame)
                self.fit_column_names_ = x.columns.values.tolist()
            if isinstance(x.index, pd.DatetimeIndex):
                self.fit_index_name_ = x.index.name
                self.fit_index_min_ = x.index.min()
                self.fit_index_max_ = x.index.max()
                self.fit_index_freq_ = pd.infer_freq(x.index)
        elif isinstance(x, list):
            values = np.array(x)
        else: # np.ndarray
            values = x

        if values.ndim == 1:
            values = values.reshape(-1, 1)
        
        # if dt is not provided then we need to infer it
        if dt is None:
            if self.fit_index_freq_ is not None:
                dt = time_series_freq_to_duration(self.fit_index_freq_)
                if dt is None:
                    raise ValueError("Unable to determine dt based on freq", self.fit_index_freq_)
            else:
                raise ValueError("Unable to infer dt")

        # validation
        assert values.ndim == 2
        assert values.shape[0] >= 3
        assert dt > 0.0

        # Save fitting information
        self.fit_dt_ = dt
        self.fit_num_rows_ = values.shape[0]
        self.fit_num_cols_ = values.shape[1]
        self.fit_x0_ = values[0, :].reshape(1, -1)
        self.fit_xn_ = values[-1, :].reshape(1, -1)

        return values, dt

    def normalize_sim_path_args(
        self,
        x0: Optional[Union[float, int, list, np.ndarray, pd.DataFrame, pd.Series, str]] = None,
        dt: Optional[Union[float, int]] = None,
        label_start = None,
        label_freq: Optional[str] = None,
        x0_default: float = 0.0 
        ):

        need_labels = (
            (self.fit_index_min_ is not None) 
            or (label_start is not None)
            or (label_freq is not None)
        )

        is_fitted = (self.fit_x0_ is not None)

        # Defaults for x0
        if x0 is None:
            if is_fitted:
                x0 = self.fit_x0_
                if need_labels and (label_start is None):
                    label_start = self.fit_index_min_
            else:
                x0 = np.array([[x0_default]])
        elif isinstance(x0, str):
            if x0 == "first":
                if not is_fitted:
                    raise ValueError('x0: "first" can not be used because the model is not fitted.')
                x0 = self.fit_x0_
                if need_labels and (label_start is None):
                    label_start = self.fit_index_min_
            elif x0 == "last":
                if not is_fitted:
                    raise ValueError('x0: "last" can not be used because the model is not fitted.')            
                x0 = self.fit_xn_
                if need_labels and (label_start is None):
                    label_start = self.fit_index_max_
            else:
                raise ValueError(f'x0: Unknown string value "{x0}", valid string values are "first" or "last".')
        elif isinstance(x0, float):
            x0 = np.array([[x0]])
        elif isinstance(x0, int):
            x0 = np.array([[x0]], dtype=float)
        elif isinstance(x0, list):
            x0 = np.array(x0)
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
        if need_labels and (label_freq is None):
            label_freq = self.fit_index_freq_

        return x0, dt, label_start, label_freq
    
    def format_ans(
        self,
        ans: np.ndarray,
        label_start: Optional[str],
        label_freq: Optional[str],
        include_x0: bool = True
        ):

        need_labels = (
            (self.fit_index_min_ is not None) 
            or (label_start is not None)
            or (label_freq is not None)
        )

        is_fitted = (self.fit_x0_ is not None)

        
        # configure the return container
        if need_labels:
            assert label_start is not None
            assert label_freq is not None
            index = pd.date_range(
                start=label_start, 
                freq=label_freq, 
                periods=num_num_steps + 1,
                name=self.fit_index_name_
            )

            # Return a series if we fitted with a series, and if we only have 1 scenario
            if isinstance(self.fit_container_dtype_, pd.Series) and (ans.shape[1] == 1):
                ans = pd.Series(
                    data = ans,
                    index = index,
                    name = self.fit_column_names_[0] + '_0'
                )
            else:
                columns = [f"{c}_{i}" for i in range(num_paths) for c in self.fit_column_names_]

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

    @staticmethod
    def set_x0(ans: np.ndarray, x0: np.ndarray):
        x0 = x0.reshape(1, -1)
        ans[0, :] = np.tile(x0, ans.shape[1] // x0.shape[1])
