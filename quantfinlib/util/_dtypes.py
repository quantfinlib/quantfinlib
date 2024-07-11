"""Type aliases for data types used in the library."""

from typing import Union
import numpy as np
import pandas as pd


SeriesOrArray = Union[pd.Series, np.ndarray]
DataFrameOrArray = Union[pd.DataFrame, np.ndarray]
