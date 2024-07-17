"""Type aliases for data types used in the library."""

from typing import Union
import numpy as np
import pandas as pd


SeriesOrArray = Union[pd.Series, np.ndarray]
DataFrameOrArray = Union[pd.DataFrame, np.ndarray]
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert array-like input to numpy array."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return x
