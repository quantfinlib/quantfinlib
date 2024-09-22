from typing import Union 

import numpy as np
import pandas as pd


def _num_columns(x: Union[list, np.ndarray, pd.Series, pd.DataFrame]) -> int:
    """Return the number of columns in the input."""
    if isinstance(x, pd.Series):
        return 1
    elif isinstance(x, pd.DataFrame):
        return x.shape[1]
    elif isinstance(x, np.ndarray):
        if x.size == 0:
            return 0
        if x.ndim == 1:
            return 1
        if x.ndim == 2:
            return x.shape[1]
        else:
            return x.reshape(1, -1).shape[1] 
    else:
        return _num_columns(np.asarray(x))
