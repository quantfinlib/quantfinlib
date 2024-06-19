from typing import Union

import numpy as np
import pandas as pd


def num_columns(x: Union[list, np.ndarray, pd.DataFrame, pd.Series]) -> int:
    """Estimate the number of columns in a data object.

    Parameters
    ----------
    x : Union[list, np.ndarray, pd.DataFrame, pd.Series]
        The object for which we want to estimate the number of columns.

    Returns
    -------
    : int
        The number of columns is estimated as follows:

            * list: 1
            * pd.Series: 1
            * pd.DataFrame: number of columns
            * 1d np.ndarray: 1
            * n-d np.ndarray: length of the 2nd dimension
            * other object: num_columns of the np.ndarray when the objected is casted to a np.ndarray using np.asarray.

    """
    if isinstance(x, pd.Series):
        return 1
    elif isinstance(x, pd.DataFrame):
        return len(x.columns)
    elif isinstance(x, np.ndarray):
        if x.size == 0:
            return 0
        if x.ndim == 1:
            return 1
        elif x.ndim == 2:
            return x.shape[1]
        else:
            return x.reshape(1, -1).shape[1]
    else:
        return num_columns(np.asarray(x))


def num_rows(x: Union[list, np.ndarray, pd.DataFrame, pd.Series]):
    """Estimate the number of rows in a data object.

    Parameters
    ----------
    x : Union[list, np.ndarray, pd.DataFrame, pd.Series]
        The object for which we want to estimate the number of rows.

    Returns
    -------
    : int
        The number of rows is estimated as follows:

            * list: the length
            * pd.Series: number of rows
            * pd.DataFrame: number of rows
            * 1d np.ndarray: length of the array
            * n-d np.ndarray: length of the first dimension
            * other object: num_rows of the np.ndarray when the objected is casted to a np.ndarray using np.asarray.

    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return len(x)
    elif isinstance(x, np.ndarray):
        if x.size == 0:
            return 0

        if x.ndim >= 1:
            return x.shape[0]
    else:
        return num_rows(np.asarray(x))
