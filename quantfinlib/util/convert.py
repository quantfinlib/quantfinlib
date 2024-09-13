"""Module for converting data structures to numpy arrays and vice versa."""

from typing import Union

import numpy as np
import pandas as pd


def type_to_np(x: Union[list, np.ndarray, pd.DataFrame, pd.Series], to_float: bool = True) -> np.ndarray:
    r"""Convert a data structure to a numpy array of floats.

    Parameters
    ----------
    x: object
        Source data structure that needs to be converted to a numpy array.
    to_float : bool, default=True
        convert element to float.

    Returns
    -------
    np.ndarray
        Output numpy array.

    Example
    -------
    Converting a list of integers:

    .. exec_code::

        # --- hide: start ---
        import numpy as np
        from quantfinlib.util import type_to_np

        np.random.seed(42)
        np.set_printoptions(precision=2)
        # --- hide: stop ---

        x = [1,2,3,4]
        y = type_to_np(x)

        print('x:\n', x, '\n\ny:\n', y, f'type={type(y)} shape={y.shape} dtype={y.dtype}')

    Example
    -------
    Converting a list of list of integers:

    .. exec_code::

        # --- hide: start ---
        import numpy as np
        from quantfinlib.util import type_to_np

        np.random.seed(42)
        np.set_printoptions(precision=2)
        # --- hide: stop ---

        x = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        y = type_to_np(x)

        print('x:\n', x, '\n\ny:\n', y, f'type={type(y)} shape={y.shape} dtype={y.dtype}')

    Example
    -------
    Converting a Pandas Series:

    .. exec_code::

        # --- hide: start ---
        import numpy as np
        import pandas as pd
        from quantfinlib.util import type_to_np

        np.random.seed(42)
        np.set_printoptions(precision=2)
        # --- hide: stop ---

        x = pd.Series([110,112,109,107], name='Prices')
        y = type_to_np(x)

        print('x:\n', x, '\n\ny:\n', y, f'type={type(y)} shape={y.shape} dtype={y.dtype}')


    Example
    -------
    Converting a Pandas DataFrame:

    .. exec_code::

        # --- hide: start ---
        import numpy as np
        import pandas as pd
        from quantfinlib.util import type_to_np

        np.random.seed(42)
        np.set_printoptions(precision=2)
        # --- hide: stop ---

        x = pd.DataFrame([[100,200],[101,199],[101,203],[103,208]], columns=['S&P500', 'DJIA'])
        y = type_to_np(x)

        print('x:\n', x, '\n\ny:\n', y, f'type={type(y)} shape={y.shape} dtype={y.dtype}')
    """
    if to_float:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.values.astype(float)
        return np.asarray(x).astype(float)

    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return np.asarray(x)


def np_to_type(
    x: Union[list, np.ndarray],
    obj: Union[list, np.ndarray, pd.DataFrame, pd.Series],
    columns: Union[list, None] = None,
    pre: str = "",
    post: str = "",
    to_float: bool = True,
) -> Union[list, np.ndarray, pd.DataFrame, pd.Series]:
    """Converts a numpy array to a Pandas Series or DataFrame target data type.

    Parameters
    ----------
    x: Union[list, np.ndarray]
        The numpy array that needs to be converted.
    obj : Union[list, np.ndarray, pd.DataFrame, pd.Series]
        The target data type.
    columns : Union[list, None], default=None
        The column names of the target data type.
    pre : str, default=''
        A prefix that is added to the column names.
    post : str, default=''
        A postfix that is added to the column names.
    to_float : bool, default=True
        Convert the numpy array to float.

    Returns
    -------
        An object of the same type as target_type with the content of x.

    Raises
    ------
    ValueError
        If the target type is not supported.
    """
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    assert x.ndim in (1, 2)
    assert isinstance(obj, (list, np.ndarray, pd.Series, pd.DataFrame))

    assert len(x) == len(obj), f"source rows {len(x)} and target rows {len(obj)} are not the same."
    if columns is not None:
        if not isinstance(columns, list):
            columns = [columns]

    if isinstance(obj, (list, np.ndarray)):
        if to_float:
            return x.astype(float)
        else:
            return x

    # vectors are converted to a 1 col matrix. Everything below assumed that x is a matrix
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if isinstance(obj, pd.Series) and x.shape[1] == 1:
        if columns is None:
            columns = [f"{pre}{obj.name}{post}"]
        return pd.Series(data=x.flatten(), index=obj.index, name=columns[0])

    if isinstance(obj, pd.Series) and x.shape[1] > 1:
        if columns is None:
            columns = [f"{obj.name}{i}" for i in range(x.shape[1])]
        columns = [f"{pre}{c}{post}" for c in columns]
        assert len(columns) == x.shape[1]
        return pd.DataFrame(data=x, index=obj.index, columns=columns)

    if isinstance(obj, pd.DataFrame):
        if columns is None:
            if x.shape[1] == obj.shape[1]:
                columns = obj.columns
            else:
                columns = [f"c{i}" for i in range(x.shape[1])]
        columns = [f"{pre}{c}{post}" for c in columns]
        assert len(columns) == x.shape[1]
        return pd.DataFrame(data=x, index=obj.index, columns=columns)

    raise ValueError(
        f"Unsupported target type {type(obj)}. Possible types are numpy arrays, pandas series or " "pandas dataframes."
    )
