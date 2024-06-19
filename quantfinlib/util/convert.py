from typing import Union

import numpy as np
import pandas as pd


def try_str_to_num(s: str) -> Union[int, float, bool, str]:
    """Try to convert a string to an int or float. Return the string if that fails.

    Parameters
    ----------
    s: str
        Input string.

    Returns
    -------
    : Union[int, float, bool, str]
        Potentially converted value.
    """
    try:
        return int(str(s))
    except:
        pass
    try:
        return float(str(s))
    except:
        pass
    return s


def type_to_np(
    x: Union[list, np.ndarray, pd.DataFrame, pd.Series], to_float: bool = True
):
    r"""Convert a data structure to a numpy array of floats.

    Example
    -------
    Converting a list of integers:

    .. exec_code::

        # --- hide: start ---
        import numpy as np
        from samcoml.util import type_to_np

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
        from samcoml.util import type_to_np

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
        from samcoml.util import type_to_np

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
        from samcoml.util import type_to_np

        np.random.seed(42)
        np.set_printoptions(precision=2)
        # --- hide: stop ---

        x = pd.DataFrame([[100,200],[101,199],[101,203],[103,208]], columns=['S&P500', 'DJIA'])
        y = type_to_np(x)

        print('x:\n', x, '\n\ny:\n', y, f'type={type(y)} shape={y.shape} dtype={y.dtype}')

    Parameters
    ----------
    x: object
        Source data structure that needs to be converted to a numpy array.
    to_float : bool, default=True
        convert element to float.
    Returns
    -------
    numpy array

    """
    """Tried to convert a data structure to a numpy array."""
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
):
    """Converts a numpy array to a Pandas Series or DataFrame target data type

    Parameters
    ----------
    x: Union[list, np.ndarray]
        The numpy array that needs to be converted.
    obj
    columns
    pre
    post
    to_float

    Returns
    -------
        An object of the same type as target_type with the content of x.
    """
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    assert x.ndim in (1, 2)
    assert isinstance(obj, (list, np.ndarray, pd.Series, pd.DataFrame))

    assert len(x) == len(
        obj
    ), f"source rows {len(x)} and target rows {len(obj)} are not the same."
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
        "Unsupported target type {type(target_obj)}. Possible types are numpy arrays, pandas series or "
        "pandas dataframes."
    )


def freq_to_dt(freq: str) -> float:
    """Convert frequency codes to float

    Parameters
    ----------
    freq: str
        Frequency code string.
        * 'B' Business days, dt = 1 / 252
        * 'D' Regular days, dt = 1 / 365
        * 'M' or "MS': Monthly, dt = 1 / 12

    Returns
    -------
        Timestep in years.
    """
    if freq == "B":
        return 1.0 / 252
    if freq == "D":
        return 1.0 / 365
    if freq in ("M", "MS"):
        return 1.0 / 12
    raise ValueError("Invalid freq argument. Valid values are 'D', 'B', 'M' or 'MS'.")


def freq_to_index(freq: str, t0: str, length: int) -> pd.DatetimeIndex:
    """Create a Pandas date-time index with equally spaced dates.

    Example
    -------

    .. exec_code::

        from samcoml.util import freq_to_index

        ndx = freq_to_index('B', t0='2020-1-6', length=10)
        print(ndx)


    Parameters
    ----------
    freq: str
        Frequency code string.
        * 'B' Business days, dt = 1 / 252
        * 'D' Regular days, dt = 1 / 365
        * 'M' or "MS': Monthly, dt = 1 / 12
    t0: str
        Start date string.
    length: int
        Length of the index.

    Returns
    -------

    """
    if freq == "B":
        return pd.bdate_range(start=t0, periods=length)
    elif freq == "D":
        return pd.date_range(start=t0, periods=length, freq="D")
    elif freq == "M":
        return pd.date_range(start=t0, periods=length, freq="M")
    elif freq == "MS":
        return pd.date_range(start=t0, periods=length, freq="MS")
    raise ValueError("Invalid freq argument. Valid values are 'D', 'B', 'M' or 'MS'.")
