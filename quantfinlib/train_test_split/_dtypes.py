"""Type aliases for the `train_test_split` module."""

from typing import Any, Union
from typing_extensions import Annotated, TypeAlias

import numpy as np
import pandas as pd

from quantfinlib.util._custom_types import (
    _CustomNumpyArrayPydanticAnnotation,
    _CustomPandasDataFramePydanticAnnotation,
    _CustomPandasSeriesPydanticAnnotation,
)


IndexType: TypeAlias = Annotated[np.ndarray, _CustomNumpyArrayPydanticAnnotation(shape=("*",), dtype=np.int_)]

YTypeArray: TypeAlias = Union[
    Annotated[np.ndarray, _CustomNumpyArrayPydanticAnnotation(shape=("*",), dtype=np.int64)],
    Annotated[np.ndarray, _CustomNumpyArrayPydanticAnnotation(shape=("*",), dtype=np.float_)],
    Annotated[np.ndarray, _CustomNumpyArrayPydanticAnnotation(shape=("*",), dtype=np.bool_)],
]
YTypeSeries: TypeAlias = Union[
    Annotated[pd.Series, _CustomPandasSeriesPydanticAnnotation(dtype=np.int64)],
    Annotated[pd.Series, _CustomPandasSeriesPydanticAnnotation(dtype=np.float_)],
    Annotated[pd.Series, _CustomPandasSeriesPydanticAnnotation(dtype=np.bool_)],
]
YType: TypeAlias = Union[YTypeArray, YTypeSeries]

GroupType: TypeAlias = Union[
    Annotated[np.ndarray, _CustomNumpyArrayPydanticAnnotation(shape=("*",), dtype=np.int64)],
    Annotated[pd.Series, _CustomPandasSeriesPydanticAnnotation(dtype=np.int64)],
    Annotated[pd.DatetimeIndex, _CustomPandasSeriesPydanticAnnotation()],
]

BoundPerFoldType: TypeAlias = Annotated[
    np.ndarray[Any, Any], _CustomNumpyArrayPydanticAnnotation(shape=("*", 2), dtype=np.int64)
]

XType: TypeAlias = Union[
    Annotated[np.ndarray, _CustomNumpyArrayPydanticAnnotation(shape=("*", "*"), dtype=Any)],
    Annotated[pd.DataFrame, _CustomPandasDataFramePydanticAnnotation()],
]
