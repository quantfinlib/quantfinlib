from typing import Any, Union

import nptyping as npt
import pandas as pd

INDEX_TYPE = npt.NDArray[npt.Shape["*"], npt.Int]
ARRAYORDF_TYPE = Union[npt.NDArray[npt.Shape["*,*"], Any], npt.DataFrame[Any]]
Y_TYPE = Union[
    npt.NDArray[npt.Shape["*"], npt.Int],
    npt.NDArray[npt.Shape["*"], npt.Float],
    npt.NDArray[npt.Shape["*"], npt.Bool8],  
    pd.Series,
]
GROUP_TYPE = Union[
    npt.NDArray[npt.Shape["*"], npt.Int],
    pd.Series,
    pd.DatetimeIndex,
]

BND_PER_FOLD_TYPE = npt.NDArray[npt.Shape["*, 2"], npt.Int]
