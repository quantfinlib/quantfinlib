"""Helper classes for building custom data types for arrays and dataframes."""

from typing import Any, Union, Optional
import numpy as np
import pandas as pd
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue


class _CustomNumpyArrayPydanticAnnotation:  # pragma: no cover
    """Custom numpy array type annotation with Pydantic.

    Attributes
    ----------
    shape : Union[str, tuple]
        Expected shape of the numpy array.
    dtype : Union[np.dtype, Any]
        Expected data type of the numpy array.
    """

    def __init__(self, shape: Union[str, tuple], dtype: Union[np.dtype, Any]):
        self.shape = shape
        self.dtype = dtype

    def __get_pydantic_core_schema__(self, _source_type: Any, _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """
        Follow these shape and data type validation logics.
        - Checks if input is a numpy array with specified dtype and shape.
        - Wildcard (`*`) shapes allow for arbitrary dimensions.
        """

        def validate_numpy_array(value: Any) -> np.ndarray:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Expected a numpy array, got {type(value)}")
            # Validate dtype
            if self.dtype is not Any and not np.can_cast(value.dtype, np.dtype(self.dtype)):
                raise TypeError(f"Expected array of dtype {self.dtype}, got {value.dtype}")
            # Validate shape
            if isinstance(self.shape, tuple):
                if len(self.shape) != value.ndim:
                    raise TypeError(f"Expected {len(self.shape)}-dimensional array, got {value.ndim}-dimensional array")

                for dim, expected_dim in zip(value.shape, self.shape):
                    if expected_dim != "*" and dim != int(expected_dim):
                        raise TypeError(f"Expected array shape {self.shape}, but got {value.shape}")

            return value

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema([core_schema.any_schema()]),
            python_schema=core_schema.chain_schema(
                [
                    core_schema.no_info_plain_validator_function(validate_numpy_array),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.tolist()  # Convert to list for JSON serialization
            ),
        )

    def __get_pydantic_json_schema__(
        self, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Define the JSON schema to reflect shape and dtype constraints."""
        return {
            "type": "array",
            "items": {"type": "number" if self.dtype in [np.int_, np.float64, float, int] else "any"},
            "minItems": self.shape[0] if isinstance(self.shape, tuple) and self.shape[0] != "*" else 0,
        }


class _CustomPandasDataFramePydanticAnnotation:  # pragma: no cover
    """Custom pandas DataFrame type annotation with Pydantic.

    Attributes
    ----------
        columns : Optional[dict], optional
            Dictionary where keys are column names, and values are expected dtypes, by default None.
        dtype : Optional[Any], optional
            Expected dtype for the entire DataFrame, by default None.
        shape : Union[tuple, str], optional
            A tuple representing the expected shape (rows, columns) or '*' for arbitrary, by default "*".

    """

    def __init__(self, columns: Optional[dict] = None, dtype: Optional[Any] = None, shape: Union[tuple, str] = "*"):
        self.columns = columns
        self.dtype = dtype
        self.shape = shape

    def __get_pydantic_core_schema__(self, _source_type: Any, _handler: Any) -> core_schema.CoreSchema:
        def validate_dataframe(value: Any) -> pd.DataFrame:
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"Expected a pandas DataFrame, got {type(value)}")

            # Validate columns if columns are provided
            if self.columns is not None:
                if not isinstance(self.columns, dict):
                    raise TypeError(f"Expected 'columns' to be a dictionary, got {type(self.columns)}")
                for col, dtype in self.columns.items():
                    if col not in value.columns:
                        raise TypeError(f"Missing expected column: {col}")
                    if not pd.api.types.is_dtype_equal(value[col].dtype, dtype):
                        raise TypeError(
                            f"Expected column '{col}' to have dtype '{dtype}', but got '{value[col].dtype}'"
                        )

            # If dtype is provided, validate the entire DataFrame's dtype
            elif self.dtype is not None:
                if not all(pd.api.types.is_dtype_equal(value.dtypes[col], self.dtype) for col in value.columns):
                    raise TypeError(f"Expected all columns to have dtype '{self.dtype}', but found mismatch.")

            # Validate shape
            if self.shape != "*" and len(self.shape) == 2:
                if value.shape != self.shape:
                    raise TypeError(f"Expected DataFrame shape {self.shape}, but got {value.shape}")

            return value

        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(validate_dataframe),
            ]
        )

    def __get_pydantic_json_schema__(self, _core_schema: core_schema.CoreSchema, handler: Any) -> dict:
        # Define a simple JSON schema for DataFrame validation
        return {
            "type": "object",
            "properties": {
                col: {
                    "type": "array",
                    "items": {"type": "number" if pd.api.types.is_numeric_dtype(dtype) else "string"},
                }
                for col, dtype in (self.columns or {}).items()
            },
            "additionalProperties": False,
        }


class _CustomPandasSeriesPydanticAnnotation:  # pragma: no cover
    """Custom pandas Series type annotation with Pydantic.

    Attributes
    ----------
    dtype : Optional[Any], optional
        Expected dtype for the Series, by default None.
    shape : Union[int, str], optional
        Expected number of elements in the Series, or '*' for arbitrary length, by default "*".
    """

    def __init__(self, dtype: Optional[Any] = None, shape: Union[int, str] = "*"):

        self.dtype = dtype
        self.shape = shape

    def __get_pydantic_core_schema__(self, _source_type: Any, _handler: Any) -> core_schema.CoreSchema:
        def validate_series(value: Any) -> pd.Series:
            if not isinstance(value, pd.Series):
                raise TypeError(f"Expected a pandas Series, got {type(value)}")

            # Validate dtype if provided
            if self.dtype is not None:
                if not pd.api.types.is_dtype_equal(value.dtype, self.dtype):
                    raise TypeError(f"Expected Series to have dtype '{self.dtype}', but got '{value.dtype}'")

            # Validate shape (length)
            if self.shape != "*":
                if not isinstance(self.shape, int):
                    raise TypeError(f"Expected 'shape' to be an integer or '*', got {type(self.shape)}")
                if len(value) != self.shape:
                    raise TypeError(f"Expected Series to have length {self.shape}, but got {len(value)}")

            return value

        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(validate_series),
            ]
        )

    def __get_pydantic_json_schema__(self, _core_schema: core_schema.CoreSchema, handler: Any) -> dict:
        # Define a simple JSON schema for Series validation
        return {
            "type": "array",
            "items": {"type": "number" if pd.api.types.is_numeric_dtype(self.dtype) else "string"},
            "maxItems": self.shape if self.shape != "*" else None,
            "minItems": self.shape if isinstance(self.shape, int) else None,
        }
