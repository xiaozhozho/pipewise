"""
Pipewise public package interface.
"""

from ._schema import SchemaRule
from .core import Pipewise
from .errors import (
    PipewiseError,
    PipewiseExecutionError,
    PipewiseGroupByError,
    PipewiseInputColumnError,
    PipewiseInputSchemaError,
    PipewiseOutputAssignmentError,
    PipewiseOutputSchemaError,
    PipewiseRegistrationError,
    PipewiseSchemaError,
    PipewiseTaskSelectionError,
    PipewiseTypeConversionError,
)

__version__ = "2.0.0"
__author__ = "XiaoZhouZhou"

__all__ = [
    "Pipewise",
    "SchemaRule",
    "PipewiseError",
    "PipewiseExecutionError",
    "PipewiseGroupByError",
    "PipewiseInputColumnError",
    "PipewiseInputSchemaError",
    "PipewiseOutputAssignmentError",
    "PipewiseOutputSchemaError",
    "PipewiseRegistrationError",
    "PipewiseSchemaError",
    "PipewiseTaskSelectionError",
    "PipewiseTypeConversionError",
    "__version__",
    "__author__",
]
