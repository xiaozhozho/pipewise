"""
Pipewise public package interface.
"""

from .core import Pipewise
from ._schema import SchemaRule
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

__version__ = "1.0.1"
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
