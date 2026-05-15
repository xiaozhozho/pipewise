"""
Pipewise public package interface.
"""

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
    PipewiseTypeConversionError,
)

__version__ = "1.0.0"
__author__ = "XiaoZhouZhou"

__all__ = [
    "Pipewise",
    "PipewiseError",
    "PipewiseExecutionError",
    "PipewiseGroupByError",
    "PipewiseInputColumnError",
    "PipewiseInputSchemaError",
    "PipewiseOutputAssignmentError",
    "PipewiseOutputSchemaError",
    "PipewiseRegistrationError",
    "PipewiseSchemaError",
    "PipewiseTypeConversionError",
    "__version__",
    "__author__",
]
