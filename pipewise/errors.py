class PipewiseError(Exception):
    """Base class for all Pipewise-specific errors."""


class PipewiseRegistrationError(PipewiseError):
    """Raised when a task or schema is registered with invalid metadata."""


class PipewiseSchemaError(PipewiseError):
    """Base class for schema validation failures."""


class PipewiseInputColumnError(PipewiseSchemaError):
    """Raised when required input columns are missing."""


class PipewiseInputSchemaError(PipewiseSchemaError):
    """Raised when input data does not satisfy the declared schema."""


class PipewiseOutputSchemaError(PipewiseSchemaError):
    """Raised when task output does not satisfy the declared schema."""


class PipewiseGroupByError(PipewiseSchemaError):
    """Raised when groupby configuration is invalid for the current data."""


class PipewiseOutputAssignmentError(PipewiseError):
    """Raised when a task returns data that cannot be assigned to outputs."""


class PipewiseTypeConversionError(PipewiseError):
    """Raised when declared output type coercion fails."""


class PipewiseExecutionError(PipewiseError):
    """Raised when pipeline execution fails and changes are rolled back."""

    def __init__(self, task_name=None):
        if task_name:
            message = f"Task '{task_name}' failed, all changes rolled back."
        else:
            message = "Pipeline execution failed, all changes rolled back."
        super().__init__(message)
        self.task_name = task_name
