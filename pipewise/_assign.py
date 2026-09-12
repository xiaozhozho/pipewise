"""Output assignment helpers shared by the vectorized and row-wise paths.

Every helper here receives an already-computed *result* and writes it back to
the target DataFrame. Shape and length problems are reported as
:class:`~pipewise.errors.PipewiseOutputAssignmentError` with a message that
names the offending function and column, rather than letting pandas raise an
opaque error or silently broadcast a single value across every row.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ._schema import apply_types
from ._tasks import TaskDef
from .errors import PipewiseOutputAssignmentError


def assign_result(data: pd.DataFrame, result: Any, task_def: TaskDef) -> None:
    """Write a vectorized *result* back to *data*, dispatched by output mode."""
    if task_def.dict_mode:
        assign_dict_output(data, result, task_def)
    else:
        assign_columns(data, result, task_def)


def assign_dict_output(data: pd.DataFrame, result: Any, task_def: TaskDef) -> None:
    """Assign dict/DataFrame results back to *data*."""
    func_name = task_def.func_name
    if isinstance(result, pd.DataFrame):
        result = align_result_index(result, data.index)
        for col in result.columns:
            data[col] = result[col]
        return
    if isinstance(result, dict):
        for col, value in result.items():
            require_column_length(value, len(data), func_name, col)
            data[col] = value
        return
    raise PipewiseOutputAssignmentError(
        f"Function '{func_name}' must return a dict or DataFrame when "
        "outputs='dict' in vectorized mode."
    )


def assign_columns(data: pd.DataFrame, result: Any, task_def: TaskDef) -> None:
    """Assign a tuple/list/DataFrame/Series/2-D array result to output columns."""
    col_names = task_def.col_names
    func_name = task_def.func_name
    type_map = task_def.type_map

    if not col_names:
        return

    # --- Single output column ---
    if len(col_names) == 1:
        if is_2d_arraylike(result):
            if result.shape[1] != 1:
                raise PipewiseOutputAssignmentError(
                    f"Function '{func_name}' returned {result.shape[1]} columns "
                    f"for the single output column '{col_names[0]}'."
                )
            result = result[:, 0]
        value = safe_values(result, data.index)
        require_column_length(value, len(data), func_name, col_names[0])
        data[col_names[0]] = value
        apply_types(data, type_map)
        return

    # --- Multi output column ---
    if isinstance(result, pd.DataFrame):
        if result.shape[1] != len(col_names):
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned {result.shape[1]} columns, "
                f"but outputs specifies {len(col_names)} columns: {col_names}."
            )
        result = align_result_index(result, data.index)
        for i, col in enumerate(col_names):
            data[col] = safe_values(result.iloc[:, i], data.index)
    elif is_2d_arraylike(result):
        # e.g. a 2-D ``numpy.ndarray`` returned from a vectorized function
        if result.shape[1] != len(col_names):
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned {result.shape[1]} columns, "
                f"but outputs specifies {len(col_names)} columns: {col_names}."
            )
        for i, col in enumerate(col_names):
            value = safe_values(result[:, i], data.index)
            require_column_length(value, len(data), func_name, col)
            data[col] = value
    elif isinstance(result, pd.Series):
        raise PipewiseOutputAssignmentError(
            f"Function '{func_name}' returned a Series, but outputs specifies "
            f"{len(col_names)} columns: {col_names}."
        )
    elif isinstance(result, (list, tuple)):
        if len(result) != len(col_names):
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned {len(result)} values, but "
                f"outputs specifies {len(col_names)} columns: {col_names}."
            )
        for i, col in enumerate(col_names):
            value = safe_values(result[i], data.index)
            require_column_length(value, len(data), func_name, col)
            data[col] = value
    else:
        raise PipewiseOutputAssignmentError(
            f"Function '{func_name}' must return a tuple/list, DataFrame or "
            f"2-D array for outputs {col_names}."
        )
    apply_types(data, type_map)


def safe_values(value: Any, target_index: pd.Index) -> Any:
    """Extract values respecting *target_index* alignment.

    - ``pd.Series`` → ``.values`` after index alignment
    - single-column ``pd.DataFrame`` → ``.values``
    - scalar / list / array → returned as-is
    """
    if isinstance(value, pd.Series):
        return value.reindex(target_index).values
    if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        return value.iloc[:, 0].reindex(target_index).values
    return value


def align_result_index(result: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    """Reindex *result* to match *target_index*, without mutating."""
    if not result.index.equals(target_index):
        return result.reindex(target_index)
    return result


def is_2d_arraylike(value: Any) -> bool:
    """True for 2-D array-like objects such as ``numpy.ndarray``.

    pandas ``DataFrame`` also has ``ndim == 2`` but is handled separately, so
    it is excluded here.
    """
    return getattr(value, "ndim", None) == 2 and not isinstance(value, pd.DataFrame)


def column_length(value: Any) -> Optional[int]:
    """Return the per-row length of *value*, or ``None`` if it is a scalar.

    Strings, dicts and sets are treated as single cell values so that pandas'
    normal broadcasting rules keep working; everything list-like (list, tuple,
    ``numpy.ndarray``, ``pd.Index``, …) reports its length for validation.
    """
    if isinstance(value, (str, bytes, dict, set, frozenset)):
        return None
    if isinstance(value, (list, tuple)):
        return len(value)
    if pd.api.types.is_list_like(value):
        try:
            return len(value)
        except TypeError:
            return None
    return None


def require_column_length(
    value: Any,
    expected: int,
    func_name: str,
    column: str,
) -> None:
    """Raise a clear error when a per-row sequence does not match the row count.

    Without this check pandas would either raise an opaque length error or, for
    length-1 sequences, silently broadcast a single value across every row.
    """
    length = column_length(value)
    if length is not None and length != expected:
        raise PipewiseOutputAssignmentError(
            f"Function '{func_name}' produced {length} value(s) for output "
            f"column '{column}', but the DataFrame has {expected} row(s). "
            "Return one value per row, or a scalar to broadcast."
        )


def row_value_length(value: Any) -> int:
    """Number of output values a single row-wise result expands to."""
    if isinstance(value, (str, bytes)):
        return 1
    if isinstance(value, (list, tuple, dict, set, frozenset, pd.Series)):
        return len(value)
    if pd.api.types.is_list_like(value):
        try:
            return len(value)
        except TypeError:
            return 1
    return 1
