"""Execution strategies: vectorized, row-wise and grouped.

The vectorized *call* and the output *assignment* are deliberately separate.
Only a failure inside the call can trigger the row-wise fallback; a failure
while writing results back is always a shape problem and is surfaced directly.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, List

import pandas as pd

from ._assign import assign_result, row_value_length
from ._schema import apply_types
from ._tasks import TaskDef
from .errors import PipewiseGroupByError, PipewiseOutputAssignmentError

logger = logging.getLogger(__name__)

#: Exceptions that indicate a function is incompatible with Series input.
_FALLBACK_ERRORS = (TypeError, ValueError, AttributeError)


def should_fallback(task_def: TaskDef, exc: Exception) -> bool:
    """Whether *exc* may trigger the vectorized → row-wise fallback."""
    if not task_def.fallback_on_vectorized_error:
        return False
    return isinstance(exc, _FALLBACK_ERRORS)


def call_vectorized(data: pd.DataFrame, task_def: TaskDef) -> Any:
    """Invoke the function once with whole ``Series`` columns."""
    args = [data[col] for col in task_def.input_cols]
    if task_def.has_kwargs:
        input_set = set(task_def.input_cols)
        extra = {col: data[col] for col in data.columns if col not in input_set}
        return task_def.func(*args, **extra)
    return task_def.func(*args)


def execute_on_frame(data: pd.DataFrame, task_def: TaskDef) -> None:
    """Run *task_def* against the whole frame (vectorized or row-wise)."""
    if task_def.vectorized:
        try:
            result = call_vectorized(data, task_def)
        except Exception as exc:
            if not should_fallback(task_def, exc):
                _hint_if_fallback_disabled(task_def, exc)
                raise
            logger.info(
                "Function '%s' fell back to row-wise execution because "
                "vectorized execution is incompatible: %s: %s",
                task_def.func_name,
                type(exc).__name__,
                exc,
            )
            warnings.warn(
                f"Function '{task_def.func_name}' fell back to row-wise execution "
                f"because vectorized execution is incompatible: "
                f"{type(exc).__name__}: {exc}. The function body has already run "
                f"once on the whole column, so side effects may have occurred.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            # The vectorized call succeeded, so anything that fails from here on
            # is an output-shape problem, not a vectorization incompatibility.
            assign_result(data, result, task_def)
            return

    execute_rowwise(data, task_def)


def _hint_if_fallback_disabled(task_def: TaskDef, exc: Exception) -> None:
    """Point at the row-wise escape hatch when a fallback-eligible error occurs.

    Only fires when the fallback was actually switched off, so the advice is
    never redundant.
    """
    if task_def.fallback_on_vectorized_error:
        return
    if not isinstance(exc, _FALLBACK_ERRORS):
        return
    message = (
        f"Function '{task_def.func_name}' raised {type(exc).__name__} during "
        f"vectorized execution: {exc}. If it is meant to run per row, register it "
        f"with vectorized=False (or pass fallback_on_vectorized_error=True)."
    )
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=4)


# ----------------------------------------------------------------------
# Row-wise execution
# ----------------------------------------------------------------------


def execute_rowwise(data: pd.DataFrame, task_def: TaskDef) -> None:
    """Evaluate the function once per row.

    Iterating with :meth:`pandas.DataFrame.itertuples` avoids the per-row
    ``Series`` construction that makes ``DataFrame.apply(axis=1)`` roughly an
    order of magnitude slower.
    """
    if data.columns.duplicated().any():
        # Positional lookup is ambiguous with duplicate labels; keep pandas'
        # label-based apply for that (pathological) case.
        results = list(data.apply(_label_row_func(data, task_def), axis=1))
        _write_rowwise_results(data, task_def, results)
        return

    columns = list(data.columns)
    position = {col: i for i, col in enumerate(columns)}
    input_positions = [position[col] for col in task_def.input_cols]

    if task_def.has_kwargs:
        input_set = set(task_def.input_cols)
        extra_cols = [col for col in columns if col not in input_set]
        extra_positions = [position[col] for col in extra_cols]
    else:
        extra_cols = []
        extra_positions = []

    func = task_def.func
    results = [
        _call_row(func, values, input_positions, extra_cols, extra_positions)
        for values in data.itertuples(index=False, name=None)
    ]
    _write_rowwise_results(data, task_def, results)


def _call_row(
    func,
    values: tuple,
    input_positions: List[int],
    extra_cols: List[str],
    extra_positions: List[int],
):
    args = [values[i] for i in input_positions]
    if extra_positions:
        kwargs = {col: values[i] for col, i in zip(extra_cols, extra_positions)}
        return func(*args, **kwargs)
    return func(*args)


def _label_row_func(data: pd.DataFrame, task_def: TaskDef):
    """Build a label-based row function (duplicate-column fallback)."""
    func = task_def.func
    input_cols = task_def.input_cols
    if task_def.has_kwargs:
        input_set = set(input_cols)
        extra_cols = [col for col in data.columns if col not in input_set]

        def row_func(row):
            return func(
                *[row[col] for col in input_cols],
                **{col: row[col] for col in extra_cols},
            )
    else:
        def row_func(row):
            return func(*[row[col] for col in input_cols])
    return row_func


def _write_rowwise_results(
    data: pd.DataFrame,
    task_def: TaskDef,
    results: List[Any],
) -> None:
    """Write per-row *results* back to *data*."""
    func_name = task_def.func_name

    if task_def.dict_mode:
        invalid = [
            data.index[i]
            for i, value in enumerate(results)
            if not isinstance(value, dict)
        ]
        if invalid:
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' must return a dict for each row when "
                f"outputs='dict'. Invalid rows: {invalid[:5]}."
            )
        updates = pd.DataFrame(results, index=data.index)
        for col in updates.columns:
            data[col] = updates[col]
        return

    if not task_def.col_names:
        return  # side-effect only: the function has already run

    col_names = task_def.col_names
    if len(col_names) == 1:
        data[col_names[0]] = pd.Series(results, index=data.index)
        apply_types(data, task_def.type_map)
        return

    expected = len(col_names)
    for i, value in enumerate(results):
        returned = row_value_length(value)
        if returned != expected:
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned {returned} value(s) for row "
                f"{data.index[i]!r}, but outputs specifies {expected} columns: "
                f"{col_names}."
            )

    if not results:
        for col in col_names:
            data[col] = pd.Series(dtype="object", index=data.index)
        apply_types(data, task_def.type_map)
        return

    # Rows are known to be uniform in length, so this expands cleanly instead of
    # letting pandas raise an opaque ragged-array error.
    expanded = pd.DataFrame(results, index=data.index)
    for i, col in enumerate(col_names):
        data[col] = expanded.iloc[:, i].values
    apply_types(data, task_def.type_map)


# ----------------------------------------------------------------------
# Grouped execution
# ----------------------------------------------------------------------


def execute_groupby(data: pd.DataFrame, task_def: TaskDef) -> None:
    """Split-apply-combine: group by column(s), apply the function per group."""
    groupby_cols = task_def.groupby
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]

    missing = [col for col in groupby_cols if col not in data.columns]
    if missing:
        raise PipewiseGroupByError(
            f"GroupBy columns {missing} not found in the DataFrame."
        )

    def apply_per_group(group: pd.DataFrame) -> pd.DataFrame:
        result = group.copy()
        execute_on_frame(result, task_def)
        return result

    results = data.groupby(groupby_cols, group_keys=False).apply(apply_per_group)
    for col in results.columns:
        if col in groupby_cols:
            continue
        data[col] = results[col]
