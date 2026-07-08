"""A lightweight pandas DataFrame pipeline with schema validation and rollback."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ._schema import (
    SchemaRule,
    apply_types,
    matches_dtype,
    merge_output_schema,
    normalize_schema,
    validate_frame_schema,
    validate_output_schema_targets,
)
from .errors import (
    PipewiseExecutionError,
    PipewiseGroupByError,
    PipewiseInputColumnError,
    PipewiseInputSchemaError,
    PipewiseOutputAssignmentError,
    PipewiseOutputSchemaError,
    PipewiseRegistrationError,
    PipewiseTaskSelectionError,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


logger = logging.getLogger(__name__)


class Pipewise:
    """A pipeline tool for functional transformation of a pandas DataFrame.

    Register processing functions with the :meth:`register` decorator. Function
    parameter names map to DataFrame column names. Return values are written back
    as new columns. Multiple functions execute sequentially and can build on each
    other's outputs.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_schema: Optional[Dict[str, SchemaRule]] = None,
    ):
        self.data: pd.DataFrame = data
        self._input_schema = normalize_schema(input_schema, "input_schema")
        self._tasks: List[dict] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        outputs: Optional[Union[str, List[str], Iterable[str], Dict[str, type]]] = None,
        groupby: Optional[Union[str, List[str]]] = None,
        vectorized: bool = True,
        input_schema: Optional[Dict[str, SchemaRule]] = None,
        output_schema: Optional[Dict[str, SchemaRule]] = None,
        fallback_on_vectorized_error: bool = True,
    ) -> Callable:
        """Register a processing function. Can be used as a decorator with or
        without arguments.

        Parameters
        ----------
        func:
            The function to register (or ``None`` for bare decorator usage).
        outputs:
            Output column specification. See ``_normalize_outputs`` for accepted
            forms.
        groupby:
            Column name(s) to group by before executing the function.
        vectorized:
            If ``True``, the function receives entire pandas ``Series`` as arguments.
        input_schema:
            Schema rules for input columns.
        output_schema:
            Schema rules for output columns.
        fallback_on_vectorized_error:
            If ``True`` and vectorized execution fails with a known error, fall
            back to row-wise execution.
        """
        if func is None:
            return lambda f: self.register(
                f,
                outputs=outputs,
                groupby=groupby,
                vectorized=vectorized,
                input_schema=input_schema,
                output_schema=output_schema,
                fallback_on_vectorized_error=fallback_on_vectorized_error,
            )

        col_names, type_map, dict_mode = _normalize_outputs(outputs)
        normalized_input_schema = normalize_schema(input_schema, "input_schema")
        normalized_output_schema = normalize_schema(output_schema, "output_schema")
        merged_output_schema = merge_output_schema(
            normalized_output_schema,
            type_map,
            func.__name__,
        )
        normalized_groupby = _normalize_groupby(groupby, func.__name__)
        validate_output_schema_targets(
            merged_output_schema,
            col_names,
            dict_mode,
            func.__name__,
        )

        self._tasks.append(
            {
                "func": func,
                "func_name": func.__name__,
                "col_names": col_names,
                "type_map": type_map,
                "dict_mode": dict_mode,
                "groupby": normalized_groupby,
                "vectorized": vectorized,
                "input_schema": normalized_input_schema,
                "output_schema": merged_output_schema,
                "fallback_on_vectorized_error": fallback_on_vectorized_error,
            }
        )
        return func

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        inplace: bool = False,
        task: Optional[str] = None,
    ) -> pd.DataFrame:
        """Execute all registered functions in order and return the modified
        DataFrame.

        When ``task`` is provided, only the uniquely matching registered task
        is executed — useful for faster debugging of a single step.

        Parameters
        ----------
        inplace:
            If ``True``, modify the internal DataFrame directly instead of
            returning a copy.
        task:
            Name of a single task to execute.
        """
        data = self.data if inplace else self.data.copy()
        validate_frame_schema(
            data,
            self._input_schema,
            phase="pipeline input",
            exc_cls=PipewiseInputSchemaError,
        )

        tasks_to_run = self._select_tasks(task)
        snapshot = data.copy()
        iterator: Any = (
            _tqdm(tasks_to_run, ncols=60, colour="green") if _tqdm else tasks_to_run
        )
        current_task_name: Optional[str] = None

        try:
            for task_def in iterator:
                current_task_name = task_def["func_name"]
                if _tqdm is not None:
                    iterator.set_description(current_task_name)
                self._execute(data, task_def)
        except Exception as exc:
            _rollback(data, snapshot)
            raise PipewiseExecutionError(current_task_name) from exc

        return data

    def _select_tasks(self, task_name: Optional[str]) -> List[dict]:
        if task_name is None:
            return self._tasks
        if not isinstance(task_name, str) or not task_name:
            raise PipewiseTaskSelectionError(
                "run(task=...) expects a non-empty task name string."
            )
        matches = [task for task in self._tasks if task["func_name"] == task_name]
        if not matches:
            raise PipewiseTaskSelectionError(
                f"No registered task named '{task_name}' was found."
            )
        if len(matches) > 1:
            raise PipewiseTaskSelectionError(
                "Task name '{task_name}' is ambiguous; register tasks with unique "
                "function names to run them individually."
            )
        return matches

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    @property
    def tasks(self) -> List[Tuple[str, List[str], Any, bool]]:
        """Return a concise summary of registered tasks."""
        return [
            (
                t["func_name"],
                t["col_names"] if not t["dict_mode"] else "dict",
                t["groupby"],
                t["vectorized"],
            )
            for t in self._tasks
        ]

    def remove(self, func: Callable) -> bool:
        """Remove a previously registered function by reference."""
        for i, task_def in enumerate(self._tasks):
            if task_def["func"] is func:
                self._tasks.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all registered tasks."""
        self._tasks.clear()

    def plan(self) -> None:
        """Pretty-print the current execution plan."""
        if not self._tasks:
            logger.info("No tasks registered.")
            return
        info_lines = ["Execution Plan:"]
        header = (
            f"{'#':>3}  {'Function':<22} {'Outputs':<28} "
            f"{'GroupBy':<12} {'Vec':<5}"
        )
        info_lines.append(header)
        info_lines.append("-" * len(header))
        for i, task_def in enumerate(self._tasks):
            outputs = str(task_def["col_names"] if not task_def["dict_mode"] else "dict")
            outputs = outputs or "side-effect"
            gb = str(task_def["groupby"] or "-")
            vec = "Y" if task_def["vectorized"] else "N"
            info_lines.append(
                f"{i+1:>3}. {task_def['func_name']:<22} {outputs:<28} {gb:<12} {vec:<5}"
            )
        logger.info("\n".join(info_lines))

    # ------------------------------------------------------------------
    # Internal execution dispatcher
    # ------------------------------------------------------------------

    def _execute(self, data: pd.DataFrame, task_def: dict) -> None:
        func = task_def["func"]
        input_cols, has_kwargs = _parse_inputs(func)

        _validate_input_columns(data, input_cols, func.__name__)
        validate_frame_schema(
            data,
            task_def["input_schema"],
            phase=f"task input for '{func.__name__}'",
            exc_cls=PipewiseInputSchemaError,
        )

        if task_def["groupby"]:
            self._execute_groupby(data, task_def, input_cols, has_kwargs)
        else:
            self._execute_on_frame(data, task_def, input_cols, has_kwargs)

        validate_frame_schema(
            data,
            task_def["output_schema"],
            phase=f"task output for '{func.__name__}'",
            exc_cls=PipewiseOutputSchemaError,
        )

    def _execute_on_frame(
        self,
        data: pd.DataFrame,
        task_def: dict,
        input_cols: List[str],
        has_kwargs: bool,
    ) -> None:
        if task_def["vectorized"]:
            try:
                self._execute_vectorized(
                    data,
                    task_def,
                    input_cols,
                    has_kwargs,
                )
                return
            except Exception as exc:
                if _should_fallback(task_def, exc):
                    logger.debug(
                        "Function '%s' fell back to row-wise (vectorized failed: %s)",
                        task_def["func_name"],
                        exc,
                    )
                else:
                    raise

        self._execute_rowwise(data, task_def, input_cols, has_kwargs)

    # ------------------------------------------------------------------
    # Vectorized execution
    # ------------------------------------------------------------------

    def _execute_vectorized(
        self,
        data: pd.DataFrame,
        task_def: dict,
        input_cols: List[str],
        has_kwargs: bool,
    ) -> None:
        """Pass entire Series to the registered function."""
        func = task_def["func"]
        args = [data[col] for col in input_cols]
        if has_kwargs:
            extra_cols = {
                col: data[col]
                for col in data.columns
                if col not in input_cols
            }
            result = func(*args, **extra_cols)
        else:
            result = func(*args)

        if task_def["dict_mode"]:
            _assign_dict_output(data, result, task_def)
        else:
            _assign_columns(data, result, task_def)

    # ------------------------------------------------------------------
    # Row-wise execution
    # ------------------------------------------------------------------

    def _execute_rowwise(
        self,
        data: pd.DataFrame,
        task_def: dict,
        input_cols: List[str],
        has_kwargs: bool,
    ) -> None:
        """Evaluate the function once per row via ``data.apply()``."""
        func = task_def["func"]

        if has_kwargs:
            extra_cols = [c for c in data.columns if c not in input_cols]

            def row_func(row):
                args = [row[col] for col in input_cols]
                kwargs = {col: row[col] for col in extra_cols}
                return func(*args, **kwargs)
        else:
            def row_func(row):
                args = [row[col] for col in input_cols]
                return func(*args)

        if task_def["dict_mode"]:
            result_series = data.apply(row_func, axis=1)
            invalid_rows = [
                idx for idx, val in result_series.items()
                if not isinstance(val, dict)
            ]
            if invalid_rows:
                raise PipewiseOutputAssignmentError(
                    f"Function '{func.__name__}' must return a dict for each row when "
                    f"outputs='dict'. Invalid rows: {invalid_rows[:5]}."
                )
            updates = pd.DataFrame(result_series.tolist(), index=data.index)
            for col in updates.columns:
                data[col] = updates[col]
            return

        if not task_def["col_names"]:
            data.apply(row_func, axis=1)
            return

        if len(task_def["col_names"]) == 1:
            data[task_def["col_names"][0]] = data.apply(row_func, axis=1)
            apply_types(data, task_def["type_map"])
            return

        expanded = data.apply(row_func, axis=1, result_type="expand")
        n_returned = 1 if expanded.ndim == 1 else expanded.shape[1]
        if n_returned != len(task_def["col_names"]):
            raise PipewiseOutputAssignmentError(
                f"Function '{func.__name__}' returned {n_returned} values, but "
                f"outputs specifies {len(task_def['col_names'])} columns: {task_def['col_names']}."
            )
        if expanded.ndim == 1:
            data[task_def["col_names"][0]] = expanded
        else:
            expanded.columns = task_def["col_names"]
            for col in task_def["col_names"]:
                data[col] = expanded[col]
        apply_types(data, task_def["type_map"])

    # ------------------------------------------------------------------
    # GroupBy execution
    # ------------------------------------------------------------------

    def _execute_groupby(
        self,
        data: pd.DataFrame,
        task_def: dict,
        input_cols: List[str],
        has_kwargs: bool,
    ) -> None:
        """Split-apply-combine: group by column(s), apply function per group."""
        groupby_cols = task_def["groupby"]
        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]

        missing_gb = [col for col in groupby_cols if col not in data.columns]
        if missing_gb:
            raise PipewiseGroupByError(
                f"GroupBy columns {missing_gb} not found in the DataFrame."
            )

        def apply_per_group(group: pd.DataFrame) -> pd.DataFrame:
            result = group.copy()
            self._execute_on_frame(result, task_def, input_cols, has_kwargs)
            return result

        results = data.groupby(groupby_cols, group_keys=False).apply(apply_per_group)
        for col in results.columns:
            if col in groupby_cols:
                continue
            data[col] = results[col]

    # ------------------------------------------------------------------
    # Schema adapter (delegation for backward compat)
    # ------------------------------------------------------------------

    def _validate_frame_schema(self, data, schema, phase, exc_cls):
        validate_frame_schema(data, schema, phase, exc_cls)

    @classmethod
    def _matches_dtype(cls, series, expected) -> bool:
        return matches_dtype(series, expected)

    @classmethod
    def _validate_dtype(cls, series, rule, column, phase, exc_cls):
        expected = rule.get("dtype")
        if expected is not None and not matches_dtype(series, expected):
            raise exc_cls(
                f"{phase} column '{column}' has dtype {series.dtype}, "
                f"which does not match expected {expected!r}."
            )

    @staticmethod
    def _apply_types(data, type_map):
        apply_types(data, type_map)

    @staticmethod
    def _rollback(data, snapshot):
        _rollback(data, snapshot)


# ======================================================================
# Module-level helpers
# ======================================================================


def _normalize_outputs(outputs):
    """Parse the ``outputs`` parameter into ``(col_names, type_map, dict_mode)``."""
    if outputs is None:
        return [], None, False
    if isinstance(outputs, str):
        if outputs == "dict":
            return [], None, True
        return [outputs], None, False
    if isinstance(outputs, dict):
        return list(outputs.keys()), outputs, False
    if isinstance(outputs, (list, tuple)):
        return list(outputs), None, False
    raise PipewiseRegistrationError(
        f"outputs must be None, str, list, dict, or 'dict', got {type(outputs)}"
    )


def _normalize_groupby(groupby, func_name: str):
    if groupby is None:
        return None
    if isinstance(groupby, str):
        return groupby
    if isinstance(groupby, (list, tuple)) and all(
        isinstance(col, str) for col in groupby
    ):
        return list(groupby)
    raise PipewiseRegistrationError(
        f"Function '{func_name}' received invalid groupby={groupby!r}."
    )


def _validate_input_columns(data: pd.DataFrame, input_cols: List[str], func_name: str) -> None:
    """Check that every *input_cols* exists in *data*."""
    missing = [col for col in input_cols if col not in data.columns]
    if missing:
        raise PipewiseInputColumnError(
            f"Function '{func_name}' requires input columns {missing}, but they "
            "are missing from the DataFrame."
        )


def _parse_inputs(func):
    """Extract explicit input column names from function signature.

    Returns ``(input_cols, has_kwargs)``.
    """
    sig = inspect.signature(func)
    input_cols: List[str] = []
    has_kwargs = False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_kwargs = True
        elif param.default is inspect.Parameter.empty:
            input_cols.append(param.name)
    return input_cols, has_kwargs


def _should_fallback(task_def: dict, exc: Exception) -> bool:
    """Decide whether vectorized *exc* should trigger a row-wise fallback.

    Falls back only when the task allows it and the exception is a
    :class:`TypeError` or a :class:`ValueError` — the two exception types
    pandas raises when an operation is incompatible with a ``Series``
    (e.g. ``if series < 0`` raises ``ValueError: truth value ambiguous``).
    """
    if not task_def["fallback_on_vectorized_error"]:
        return False
    return isinstance(exc, (TypeError, ValueError))


def _rollback(data: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    """Restore *data* to the state captured in *snapshot*."""
    cols_to_remove = [c for c in data.columns if c not in snapshot.columns]
    if cols_to_remove:
        data.drop(columns=cols_to_remove, inplace=True)
    for col in snapshot.columns:
        data[col] = snapshot[col].copy()


# ---- Output assignment helpers (shared by vectorized & row-wise) ----


def _assign_dict_output(
    data: pd.DataFrame,
    result: Any,
    task_def: dict,
) -> None:
    """Assign dict/DataFrame results back to *data*."""
    func_name = task_def["func_name"]
    if isinstance(result, pd.DataFrame):
        for col in result.columns:
            data[col] = result[col]
        return
    if isinstance(result, dict):
        for col, value in result.items():
            data[col] = value
        return
    raise PipewiseOutputAssignmentError(
        f"Function '{func_name}' must return a dict or DataFrame when "
        "outputs='dict' in vectorized mode."
    )


def _assign_columns(
    data: pd.DataFrame,
    result: Any,
    task_def: dict,
) -> None:
    """Assign a tuple/list/DataFrame result to declared output columns."""
    col_names = task_def["col_names"]
    func_name = task_def["func_name"]
    type_map = task_def["type_map"]

    if not col_names:
        return
    if len(col_names) == 1:
        data[col_names[0]] = result
        apply_types(data, type_map)
        return

    if isinstance(result, pd.DataFrame):
        if result.shape[1] != len(col_names):
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned {result.shape[1]} columns, "
                f"but outputs specifies {len(col_names)} columns: {col_names}."
            )
        for i, col in enumerate(col_names):
            data[col] = result.iloc[:, i]
    elif isinstance(result, (list, tuple)):
        if len(result) != len(col_names):
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned {len(result)} values, but "
                f"outputs specifies {len(col_names)} columns: {col_names}."
            )
        for i, col in enumerate(col_names):
            data[col] = result[i]
    else:
        raise PipewiseOutputAssignmentError(
            f"Function '{func_name}' must return a tuple/list or DataFrame "
            f"for outputs {col_names}."
        )
    apply_types(data, type_map)
