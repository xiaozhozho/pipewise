"""A lightweight pandas DataFrame pipeline with schema validation and rollback."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

from ._execution import execute_groupby, execute_on_frame
from ._hazards import warn_if_vectorized_hazard
from ._schema import (
    SchemaRule,
    merge_output_schema,
    normalize_schema,
    validate_frame_schema,
    validate_output_schema_targets,
)
from ._tasks import (
    TaskDef,
    TaskSummary,
    normalize_groupby,
    normalize_outputs,
    parse_signature,
    validate_input_columns,
    warn_on_shadowed_defaults,
)
from .errors import (
    PipewiseExecutionError,
    PipewiseInputSchemaError,
    PipewiseOutputSchemaError,
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
        self._tasks: List[TaskDef] = []

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
        fallback_on_vectorized_error: bool = False,
    ) -> Callable:
        """Register a processing function. Can be used as a decorator with or
        without arguments.

        Parameters
        ----------
        func:
            The function to register (or ``None`` for bare decorator usage).
        outputs:
            Output column specification. Accepted forms:
            ``None`` (side-effect only), ``"col"`` (single column),
            ``["c1", "c2"]`` (multi-column), ``{"col": type}`` (typed),
            or ``"dict"`` (dynamic dict output).
        groupby:
            Column name(s) to group by before executing the function.
        vectorized:
            If ``True``, the function receives whole ``Series`` objects.
        input_schema:
            Schema rules for input columns.
        output_schema:
            Schema rules for output columns.
        fallback_on_vectorized_error:
            If ``True`` and the vectorized call fails with a Series-incompatible
            error, retry the function row by row. Off by default: the vectorized
            call has already run once, so a fallback re-executes the function
            body and can repeat its side effects.
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

        parsed = parse_signature(func)
        if vectorized:
            warn_if_vectorized_hazard(func, parsed.input_cols)
        warn_on_shadowed_defaults(func.__name__, parsed.defaulted, self.data.columns)

        col_names, type_map, dict_mode = normalize_outputs(outputs)
        normalized_input_schema = normalize_schema(input_schema, "input_schema")
        normalized_output_schema = normalize_schema(output_schema, "output_schema")
        merged_output_schema = merge_output_schema(
            normalized_output_schema,
            type_map,
            func.__name__,
        )
        normalized_groupby = normalize_groupby(groupby, func.__name__)
        validate_output_schema_targets(
            merged_output_schema,
            col_names,
            dict_mode,
            func.__name__,
        )

        self._tasks.append(
            TaskDef(
                func=func,
                func_name=func.__name__,
                col_names=col_names,
                type_map=type_map,
                dict_mode=dict_mode,
                groupby=normalized_groupby,
                vectorized=vectorized,
                input_schema=normalized_input_schema,
                output_schema=merged_output_schema,
                input_cols=parsed.input_cols,
                has_kwargs=parsed.has_kwargs,
                fallback_on_vectorized_error=fallback_on_vectorized_error,
            )
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
            If ``True``, modify the internal DataFrame directly.
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
                current_task_name = task_def.func_name
                if _tqdm is not None:
                    iterator.set_description(current_task_name)
                self._execute(data, task_def)
        except Exception as exc:
            _rollback(data, snapshot)
            raise PipewiseExecutionError(current_task_name) from exc

        return data

    def _execute(self, data: pd.DataFrame, task_def: TaskDef) -> None:
        validate_input_columns(data, task_def)
        validate_frame_schema(
            data,
            task_def.input_schema,
            phase=f"task input for '{task_def.func_name}'",
            exc_cls=PipewiseInputSchemaError,
        )

        if task_def.groupby:
            execute_groupby(data, task_def)
        else:
            execute_on_frame(data, task_def)

        validate_frame_schema(
            data,
            task_def.output_schema,
            phase=f"task output for '{task_def.func_name}'",
            exc_cls=PipewiseOutputSchemaError,
        )

    def _select_tasks(self, task_name: Optional[str]) -> List[TaskDef]:
        if task_name is None:
            return self._tasks
        if not isinstance(task_name, str) or not task_name:
            raise PipewiseTaskSelectionError(
                "run(task=...) expects a non-empty task name string."
            )
        matches = [task for task in self._tasks if task.func_name == task_name]
        if not matches:
            raise PipewiseTaskSelectionError(
                f"No registered task named '{task_name}' was found."
            )
        if len(matches) > 1:
            raise PipewiseTaskSelectionError(
                f"Task name '{task_name}' is ambiguous; register tasks with unique "
                "function names to run them individually."
            )
        return matches

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    @property
    def tasks(self) -> List[TaskSummary]:
        """Return a concise summary of registered tasks."""
        return [
            TaskSummary(t.func_name, t.output_label, t.groupby, t.vectorized)
            for t in self._tasks
        ]

    def remove(self, func: Callable) -> bool:
        """Remove a previously registered function by reference."""
        for i, task_def in enumerate(self._tasks):
            if task_def.func is func:
                self._tasks.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all registered tasks."""
        self._tasks.clear()

    def plan(self) -> None:
        """Log the current execution plan."""
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
            outputs = str(task_def.output_label) or "side-effect"
            groupby = str(task_def.groupby or "-")
            vectorized = "Y" if task_def.vectorized else "N"
            info_lines.append(
                f"{i+1:>3}. {task_def.func_name:<22} {outputs:<28} "
                f"{groupby:<12} {vectorized:<5}"
            )
        logger.info("\n".join(info_lines))


def _rollback(data: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    """Restore *data* to the state captured in *snapshot*."""
    cols_to_remove = [c for c in data.columns if c not in snapshot.columns]
    if cols_to_remove:
        data.drop(columns=cols_to_remove, inplace=True)
    for col in snapshot.columns:
        data[col] = snapshot[col].copy()
