"""A lightweight pandas DataFrame pipeline with schema validation and rollback."""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
import warnings
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
            If ``True`` and vectorized execution fails, fall back to row-wise.
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

        if vectorized:
            _warn_if_vectorized_hazard(func)

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
                    logger.info(
                        "Function '%s' fell back to row-wise execution because "
                        "vectorized execution is incompatible: %s: %s",
                        task_def["func_name"],
                        type(exc).__name__,
                        exc,
                    )
                    warnings.warn(
                        f"Function '{task_def['func_name']}' fell back to row-wise "
                        f"execution because vectorized execution is incompatible: "
                        f"{type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
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

    Falls back when the task allows it and the exception is one of the common
    types raised when a function is incompatible with pandas ``Series``:
    :class:`TypeError` (e.g. ``if series < 0``), :class:`ValueError`
    (e.g. ``truth value ambiguous``), or :class:`AttributeError`
    (e.g. ``sc.split(',')`` where *sc* is a Series, not a scalar).
    """
    if not task_def["fallback_on_vectorized_error"]:
        return False
    return isinstance(exc, (TypeError, ValueError, AttributeError))


def _rollback(data: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    """Restore *data* to the state captured in *snapshot*."""
    cols_to_remove = [c for c in data.columns if c not in snapshot.columns]
    if cols_to_remove:
        data.drop(columns=cols_to_remove, inplace=True)
    for col in snapshot.columns:
        data[col] = snapshot[col].copy()


# ======================================================================
# AST-based vectorized-hazard detection
# ======================================================================

_VECTORIZED_HAZARD_PATTERNS: Dict[str, str] = {
    "Call@len": (
        "`len(%s)` returns the number of rows when applied to a Series, "
        "not the length of each element. Use `.str.len()` (vectorized) or "
        "`vectorized=False` (row-wise)."
    ),
    "Call@isinstance": (
        "`isinstance(%s, ...)` always returns `False` when applied to a Series. "
        "Use `vectorized=False` to check each element individually."
    ),
    "Call@type": (
        "`type(%s)` returns `pandas.core.series.Series` when applied to a Series, "
        "not the type of each element. Use `vectorized=False` for per-element checks."
    ),
    "Call@.split": (
        "`%s.split(...)` raises AttributeError on a Series — use `.str.split()` "
        "(vectorized) or `vectorized=False` (row-wise)."
    ),
    "Call@.strip": (
        "`%s.strip(...)` raises AttributeError on a Series — use `.str.strip()` "
        "(vectorized) or `vectorized=False` (row-wise)."
    ),
    "Call@.lower": (
        "`%s.lower()` raises AttributeError on a Series — use `.str.lower()` "
        "(vectorized) or `vectorized=False` (row-wise)."
    ),
    "Call@.upper": (
        "`%s.upper()` raises AttributeError on a Series — use `.str.upper()` "
        "(vectorized) or `vectorized=False` (row-wise)."
    ),
    "Subscript@int": (
        "`%s[0]` on a Series does label-based indexing, not per-element access. "
        "Use `.str[0]` (vectorized) or `vectorized=False` (row-wise)."
    ),
    "Subscript@str": (
        "`%s['key']` on a Series does label-based indexing, not dict-access per "
        "element. Use `.str['key']` (vectorized) or `vectorized=False` (row-wise)."
    ),
}


def _warn_if_vectorized_hazard(func: Callable) -> None:
    """Scan *func*'s source for common vectorized-incompatible patterns.

    Emits ``logger.warning`` if any hazard is found.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return  # can't inspect (e.g. built-in / C extension) — skip

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    detector = _VectorizedHazardVisitor()
    detector.visit(tree)

    for (category, name), detail in sorted(detector.hazards.items()):
        msg = _VECTORIZED_HAZARD_PATTERNS[category] % name
        if detail:
            msg = f"{msg} (detected: `{detail}`)"
        logger.warning(
            "Function '%s' may be vectorized-incompatible: %s",
            func.__name__,
            msg,
        )


class _VectorizedHazardVisitor(ast.NodeVisitor):
    """Walk the AST and record vectorized-incompatible patterns."""

    def __init__(self):
        self.hazards: Dict[Tuple[str, str], Optional[str]] = {}  # (category, name) → detail

    def _record(self, category: str, name: str, detail: Optional[str] = None) -> None:
        if category not in _VECTORIZED_HAZARD_PATTERNS:
            return
        # deduplicate by (category, name)
        if (category, name) not in self.hazards:
            self.hazards[(category, name)] = detail

    # --- Call patterns ---

    def _visit_call_by_name(self, node: ast.Call, func_name: str) -> None:
        """Called when we see `func_name(...)`."""
        # len(x)
        if func_name == "len" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                self._record("Call@len", arg.id)
        # isinstance(x, ...)
        elif func_name == "isinstance" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                self._record("Call@isinstance", arg.id)
        # type(x)
        elif func_name == "type" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                self._record("Call@type", arg.id)

    def visit_Call(self, node: ast.Call) -> None:
        # Direct name call: len(x), isinstance(x), type(x)
        if isinstance(node.func, ast.Name):
            self._visit_call_by_name(node, node.func.id)

        # Attribute method call: obj.split(), obj.strip(), etc.
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            # Only flag known string scalar methods
            for suffix, category in [
                ("split", "Call@.split"),
                ("strip", "Call@.strip"),
                ("lower", "Call@.lower"),
                ("upper", "Call@.upper"),
            ]:
                if method_name == suffix:
                    # Capture the expression text as best-effort detail
                    detail = _expr_source(node)
                    if isinstance(node.func.value, ast.Name):
                        self._record(category, node.func.value.id, detail)
                    break

        self.generic_visit(node)

    # --- Subscript patterns ---

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # obj[0] — int constant subscript
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                self._record("Subscript@int", var_name, f"{var_name}[{node.slice.value}]")
            elif isinstance(node.slice, ast.Index):  # Python < 3.9 compat
                inner = node.slice.value
                if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
                    self._record("Subscript@int", var_name, f"{var_name}[{inner.value}]")
            elif isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                self._record("Subscript@str", var_name, f"{var_name}['{node.slice.value}']")

        self.generic_visit(node)


def _expr_source(node: ast.AST) -> Optional[str]:
    """Return a short source representation of *node*, best-effort."""
    try:
        return ast.unparse(node)
    except Exception:
        return None


# ---- Output assignment helpers (shared by vectorized & row-wise) ----


def _assign_dict_output(
    data: pd.DataFrame,
    result: Any,
    task_def: dict,
) -> None:
    """Assign dict/DataFrame results back to *data*."""
    func_name = task_def["func_name"]
    if isinstance(result, pd.DataFrame):
        result = _align_result_index(result, data.index)
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
    """Assign a tuple/list/DataFrame/Series result to declared output columns."""
    col_names = task_def["col_names"]
    func_name = task_def["func_name"]
    type_map = task_def["type_map"]

    if not col_names:
        return

    # --- Single output column ---
    if len(col_names) == 1:
        data[col_names[0]] = _safe_values(result, data.index)
        apply_types(data, type_map)
        return

    # --- Multi output column ---
    if isinstance(result, pd.DataFrame):
        result = _align_result_index(result, data.index)
        if result.shape[1] != len(col_names):
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned {result.shape[1]} columns, "
                f"but outputs specifies {len(col_names)} columns: {col_names}."
            )
        for i, col in enumerate(col_names):
            data[col] = _safe_values(result.iloc[:, i], data.index)
    elif isinstance(result, pd.Series):
        if len(col_names) != 1:
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned a Series, but outputs specifies "
                f"{len(col_names)} columns: {col_names}."
            )
        data[col_names[0]] = _safe_values(result, data.index)
    elif isinstance(result, (list, tuple)):
        if len(result) != len(col_names):
            raise PipewiseOutputAssignmentError(
                f"Function '{func_name}' returned {len(result)} values, but "
                f"outputs specifies {len(col_names)} columns: {col_names}."
            )
        for i, col in enumerate(col_names):
            data[col] = _safe_values(result[i], data.index)
    else:
        raise PipewiseOutputAssignmentError(
            f"Function '{func_name}' must return a tuple/list or DataFrame "
            f"for outputs {col_names}."
        )
    apply_types(data, type_map)


def _safe_values(value: Any, target_index: pd.Index) -> Any:
    """Extract values respecting *target_index* alignment.

    - ``pd.Series`` → ``.values`` after index alignment
    - ``pd.DataFrame`` single column → ``.values``
    - scalar / list / array → returned as-is
    """
    if isinstance(value, pd.Series):
        return value.reindex(target_index).values
    if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        return value.iloc[:, 0].reindex(target_index).values
    return value


def _align_result_index(result: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    """Reindex *result* to match *target_index*, without mutating."""
    if not result.index.equals(target_index):
        return result.reindex(target_index)
    return result
