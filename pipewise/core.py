import inspect
import warnings
from numbers import Integral, Real
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from .errors import (
    PipewiseExecutionError,
    PipewiseGroupByError,
    PipewiseInputColumnError,
    PipewiseInputSchemaError,
    PipewiseOutputAssignmentError,
    PipewiseOutputSchemaError,
    PipewiseRegistrationError,
    PipewiseTypeConversionError,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


SchemaRule = Union[type, Tuple[type, ...], str, Dict[str, Any]]


class Pipewise:
    """
    A pipeline tool for functional transformation of a pandas DataFrame.

    Register processing functions with the ``register`` decorator. Function
    parameter names map to DataFrame column names. Return values can be written
    back as new columns. Multiple functions execute sequentially and can build
    on each other's outputs.
    """

    _SCHEMA_KEYS = {"dtype", "nullable", "allowed_values", "min", "max"}
    _FALLBACK_ERROR_MARKERS = (
        "truth value of a series is ambiguous",
        "cannot convert the series to",
        "'series' object cannot be interpreted as an integer",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        input_schema: Optional[Dict[str, SchemaRule]] = None,
    ):
        self.data = data
        self._input_schema = self._normalize_schema(input_schema, "input_schema")
        self._tasks: List[dict] = []

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
        """
        Register a processing function. Can be used as a decorator with or
        without arguments.
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

        col_names, type_map, dict_mode = self._normalize_outputs(outputs)
        normalized_input_schema = self._normalize_schema(
            input_schema, "input_schema"
        )
        normalized_output_schema = self._normalize_schema(
            output_schema, "output_schema"
        )
        merged_output_schema = self._merge_output_schema(
            normalized_output_schema,
            type_map,
            func.__name__,
        )
        normalized_groupby = self._normalize_groupby(groupby, func.__name__)
        self._validate_output_schema_targets(
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

    def run(self, inplace: bool = False) -> pd.DataFrame:
        """
        Execute all registered functions in order and return the modified
        DataFrame.
        """
        data = self.data if inplace else self.data.copy()
        self._validate_frame_schema(
            data,
            self._input_schema,
            phase="pipeline input",
            exc_cls=PipewiseInputSchemaError,
        )

        snapshot = data.copy()
        iterator = _tqdm(self._tasks, ncols=60, colour="green") if _tqdm else self._tasks
        current_task_name = None

        try:
            for task in iterator:
                current_task_name = task["func_name"]
                if _tqdm is not None:
                    iterator.set_description(current_task_name)
                self._execute(data, task)
        except Exception as exc:
            self._rollback(data, snapshot)
            raise PipewiseExecutionError(current_task_name) from exc

        return data

    @property
    def tasks(self) -> List[Tuple[str, Any, Any, bool]]:
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
        for i, task in enumerate(self._tasks):
            if task["func"] is func:
                self._tasks.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all registered tasks."""
        self._tasks.clear()

    def plan(self) -> None:
        """Pretty-print the current execution plan."""
        if not self._tasks:
            print("No tasks registered.")
            return
        print("Execution Plan:")
        header = (
            f"{'#':>3}  {'Function':<22} {'Outputs':<28} "
            f"{'GroupBy':<12} {'Vec':<5}"
        )
        print(header)
        print("-" * len(header))
        for i, task in enumerate(self._tasks):
            outputs = str(task["col_names"] if not task["dict_mode"] else "dict")
            outputs = outputs or "side-effect"
            gb = str(task["groupby"] or "-")
            vec = "Y" if task["vectorized"] else "N"
            print(f"{i+1:>3}. {task['func_name']:<22} {outputs:<28} {gb:<12} {vec:<5}")

    @staticmethod
    def _normalize_outputs(outputs):
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

    @staticmethod
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

    @classmethod
    def _normalize_schema(cls, schema, schema_name: str):
        if schema is None:
            return {}
        if not isinstance(schema, dict):
            raise PipewiseRegistrationError(
                f"{schema_name} must be a dict mapping column names to schema rules."
            )

        normalized = {}
        for column, rule in schema.items():
            if not isinstance(column, str):
                raise PipewiseRegistrationError(
                    f"{schema_name} column names must be strings, got {type(column)}."
                )
            if isinstance(rule, dict):
                unknown_keys = set(rule) - cls._SCHEMA_KEYS
                if unknown_keys:
                    raise PipewiseRegistrationError(
                        f"{schema_name} for column '{column}' contains unsupported "
                        f"keys: {sorted(unknown_keys)}."
                    )
                normalized[column] = dict(rule)
            else:
                normalized[column] = {"dtype": rule}
        return normalized

    @staticmethod
    def _merge_output_schema(output_schema, type_map, func_name: str):
        merged = {column: dict(rule) for column, rule in output_schema.items()}
        if not type_map:
            return merged

        for column, dtype in type_map.items():
            rule = merged.setdefault(column, {})
            existing = rule.get("dtype")
            if existing is not None and existing != dtype:
                raise PipewiseRegistrationError(
                    f"Function '{func_name}' declares conflicting dtype rules for "
                    f"output column '{column}'."
                )
            rule["dtype"] = dtype
        return merged

    @staticmethod
    def _validate_output_schema_targets(output_schema, col_names, dict_mode, func_name):
        if not output_schema:
            return
        if not dict_mode and not col_names:
            raise PipewiseRegistrationError(
                f"Function '{func_name}' defines output_schema but has no outputs."
            )
        if dict_mode:
            return
        invalid_targets = [column for column in output_schema if column not in col_names]
        if invalid_targets:
            raise PipewiseRegistrationError(
                f"Function '{func_name}' output_schema references undeclared outputs "
                f"{invalid_targets}."
            )

    @staticmethod
    def _rollback(data, snapshot):
        """Restore *data* to the state captured in *snapshot*."""
        cols_to_remove = [c for c in data.columns if c not in snapshot.columns]
        if cols_to_remove:
            data.drop(columns=cols_to_remove, inplace=True)
        for col in snapshot.columns:
            data[col] = snapshot[col].copy()

    @staticmethod
    def _parse_inputs(func):
        """
        Extract explicit input column names from function signature.
        Returns ``(input_cols, has_kwargs)``.
        """
        sig = inspect.signature(func)
        input_cols: List[str] = []
        has_kwargs = False
        for parameter in sig.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                has_kwargs = True
            elif parameter.default is inspect.Parameter.empty:
                input_cols.append(parameter.name)
        return input_cols, has_kwargs

    def _execute(self, data: pd.DataFrame, task: dict):
        func = task["func"]
        input_cols, has_kwargs = self._parse_inputs(func)

        self._validate_input_columns(data, input_cols, func.__name__)
        self._validate_frame_schema(
            data,
            task["input_schema"],
            phase=f"task input for '{func.__name__}'",
            exc_cls=PipewiseInputSchemaError,
        )

        if task["groupby"]:
            self._execute_groupby(data, task, input_cols, has_kwargs)
        else:
            self._execute_on_frame(data, task, input_cols, has_kwargs)

        self._validate_frame_schema(
            data,
            task["output_schema"],
            phase=f"task output for '{func.__name__}'",
            exc_cls=PipewiseOutputSchemaError,
        )

    @staticmethod
    def _validate_input_columns(data, input_cols, func_name: str):
        missing = [column for column in input_cols if column not in data.columns]
        if missing:
            raise PipewiseInputColumnError(
                f"Function '{func_name}' requires input columns {missing}, but they "
                "are missing from the DataFrame."
            )

    def _execute_on_frame(self, data, task, input_cols, has_kwargs):
        if task["vectorized"]:
            try:
                self._execute_vectorized(
                    data,
                    task["func"],
                    input_cols,
                    has_kwargs,
                    task["col_names"],
                    task["type_map"],
                    task["dict_mode"],
                )
                return
            except Exception as exc:
                if self._should_fallback_to_rowwise(task, exc):
                    warnings.warn(
                        (
                            f"Function '{task['func_name']}' fell back to row-wise "
                            f"execution because vectorized execution is incompatible: "
                            f"{exc}"
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    raise

        self._execute_rowwise(
            data,
            task["func"],
            input_cols,
            has_kwargs,
            task["col_names"],
            task["type_map"],
            task["dict_mode"],
        )

    def _should_fallback_to_rowwise(self, task, exc: Exception) -> bool:
        if not task["fallback_on_vectorized_error"]:
            return False
        message = str(exc).lower()
        return any(marker in message for marker in self._FALLBACK_ERROR_MARKERS)

    def _execute_rowwise(
        self,
        data,
        func,
        input_cols,
        has_kwargs,
        col_names,
        type_map,
        dict_mode,
    ):
        """Row-wise execution via ``data.apply(func, axis=1)``."""
        if has_kwargs:
            extra_cols = [column for column in data.columns if column not in input_cols]

            def row_func(row):
                args = [row[column] for column in input_cols]
                kwargs = {column: row[column] for column in extra_cols}
                return func(*args, **kwargs)

        else:

            def row_func(row):
                args = [row[column] for column in input_cols]
                return func(*args)

        if dict_mode:
            result_series = data.apply(row_func, axis=1)
            invalid_rows = [
                index
                for index, value in result_series.items()
                if not isinstance(value, dict)
            ]
            if invalid_rows:
                raise PipewiseOutputAssignmentError(
                    f"Function '{func.__name__}' must return a dict for each row when "
                    f"outputs='dict'. Invalid rows: {invalid_rows[:5]}."
                )
            updates = pd.DataFrame(result_series.tolist(), index=data.index)
            for column in updates.columns:
                data[column] = updates[column]
            return

        if not col_names:
            data.apply(row_func, axis=1)
            return

        if len(col_names) == 1:
            result = data.apply(row_func, axis=1)
            data[col_names[0]] = result
            self._apply_types(data, type_map)
            return

        expanded = data.apply(row_func, axis=1, result_type="expand")
        n_returned = 1 if expanded.ndim == 1 else expanded.shape[1]
        if n_returned != len(col_names):
            raise PipewiseOutputAssignmentError(
                f"Function '{func.__name__}' returned {n_returned} values, but "
                f"outputs specifies {len(col_names)} columns: {col_names}."
            )
        if expanded.ndim == 1:
            data[col_names[0]] = expanded
        else:
            expanded.columns = col_names
            for column in col_names:
                data[column] = expanded[column]
        self._apply_types(data, type_map)

    def _execute_vectorized(
        self,
        data,
        func,
        input_cols,
        has_kwargs,
        col_names,
        type_map,
        dict_mode,
    ):
        """Vectorized execution: pass entire Series/columns to *func*."""
        args = [data[column] for column in input_cols]
        if has_kwargs:
            extra_cols = {
                column: data[column] for column in data.columns if column not in input_cols
            }
            result = func(*args, **extra_cols)
        else:
            result = func(*args)

        if dict_mode:
            self._assign_dict_output(data, result, func.__name__)
            return

        if not col_names:
            return

        if len(col_names) == 1:
            data[col_names[0]] = result
            self._apply_types(data, type_map)
            return

        if isinstance(result, pd.DataFrame):
            if result.shape[1] != len(col_names):
                raise PipewiseOutputAssignmentError(
                    f"Function '{func.__name__}' returned {result.shape[1]} columns, "
                    f"but outputs specifies {len(col_names)} columns: {col_names}."
                )
            for i, column in enumerate(col_names):
                data[column] = result.iloc[:, i]
        elif isinstance(result, (list, tuple)):
            if len(result) != len(col_names):
                raise PipewiseOutputAssignmentError(
                    f"Function '{func.__name__}' returned {len(result)} values, but "
                    f"outputs specifies {len(col_names)} columns: {col_names}."
                )
            for i, column in enumerate(col_names):
                data[column] = result[i]
        else:
            raise PipewiseOutputAssignmentError(
                f"Function '{func.__name__}' must return a tuple/list or DataFrame "
                f"for outputs {col_names}."
            )
        self._apply_types(data, type_map)

    @staticmethod
    def _assign_dict_output(data, result, func_name: str):
        if isinstance(result, pd.DataFrame):
            for column in result.columns:
                data[column] = result[column]
            return
        if isinstance(result, dict):
            for column, value in result.items():
                data[column] = value
            return
        raise PipewiseOutputAssignmentError(
            f"Function '{func_name}' must return a dict or DataFrame when "
            "outputs='dict' in vectorized mode."
        )

    def _execute_groupby(self, data, task, input_cols, has_kwargs):
        """Split-apply-combine: group by column(s), apply function per group."""
        groupby_cols = task["groupby"]
        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]

        missing_gb = [column for column in groupby_cols if column not in data.columns]
        if missing_gb:
            raise PipewiseGroupByError(
                f"GroupBy columns {missing_gb} not found in the DataFrame."
            )

        def apply_per_group(group: pd.DataFrame) -> pd.DataFrame:
            result = group.copy()
            self._execute_on_frame(result, task, input_cols, has_kwargs)
            return result

        results = data.groupby(groupby_cols, group_keys=False).apply(apply_per_group)
        for column in results.columns:
            if column in groupby_cols:
                continue
            data[column] = results[column]

    @classmethod
    def _validate_frame_schema(cls, data, schema, phase: str, exc_cls):
        if not schema:
            return
        for column, rule in schema.items():
            if column not in data.columns:
                raise exc_cls(
                    f"{phase} expects column '{column}', but it is missing."
                )
            series = data[column]
            cls._validate_nullable(series, rule, column, phase, exc_cls)
            cls._validate_allowed_values(series, rule, column, phase, exc_cls)
            cls._validate_numeric_bounds(series, rule, column, phase, exc_cls)
            cls._validate_dtype(series, rule, column, phase, exc_cls)

    @staticmethod
    def _validate_nullable(series, rule, column, phase: str, exc_cls):
        if rule.get("nullable", True):
            return
        if series.isna().any():
            raise exc_cls(
                f"{phase} column '{column}' contains null values, but nullable=False."
            )

    @staticmethod
    def _validate_allowed_values(series, rule, column, phase: str, exc_cls):
        allowed_values = rule.get("allowed_values")
        if allowed_values is None:
            return
        invalid_mask = ~series.dropna().isin(list(allowed_values))
        if invalid_mask.any():
            invalid_values = series.dropna()[invalid_mask].unique().tolist()
            raise exc_cls(
                f"{phase} column '{column}' contains values outside allowed_values: "
                f"{invalid_values[:5]}."
            )

    @staticmethod
    def _validate_numeric_bounds(series, rule, column, phase: str, exc_cls):
        non_null = series.dropna()
        if non_null.empty:
            return

        min_value = rule.get("min")
        if min_value is not None:
            try:
                if (non_null < min_value).any():
                    observed = non_null[non_null < min_value].iloc[0]
                    raise exc_cls(
                        f"{phase} column '{column}' contains value {observed!r} "
                        f"below min={min_value!r}."
                    )
            except TypeError as exc:
                raise exc_cls(
                    f"{phase} column '{column}' cannot be compared with min={min_value!r}."
                ) from exc

        max_value = rule.get("max")
        if max_value is not None:
            try:
                if (non_null > max_value).any():
                    observed = non_null[non_null > max_value].iloc[0]
                    raise exc_cls(
                        f"{phase} column '{column}' contains value {observed!r} "
                        f"above max={max_value!r}."
                    )
            except TypeError as exc:
                raise exc_cls(
                    f"{phase} column '{column}' cannot be compared with max={max_value!r}."
                ) from exc

    @classmethod
    def _validate_dtype(cls, series, rule, column, phase: str, exc_cls):
        expected = rule.get("dtype")
        if expected is None:
            return
        if cls._matches_dtype(series, expected):
            return
        raise exc_cls(
            f"{phase} column '{column}' has dtype {series.dtype}, which does not "
            f"match expected {expected!r}."
        )

    @classmethod
    def _matches_dtype(cls, series, expected) -> bool:
        non_null = series.dropna()
        if isinstance(expected, str):
            expected_lower = expected.lower()
            if expected_lower in {"number", "numeric"}:
                return pd.api.types.is_numeric_dtype(series)
            if expected_lower == "integer":
                return pd.api.types.is_integer_dtype(series) or non_null.map(
                    lambda value: isinstance(value, Integral) and not isinstance(value, bool)
                ).all()
            if expected_lower == "float":
                return pd.api.types.is_float_dtype(series) or non_null.map(
                    lambda value: isinstance(value, Real)
                    and not isinstance(value, Integral)
                    and not isinstance(value, bool)
                ).all()
            if expected_lower in {"bool", "boolean"}:
                return pd.api.types.is_bool_dtype(series) or non_null.map(
                    lambda value: isinstance(value, bool)
                ).all()
            if expected_lower == "string":
                return pd.api.types.is_string_dtype(series) or non_null.map(
                    lambda value: isinstance(value, str)
                ).all()
            if expected_lower == "datetime":
                return pd.api.types.is_datetime64_any_dtype(series)
            try:
                expected_dtype = pd.api.types.pandas_dtype(expected)
            except TypeError:
                return False
            return pd.api.types.is_dtype_equal(series.dtype, expected_dtype)

        if isinstance(expected, type):
            if expected is int:
                return cls._matches_dtype(series, "integer")
            if expected is float:
                return cls._matches_dtype(series, "float")
            if expected is bool:
                return cls._matches_dtype(series, "bool")
            if expected is str:
                return cls._matches_dtype(series, "string")
            if expected is object:
                return True
            return non_null.map(lambda value: isinstance(value, expected)).all()

        if isinstance(expected, tuple) and expected and all(
            isinstance(item, type) for item in expected
        ):
            return non_null.map(lambda value: isinstance(value, expected)).all()

        return False

    @staticmethod
    def _apply_types(data: pd.DataFrame, type_map: Optional[Dict[str, type]]):
        if not type_map:
            return
        for column, dtype in type_map.items():
            if column not in data.columns:
                continue
            try:
                data[column] = data[column].astype(dtype)
            except (ValueError, TypeError) as exc:
                raise PipewiseTypeConversionError(
                    f"Column '{column}' cannot be cast to {dtype.__name__}: {exc}"
                ) from exc
