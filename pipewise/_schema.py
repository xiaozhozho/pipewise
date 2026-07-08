"""Schema validation and type matching logic for Pipewise."""

from __future__ import annotations

import logging
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .errors import (
    PipewiseRegistrationError,
    PipewiseTypeConversionError,
)

SchemaRule = Union[type, Tuple[type, ...], str, Dict[str, Any]]

_SCHEMA_KEYS = {"dtype", "nullable", "allowed_values", "min", "max"}

logger = logging.getLogger(__name__)


def normalize_schema(
    schema: Optional[Dict[str, SchemaRule]],
    schema_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Normalize a schema dict into uniform ``{col: {key: value, ...}}`` form."""
    if schema is None:
        return {}
    if not isinstance(schema, dict):
        raise PipewiseRegistrationError(
            f"{schema_name} must be a dict mapping column names to schema rules."
        )

    normalized: Dict[str, Dict[str, Any]] = {}
    for column, rule in schema.items():
        if not isinstance(column, str):
            raise PipewiseRegistrationError(
                f"{schema_name} column names must be strings, got {type(column)}."
            )
        if isinstance(rule, dict):
            unknown_keys = set(rule) - _SCHEMA_KEYS
            if unknown_keys:
                raise PipewiseRegistrationError(
                    f"{schema_name} for column '{column}' contains unsupported "
                    f"keys: {sorted(unknown_keys)}."
                )
            normalized[column] = dict(rule)
        else:
            normalized[column] = {"dtype": rule}
    return normalized


def merge_output_schema(
    output_schema: Dict[str, Dict[str, Any]],
    type_map: Optional[Dict[str, type]],
    func_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Merge :class:`type`-based output declarations into a full schema dict."""
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


def validate_output_schema_targets(
    output_schema: Dict[str, Dict[str, Any]],
    col_names: List[str],
    dict_mode: bool,
    func_name: str,
) -> None:
    """Ensure *output_schema* references only declared output columns."""
    if not output_schema:
        return
    if not dict_mode and not col_names:
        raise PipewiseRegistrationError(
            f"Function '{func_name}' defines output_schema but has no outputs."
        )
    if dict_mode:
        return
    invalid_targets = [
        column for column in output_schema if column not in col_names
    ]
    if invalid_targets:
        raise PipewiseRegistrationError(
            f"Function '{func_name}' output_schema references undeclared outputs "
            f"{invalid_targets}."
        )


def validate_frame_schema(
    data: pd.DataFrame,
    schema: Dict[str, Dict[str, Any]],
    phase: str,
    exc_cls,
) -> None:
    """Validate *data* columns against a normalized schema."""
    if not schema:
        return
    for column, rule in schema.items():
        if column not in data.columns:
            raise exc_cls(f"{phase} expects column '{column}', but it is missing.")
        series = data[column]
        _validate_nullable(series, rule, column, phase, exc_cls)
        _validate_allowed_values(series, rule, column, phase, exc_cls)
        _validate_numeric_bounds(series, rule, column, phase, exc_cls)
        _validate_dtype(series, rule, column, phase, exc_cls)


def _validate_nullable(series: pd.Series, rule: Dict[str, Any], column: str, phase: str, exc_cls) -> None:
    if rule.get("nullable", True):
        return
    if series.isna().any():
        raise exc_cls(
            f"{phase} column '{column}' contains null values, but nullable=False."
        )


def _validate_allowed_values(series: pd.Series, rule: Dict[str, Any], column: str, phase: str, exc_cls) -> None:
    allowed = rule.get("allowed_values")
    if allowed is None:
        return
    invalid_mask = ~series.dropna().isin(list(allowed))
    if invalid_mask.any():
        invalid_values = series.dropna()[invalid_mask].unique().tolist()
        raise exc_cls(
            f"{phase} column '{column}' contains values outside allowed_values: "
            f"{invalid_values[:5]}."
        )


def _validate_numeric_bounds(series: pd.Series, rule: Dict[str, Any], column: str, phase: str, exc_cls) -> None:
    non_null = series.dropna()
    if non_null.empty:
        return

    min_value = rule.get("min")
    if min_value is not None:
        _check_bound(non_null < min_value, series, column, phase, exc_cls, "below", "min", min_value)

    max_value = rule.get("max")
    if max_value is not None:
        _check_bound(non_null > max_value, series, column, phase, exc_cls, "above", "max", max_value)


def _check_bound(condition, series, column, phase, exc_cls, direction: str, bound_name: str, bound_value) -> None:
    try:
        if condition.any():
            observed = series[condition].iloc[0]
            raise exc_cls(
                f"{phase} column '{column}' contains value {observed!r} "
                f"{direction} {bound_name}={bound_value!r}."
            )
    except TypeError as exc:
        raise exc_cls(
            f"{phase} column '{column}' cannot be compared with "
            f"{bound_name}={bound_value!r}."
        ) from exc


def _validate_dtype(series: pd.Series, rule: Dict[str, Any], column: str, phase: str, exc_cls) -> None:
    expected = rule.get("dtype")
    if expected is None:
        return
    if matches_dtype(series, expected):
        return
    raise exc_cls(
        f"{phase} column '{column}' has dtype {series.dtype}, "
        f"which does not match expected {expected!r}."
    )


def matches_dtype(series: pd.Series, expected: SchemaRule) -> bool:
    """Check whether *series* dtype satisfies ``expected``.

    Accepts a pandas dtype string (``"integer"``, ``"float"``, ``"number"``,
    ``"bool"``, ``"string"``, ``"datetime"``, or a concrete ``pd.api.types.pandas_dtype``),
    a Python built-in type (``int``, ``float``, ``bool``, ``str``, ``object``),
    or a ``tuple`` of Python types.
    """
    non_null = series.dropna()

    if isinstance(expected, str):
        return _matches_dtype_string(expected.lower(), series, non_null)

    if isinstance(expected, type):
        return _matches_dtype_type(expected, series, non_null)

    if isinstance(expected, tuple) and expected and all(isinstance(item, type) for item in expected):
        return bool(non_null.map(lambda v: isinstance(v, expected)).all())

    return False


def _matches_dtype_string(expected_lower: str, series: pd.Series, non_null: pd.Series) -> bool:
    if expected_lower in {"number", "numeric"}:
        return bool(pd.api.types.is_numeric_dtype(series))
    if expected_lower == "integer":
        return bool(
            pd.api.types.is_integer_dtype(series)
            or non_null.map(
                lambda v: isinstance(v, Integral) and not isinstance(v, bool)
            ).all()
        )
    if expected_lower == "float":
        return bool(
            pd.api.types.is_float_dtype(series)
            or non_null.map(
                lambda v: isinstance(v, Real) and not isinstance(v, Integral) and not isinstance(v, bool)
            ).all()
        )
    if expected_lower in {"bool", "boolean"}:
        return bool(
            pd.api.types.is_bool_dtype(series)
            or non_null.map(lambda v: isinstance(v, bool)).all()
        )
    if expected_lower == "string":
        return bool(
            pd.api.types.is_string_dtype(series)
            or non_null.map(lambda v: isinstance(v, str)).all()
        )
    if expected_lower == "datetime":
        return bool(pd.api.types.is_datetime64_any_dtype(series))

    try:
        expected_dtype = pd.api.types.pandas_dtype(expected_lower)
    except TypeError:
        return False
    return bool(pd.api.types.is_dtype_equal(series.dtype, expected_dtype))


def _matches_dtype_type(expected: type, series: pd.Series, non_null: pd.Series) -> bool:
    mapping: Dict[type, str] = {int: "integer", float: "float", bool: "bool", str: "string"}
    if expected in mapping:
        return _matches_dtype_string(mapping[expected], series, non_null)
    if expected is object:
        return True
    return bool(non_null.map(lambda v: isinstance(v, expected)).all())


def apply_types(data: pd.DataFrame, type_map: Optional[Dict[str, type]]) -> None:
    """Cast output columns to their declared types."""
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
