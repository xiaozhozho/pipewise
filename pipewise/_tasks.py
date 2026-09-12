"""Registered-task metadata and function-signature parsing for Pipewise."""

from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Union

from .errors import PipewiseInputColumnError, PipewiseRegistrationError

logger = logging.getLogger(__name__)


class TaskDef(NamedTuple):
    """Metadata for one registered pipeline step.

    A ``NamedTuple`` (rather than a dataclass) keeps the fields immutable and
    still unpackable, while remaining cheap to create and inspect.
    """

    func: Callable
    func_name: str
    col_names: List[str]
    type_map: Optional[Dict[str, type]]
    dict_mode: bool
    groupby: Optional[Union[str, List[str]]]
    vectorized: bool
    input_schema: Dict[str, Dict[str, Any]]
    output_schema: Dict[str, Dict[str, Any]]
    input_cols: List[str]
    has_kwargs: bool
    fallback_on_vectorized_error: bool

    @property
    def output_label(self) -> Any:
        """Human-readable output description used by ``tasks`` / ``plan``."""
        return "dict" if self.dict_mode else self.col_names


class TaskSummary(NamedTuple):
    """Concise per-task view returned by :attr:`Pipewise.tasks`."""

    func_name: str
    outputs: Any
    groupby: Optional[Union[str, List[str]]]
    vectorized: bool


@dataclass(frozen=True)
class ParsedSignature:
    """Result of inspecting a registered function's signature."""

    input_cols: List[str]
    has_kwargs: bool
    defaulted: List[str]


def parse_signature(func: Callable) -> ParsedSignature:
    """Split a function's parameters into input columns and extras.

    Parameters without a default become input columns — their names are looked
    up in the DataFrame. Variadic ``*args`` is ignored, and parameters that
    carry a default are recorded separately so they can be reported when a
    same-named column would otherwise be silently ignored.
    """
    signature = inspect.signature(func)
    input_cols: List[str] = []
    defaulted: List[str] = []
    has_kwargs = False

    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_kwargs = True
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.default is inspect.Parameter.empty:
            input_cols.append(param.name)
        else:
            defaulted.append(param.name)

    return ParsedSignature(input_cols, has_kwargs, defaulted)


def validate_input_columns(data, task_def: TaskDef) -> None:
    """Raise if any required input column is missing from *data*."""
    missing = [col for col in task_def.input_cols if col not in data.columns]
    if missing:
        raise PipewiseInputColumnError(
            f"Function '{task_def.func_name}' requires input columns {missing}, "
            "but they are missing from the DataFrame."
        )


def warn_on_shadowed_defaults(
    func_name: str,
    defaulted_params: Iterable[str],
    columns: Iterable[str],
) -> None:
    """Warn when a defaulted parameter shadows a same-named column.

    A parameter with a default is not treated as an input column, so a column
    of the same name is silently ignored in favour of the default. That is easy
    to miss, so it is reported at registration time.
    """
    columns = set(columns)
    shadowed = [name for name in defaulted_params if name in columns]
    if not shadowed:
        return
    message = (
        f"Function '{func_name}' declares parameter(s) {shadowed} with a default "
        f"value, and the DataFrame also has column(s) with the same name. The "
        f"column(s) will be ignored and the default used instead. Remove the "
        f"default, or rename the parameter, to read the column."
    )
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def normalize_outputs(
    outputs: Optional[Union[str, List[str], Iterable[str], Dict[str, type]]],
):
    """Parse ``outputs`` into ``(col_names, type_map, dict_mode)``."""
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


def normalize_groupby(
    groupby: Optional[Union[str, List[str]]],
    func_name: str,
) -> Optional[Union[str, List[str]]]:
    """Validate the ``groupby`` specification, preserving its original form."""
    if groupby is None:
        return None
    if isinstance(groupby, str):
        return groupby or None
    if isinstance(groupby, (list, tuple)):
        if not groupby:
            return None
        if all(isinstance(col, str) for col in groupby):
            return list(groupby)
    raise PipewiseRegistrationError(
        f"Function '{func_name}' received invalid groupby={groupby!r}."
    )
