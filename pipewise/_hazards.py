"""AST-based detection of vectorized-incompatible function bodies.

In vectorized mode Pipewise calls a registered function with whole
``pandas.Series`` objects. Some common Python idioms either silently do the
wrong thing on a Series (``len(s)`` returns the row count) or raise
(``if s > 0``). This module inspects a function's source at registration time
so those hazards surface before the pipeline ever runs.

Only names that are actual function parameters — i.e. the DataFrame columns
passed to the function — are considered. Helpers and local variables can never
trigger a warning.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from typing import Callable, Dict, Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_HAZARD_PATTERNS: Dict[str, str] = {
    "Call@len": (
        "`len(%s)` returns the number of rows when applied to a Series, not the "
        "length of each element. Use `.str.len()` (vectorized) or "
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
        "`%s` raises AttributeError on a Series — use `.str.split()` "
        "(vectorized) or `vectorized=False` (row-wise)."
    ),
    "Call@.strip": (
        "`%s` raises AttributeError on a Series — use `.str.strip()` "
        "(vectorized) or `vectorized=False` (row-wise)."
    ),
    "Call@.lower": (
        "`%s` raises AttributeError on a Series — use `.str.lower()` "
        "(vectorized) or `vectorized=False` (row-wise)."
    ),
    "Call@.upper": (
        "`%s` raises AttributeError on a Series — use `.str.upper()` "
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
    "If@Test": (
        "`%s` is used as a branch condition — evaluating a Series for truth "
        "raises ValueError (`truth value of a Series is ambiguous`). Use "
        "`vectorized=False` to branch per row, or a vectorized expression such "
        "as `numpy.where` / `Series.where`."
    ),
}

_STRING_METHODS = ("split", "strip", "lower", "upper")
_NAME_BUILTINS = ("len", "isinstance", "type")


def warn_if_vectorized_hazard(func: Callable, param_names: Iterable[str]) -> None:
    """Warn when *func* looks incompatible with vectorized execution.

    *param_names* restricts detection to the function's actual column inputs,
    which keeps local variables and helper objects from producing false
    positives.
    """
    params = set(param_names)
    if not params:
        return

    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return  # not inspectable (built-in, C extension, REPL) — skip

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    detector = _VectorizedHazardVisitor(params)
    detector.visit(tree)

    for (category, _name), argument in sorted(detector.hazards.items()):
        logger.warning(
            "Function '%s' may be vectorized-incompatible: %s",
            func.__name__,
            _HAZARD_PATTERNS[category] % argument,
        )


class _VectorizedHazardVisitor(ast.NodeVisitor):
    """Walk a function body and record vectorized-incompatible patterns."""

    def __init__(self, param_names: Set[str]):
        self._params = param_names
        # (category, subject) -> text substituted into the hazard message
        self.hazards: Dict[Tuple[str, str], str] = {}

    def _record(self, category: str, subject: str, argument: str) -> None:
        if category not in _HAZARD_PATTERNS:
            return
        self.hazards.setdefault((category, subject), argument)

    # --- Calls -------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self._check_builtin(node, node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self._check_string_method(node)
        self.generic_visit(node)

    def _check_builtin(self, node: ast.Call, name: str) -> None:
        if name not in _NAME_BUILTINS or not node.args:
            return
        arg = node.args[0]
        if isinstance(arg, ast.Name) and arg.id in self._params:
            self._record(f"Call@{name}", arg.id, arg.id)

    def _check_string_method(self, node: ast.Call) -> None:
        method = node.func.attr
        if method not in _STRING_METHODS:
            return
        value = node.func.value
        if isinstance(value, ast.Name) and value.id in self._params:
            text = _expr_source(node) or f"{value.id}.{method}()"
            self._record(f"Call@.{method}", value.id, text)

    # --- Subscripts --------------------------------------------------

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self._params:
            name = node.value.id
            if isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, int):
                    self._record("Subscript@int", name, f"{name}[{node.slice.value}]")
                elif isinstance(node.slice.value, str):
                    self._record("Subscript@str", name, f"{name}['{node.slice.value}']")
        self.generic_visit(node)

    # --- Branch conditions -------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        self._check_branch_test(node.test)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._check_branch_test(node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._check_branch_test(node.test)
        self.generic_visit(node)

    def _check_branch_test(self, test: ast.expr) -> None:
        subject = _param_in_condition(test, self._params)
        if subject is not None:
            self._record("If@Test", subject, _expr_source(test) or subject)


def _param_in_condition(node: ast.AST, params: Set[str]) -> Optional[str]:
    """Return the first parameter name used in a boolean *node*, if any."""
    if isinstance(node, ast.Name):
        return node.id if node.id in params else None
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        for operand in operands:
            found = _param_in_condition(operand, params)
            if found is not None:
                return found
        return None
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            found = _param_in_condition(value, params)
            if found is not None:
                return found
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _param_in_condition(node.operand, params)
    return None


def _expr_source(node: ast.AST) -> Optional[str]:
    """Return a short source representation of *node*, best-effort."""
    try:
        return ast.unparse(node)
    except Exception:
        return None
