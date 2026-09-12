# Changelog

All notable changes to this project will be documented in this file.

## 2.0.0 - 2026-09-12

### Breaking changes

- **Vectorized → row-wise fallback is now opt-in.** `register(...)` defaults to
  `fallback_on_vectorized_error=False`. Previously an incompatible function was
  retried row by row; because the vectorized call has already run once, that
  retry re-executed the function body and could repeat side effects. The failure
  now surfaces directly with a hint pointing at the fix.
  **Migration:** pass `fallback_on_vectorized_error=True` (or use
  `vectorized=False`) to keep the previous behaviour.
- The class-level schema/rollback delegation methods (`_validate_frame_schema`,
  `_matches_dtype`, `_validate_dtype`, `_apply_types`, `_rollback`) were unused
  shims and have been removed.

### Added

- Vectorized-hazard detection now also flags a column parameter used as a branch
  condition (`if col > 0:`, `if col:`, `x if col else y`).
- Registration warns when a parameter with a default value shadows a same-named
  column, which was previously ignored silently.
- Vectorized functions may return a 2-D `numpy.ndarray` for multi-column output,
  and an `(n, 1)` array for a single output column.
- `py.typed` marker (PEP 561) so type checkers see the annotations.

### Changed

- Task metadata is now a `TaskDef` `NamedTuple`; `Pipewise.tasks` returns
  `TaskSummary` entries (still plain-tuple compatible).
- Row-wise execution uses `DataFrame.itertuples` instead of
  `DataFrame.apply(axis=1)` — measured ~5x faster on a 200k-row frame. Row values
  are now Python-native (`int`/`float`) rather than numpy scalars.
- Schema validation computes the null-free view of each column once per check.
- `run(task=...)` now also validates and reports a task's input columns.

### Fixed

- AST hazard detection no longer warns about local variables and helper objects;
  it only considers actual column inputs.
- Output-assignment failures (wrong length/shape) are never mistaken for a
  vectorization incompatibility and no longer trigger a fallback.
- Ragged row-wise multi-column results raise `PipewiseOutputAssignmentError`
  naming the offending row, instead of an opaque pandas error.
- Length-mismatched results raise a clear error instead of being silently
  broadcast across every row.
- `run(task=...)` ambiguity message interpolates the task name.

### Internal

- `core.py` split into `_hazards.py`, `_tasks.py`, `_assign.py` and
  `_execution.py`, leaving `core.py` as the orchestration facade.
- CI: ruff lint, 80% coverage gate, Python 3.13, and publishing now gated on tests.

## 1.1.0 - 2026-07-08

### Added
- **AST-based vectorized-hazard detection**: `register()` now scans function source code
  for common vectorized-incompatible patterns (`len()`, `isinstance()`, `.split()`,
  `.strip()`, `.lower()`, `.upper()`, `type()`, `x[0]`, `x['key']`) and emits
  `logger.warning` at registration time.
- **Index-alignment protection**: `_assign_columns` and `_assign_dict_output` now
  reindex DataFrame results to match the target DataFrame's index, preventing NaN
  corruption from misaligned returns.
- **Series return support**: single-output tasks can now return a `pd.Series` object
  safely, with automatic index reindexing.
- **Fallback notification**: vectorized → row-wise fallback now emits both
  `logger.info` and `RuntimeWarning`, making the behaviour observable even when
  warnings are suppressed for tqdm.

### Changed
- `_should_fallback()` now catches `AttributeError` in addition to `TypeError`
  and `ValueError`, ensuring functions using scalar-only string methods
  (e.g. `sc.split(',')`) auto-fallback to row-wise execution.
- `plan()` now emits via `logging.info` instead of `print()`.

### Fixed
- DataFrame result assignment respects target index alignment.
- `PIPEWISE_ANALYSIS.md` path reference corrected.

### Internal
- Schema validation and type-matching logic extracted to `_schema.py` (private module).
- Output assignment logic unified into `_assign_columns` / `_assign_dict_output`
  module-level helpers shared by vectorized and row-wise paths.
- `SchemaRule` type alias exported from the public API.

## 1.0.1 - 2026-05-16

- Added `run(task="task_name")` support to execute a single registered task for faster debugging.
- Added `PipewiseTaskSelectionError` for missing, invalid, or ambiguous task selection.
- Added tests covering single-task execution and task selection edge cases.
