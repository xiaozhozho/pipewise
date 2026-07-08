# Changelog

All notable changes to this project will be documented in this file.

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
