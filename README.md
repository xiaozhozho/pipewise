# Pipewise

`Pipewise` is a lightweight `pandas.DataFrame` pipeline library for teams that
want reusable data-processing steps without adopting a heavyweight workflow
framework.

It helps you:

- register step functions with decorators
- map function arguments to DataFrame columns automatically
- write results back to one or more columns
- validate input and output schema rules
- run vectorized by default, row-by-row when a function needs it
- roll back all changes if any task fails

## Installation

```bash
pip install pipewise
```

For local development:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
pipewise = Pipewise(df)


@pipewise.register(outputs=["sum", "product"])
def calc(a, b):
    return a + b, a * b


result = pipewise.run()
print(result)
```

## Core Features

### Registering steps

`@pipewise.register(...)` accepts:

| `outputs` | Meaning |
|---|---|
| `None` | side effect only, nothing written back |
| `"col"` | single output column |
| `["c1", "c2"]` | fixed multi-column output |
| `{"col": type}` | typed output, cast after execution |
| `"dict"` | dynamic dict output, columns chosen per row |

Function parameters without a default are looked up as columns; `**kwargs`
receives every remaining column. A parameter that has a default is *not* read
from the frame — if a same-named column exists, registration warns that the
column will be ignored.

### Execution modes

```python
@pipewise.register(outputs="c")                      # vectorized (default)
def vec(a, b):
    return a + b


@pipewise.register(outputs="c", vectorized=False)    # one call per row
def row(a, b):
    if a > 10:          # a scalar branch — needs row-wise execution
        return a * 2
    return b
```

Vectorized mode passes whole `Series` objects, which is much faster. Row-wise
mode iterates with `DataFrame.itertuples` and receives Python scalars.

Registration scans the function's source and logs a warning for common
vectorized-incompatible patterns — `len(col)`, `isinstance(col, ...)`,
`col.split(...)`, `col[0]`, and a column used as a branch condition
(`if col > 0:`).

### Automatic fallback (opt-in)

If a vectorized call fails with a `TypeError`, `ValueError` or `AttributeError`,
Pipewise can retry the function row by row:

```python
@pipewise.register(outputs="c", fallback_on_vectorized_error=True)
def maybe_scalar(a):
    if a > 10:
        return "big"
    return "small"
```

The fallback is **off by default**: the vectorized call has already executed the
function body once, so a retry runs it again and may repeat side effects. Prefer
`vectorized=False` when a function is inherently row-wise.

### Grouped execution

```python
@pipewise.register(outputs="delta", groupby="group")
def within_group(value):
    return value - value.mean()
```

### Schema validation

Rules may be declared for pipeline input, task input and task output:

```python
Pipewise(df, input_schema={"a": {"dtype": "integer", "nullable": False, "min": 0}})

@pipewise.register(
    outputs="score",
    output_schema={"score": {"dtype": "number", "max": 100}},
)
def score(a):
    return a * 10
```

Supported keys: `dtype` (pandas dtype string or Python type), `nullable`,
`allowed_values`, `min`, `max`.

### Rollback

`run()` snapshots the frame first. If any task raises, every change made during
that run is rolled back before `PipewiseExecutionError` is raised, with the
original exception attached as `__cause__`.

### Task management

```python
pipewise.tasks          # list of (func_name, outputs, groupby, vectorized)
pipewise.plan()         # log the execution plan
pipewise.run(task="calc")   # run a single registered task
pipewise.remove(calc)
pipewise.clear()
```

### Unusual cell types

Columns may hold values that are not scalars. List, dict, set, frozenset,
tuple, `numpy.ndarray`, class objects and class instances are all supported in
both execution modes; a vectorized function may also return a 2-D
`numpy.ndarray` for multi-column output.

```python
df = pd.DataFrame({"vec": [np.array([1, 2]), np.array([3, 4, 5])]})


@pipewise.register(outputs="n", vectorized=False)
def count(vec):
    return len(vec)
```

## Package Structure

```text
pipewise/
  __init__.py     public interface and version
  core.py         Pipewise class: registration and orchestration
  _execution.py   vectorized / row-wise / grouped execution
  _assign.py      output assignment and shape guards
  _tasks.py       TaskDef metadata and signature parsing
  _schema.py      schema validation and dtype matching
  _hazards.py     AST-based vectorized-hazard detection
  errors.py       exception hierarchy
tests/
  test_pipewise.py
```

## Public Metadata

```python
from pipewise import __author__, __version__
```

- `__version__ = "2.0.0"`
- `__author__ = "XiaoZhouZhou"`

## Testing

```bash
python -m pytest -q
```

With coverage and lint, as CI runs them:

```bash
python -m pytest --cov=pipewise --cov-report=term-missing --cov-fail-under=80
ruff check pipewise tests
```

## Publish Checklist

1. Update `pipewise/__init__.py` and `pyproject.toml` versions.
2. Update `CHANGELOG.md`.
3. Run tests and lint.
4. Tag the release (`v*`); the publish workflow runs the tests before building.
