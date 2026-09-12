# Pipewise

[![Run tests](https://github.com/xiaozhozho/pipewise/actions/workflows/test.yml/badge.svg)](https://github.com/xiaozhozho/pipewise/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pipewise.svg)](https://pypi.org/project/pipewise/)
[![Python versions](https://img.shields.io/pypi/pyversions/pipewise.svg)](https://pypi.org/project/pipewise/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`Pipewise` is a lightweight `pandas.DataFrame` pipeline library for teams that
want reusable data-processing steps without adopting a heavyweight workflow
framework.

Declare inputs by column name, chain steps with a decorator, and let Pipewise
write the results back:

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame(
    {"quantity": [2, 5], "unit_price": [10.0, 4.0], "discount": [0.0, 0.25]}
)
pipewise = Pipewise(df)


@pipewise.register(outputs=["gross", "net"])
def amount(quantity, unit_price, discount):
    gross = quantity * unit_price
    return gross, gross * (1 - discount)


print(pipewise.run().to_string(index=False))
```

```
 quantity  unit_price  discount  gross  net
        2        10.0      0.00   20.0 20.0
        5         4.0      0.25   20.0 15.0
```

It gives you:

- **Registration by decorator** — function parameter names map to columns automatically
- **Five output modes** — side-effect only, single column, multi-column, typed, dynamic dict
- **Vectorized by default, row-wise when needed** — whole `Series` in, or one Python scalar per row
- **Explicit fallback** — opt in to retry row by row when a function is not Series-compatible
- **Grouped execution** — `groupby` one or many columns, split-apply-combine
- **Schema validation** — `dtype`, `nullable`, `allowed_values`, `min`, `max` on input and output
- **Rollback** — a failing task leaves the frame exactly as it was
- **Odd cell types** — lists, dicts, sets, tuples, `numpy` arrays, class instances
- **Registration-time hazard detection** — warns before a row-wise function silently misbehaves

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [1. Registering steps — the five output modes](#1-registering-steps--the-five-output-modes)
- [2. How parameters map to columns](#2-how-parameters-map-to-columns)
- [3. Execution modes: vectorized vs row-wise](#3-execution-modes-vectorized-vs-row-wise)
- [4. Automatic fallback (opt-in)](#4-automatic-fallback-opt-in)
- [5. Grouped execution](#5-grouped-execution)
- [6. Schema validation](#6-schema-validation)
- [7. Rollback on failure](#7-rollback-on-failure)
- [8. Task management and single-task runs](#8-task-management-and-single-task-runs)
- [9. Index alignment](#9-index-alignment)
- [10. Unusual cell types](#10-unusual-cell-types)
- [11. Output shape guards](#11-output-shape-guards)
- [12. Vectorized-hazard detection](#12-vectorized-hazard-detection)
- [API reference](#api-reference)
- [Exception hierarchy](#exception-hierarchy)
- [Migrating from 1.x to 2.0](#migrating-from-1x-to-20)
- [Package structure](#package-structure)
- [Development](#development)

---

## Installation

```bash
pip install pipewise
```

Requires Python 3.9+ and `pandas>=1.5.0`. `tqdm` is an optional dependency: when
installed, `run()` shows a progress bar.

For local development:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

---

## Quick start

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

```
   a   b  sum  product
0  1  10   11       10
1  2  20   22       40
2  3  30   33       90
```

`run()` returns a new frame by default. Pass `inplace=True` to write into the
bound frame instead.

---

## 1. Registering steps — the five output modes

`outputs` controls what happens to the return value of a step.

| `outputs` | Meaning |
|---|---|
| `None` | side effect only — nothing is written back |
| `"col"` | a single output column |
| `["c1", "c2"]` | a fixed number of output columns |
| `{"col": type}` | typed output — cast with `astype` after writing |
| `"dict"` | dynamic output — the columns written depend on each row |

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
pw = Pipewise(df)


# 1) side effect only
@pw.register(outputs=None)
def log_total(a):
    print("total =", int(a.sum()))


# 2) single column
@pw.register(outputs="doubled")
def doubled(a):
    return a * 2


# 3) fixed multi-column
@pw.register(outputs=["sum", "product"])
def calc(a, b):
    return a + b, a * b


# 4) typed output — cast after assignment
@pw.register(outputs={"ratio": float, "is_big": bool})
def ratio(a, b):
    return a / b, a > 2


# 5) dynamic dict — a row may emit different columns
@pw.register(outputs="dict", vectorized=False)
def dynamic(a, b):
    row = {"small": a}
    if b >= 20:
        row["large"] = b
    return row


result = pw.run()
print(result.to_string(index=False))
```

```
total = 6
 a  b  doubled  sum  product  ratio  is_big  small  large
 1 10        2   11       10    0.1   False      1    NaN
 2 20        4   22       40    0.1   False      2   20.0
 3 30        6   33       90    0.1    True      3   30.0
```

Steps run in registration order, and each step can read the columns produced by
the previous ones. That is what makes the `dynamic` step above able to work on a
frame that already carries `doubled`, `sum`, and so on.

---

## 2. How parameters map to columns

A parameter **without a default** is looked up as a column of the same name.
`**kwargs` receives every remaining column, which is handy for passthrough
payloads.

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"score": [10, 20], "weight": [0.5, 0.8], "name": ["a", "b"]})
pw = Pipewise(df)


@pw.register(outputs="scaled")
def scaled(score, weight):
    return score * weight


@pw.register(outputs="summary", vectorized=False)
def summary(score, **extra):
    parts = [f"{key}={value}" for key, value in sorted(extra.items())]
    return f"{score} <- " + ", ".join(parts)


print(pw.run().to_string(index=False))
```

```
 score  weight name  scaled                               summary
    10     0.5    a     5.0  10 <- name=a, scaled=5.0, weight=0.5
    20     0.8    b    16.0 20 <- name=b, scaled=16.0, weight=0.8
```

### Careful: a defaulted parameter shadows a same-named column

A parameter that has a default is **not** read from the frame. If a column of
the same name exists, the column is ignored — and Pipewise warns about it at
registration time, because that is easy to miss.

```python
import warnings

import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
pw = Pipewise(df)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")

    @pw.register(outputs="c")
    def add(a, b=100):  # "b" has a default -> the "b" column is NOT used
        return a + b

print("warning:", caught[-1].message)
print("result :", pw.run()["c"].tolist())
```

```
warning: Function 'add' declares parameter(s) ['b'] with a default value, and the DataFrame also has column(s) with the same name. The column(s) will be ignored and the default used instead. Remove the default, or rename the parameter, to read the column.
result : [101, 102]
```

Remove the default (or rename the parameter) to actually read the column.

---

## 3. Execution modes: vectorized vs row-wise

`vectorized=True` (the default) passes entire `Series` objects, so the step runs
at pandas speed. `vectorized=False` calls the function once per row with plain
Python scalars, which is the natural fit for branching logic.

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [5, 15, 25]})
pw = Pipewise(df)


@pw.register(outputs="vec")  # whole Series in
def vec(a):
    return a * 2


@pw.register(outputs="row", vectorized=False)  # one scalar per call
def row(a):
    return "big" if a >= 10 else "small"


print(pw.run().to_string(index=False))
```

```
 a  vec   row
 5   10 small
15   30   big
25   50   big
```

Row-wise execution iterates with `DataFrame.itertuples`, which is roughly 5x
faster than `DataFrame.apply(axis=1)` and hands you Python-native values
(`int` / `float` / `str` rather than numpy scalars).

---

## 4. Automatic fallback (opt-in)

If a vectorized call fails with `TypeError`, `ValueError` or `AttributeError`,
Pipewise can retry the step row by row:

```python
import warnings

import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [5, 15, 25]})


# opt in: try vectorized, then retry row-wise
pw = Pipewise(df)


@pw.register(outputs="label", fallback_on_vectorized_error=True)
def label_fallback(a):
    if a >= 10:  # comparing a whole Series raises ValueError
        return "big"
    return "small"


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = pw.run()

print("result :", result["label"].tolist())
print("warned :", any("fell back" in str(w.message) for w in caught))
```

```
result : ['small', 'big', 'big']
warned : True
```

### Why the fallback is off by default

The vectorized call has **already executed the function body once** before the
error is caught. Retrying row by row therefore runs it a second time — so a
function with side effects (writing a file, appending to a list, calling an API)
will observe an extra, unexpected invocation.

The default is therefore explicit: no silent retry, the failure surfaces with a
hint. Prefer `vectorized=False` when a function is inherently row-wise.

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [5, 15, 25]})
pw = Pipewise(df)


@pw.register(outputs="label")  # default: fallback_on_vectorized_error=False
def label_strict(a):
    if a >= 10:
        return "big"
    return "small"


try:
    pw.run()
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    print(f"cause : {type(exc.__cause__).__name__}: {exc.__cause__}")
```

```
PipewiseExecutionError: Task 'label_strict' failed, all changes rolled back.
cause : ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```

A `RuntimeWarning` with the same message is also emitted, pointing at
`vectorized=False` / `fallback_on_vectorized_error=True`.

---

## 5. Grouped execution

`groupby="col"` or `groupby=["c1", "c2"]` runs the step once per group and
stitches the results back into the original frame.

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame(
    {
        "region": ["north", "north", "north", "south", "south"],
        "channel": ["web", "web", "app", "web", "app"],
        "amount": [10.0, 30.0, 25.0, 40.0, 55.0],
    }
)
pw = Pipewise(df)


@pw.register(outputs="vs_region_mean", groupby="region")
def vs_region_mean(amount):
    # vectorized: the whole group's column arrives at once, so group
    # aggregates such as .mean() are available
    return amount - amount.mean()


@pw.register(outputs="band", groupby=["region", "channel"], vectorized=False)
def band(amount):
    # row-wise: called once per row inside each (region, channel) group
    return "high" if amount >= 25 else "low"


print(pw.run().to_string(index=False))
```

```
region channel  amount  vs_region_mean band
 north     web    10.0      -11.666667  low
 north     web    30.0        8.333333 high
 north     app    25.0        3.333333 high
 south     web    40.0       -7.500000 high
 south     app    55.0        7.500000 high
```

---

## 6. Schema validation

Five rule keys are supported:

| Key | Meaning |
|---|---|
| `dtype` | pandas dtype string (`"integer"`, `"float"`, `"number"`, `"bool"`, `"string"`, `"datetime"`) or a Python type |
| `nullable` | `False` rejects null values |
| `allowed_values` | value must be in this collection |
| `min` / `max` | inclusive numeric bounds |

Rules can be declared in three places: the pipeline input, a step's input, and a
step's output.

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame(
    {"qty": [1, 2, 3], "status": ["new", "paid", "new"], "price": [5.0, 6.0, 7.0]}
)

pw = Pipewise(
    df,
    input_schema={
        "qty": {"dtype": "integer", "nullable": False, "min": 1},
        "status": {"allowed_values": ["new", "paid"]},
        "price": {"dtype": "number", "min": 0},
    },
)


@pw.register(
    outputs="total",
    output_schema={"total": {"dtype": "number", "min": 0, "max": 100}},
)
def total(qty, price):
    return qty * price


print("valid input ->", pw.run()["total"].tolist())
```

```
valid input -> [5.0, 12.0, 21.0]
```

A violating frame is rejected before any step runs:

```python
import pandas as pd

from pipewise import Pipewise

bad = Pipewise(
    pd.DataFrame({"qty": [1, None, 3]}),
    input_schema={"qty": {"nullable": False}},
)

try:
    bad.run()
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
```

```
PipewiseInputSchemaError: pipeline input column 'qty' contains null values, but nullable=False.
```

And a step whose output breaks its own declared schema fails after that step:

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"qty": [1, 2, 3], "price": [5.0, 6.0, 7.0]})
pw = Pipewise(df)


@pw.register(outputs="total", output_schema={"total": {"max": 10}})
def total(qty, price):
    return qty * price


try:
    pw.run()
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    print(f"cause : {type(exc.__cause__).__name__}: {exc.__cause__}")
```

```
PipewiseExecutionError: Task 'total' failed, all changes rolled back.
cause : PipewiseOutputSchemaError: task output for 'total' column 'total' contains value 12.0 above max=10.
```

---

## 7. Rollback on failure

`run()` snapshots the frame first. If any step raises, every change made during
that run is undone before the error is re-raised as `PipewiseExecutionError`,
with the original exception attached as `__cause__`.

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2, 3]})
pw = Pipewise(df.copy())


@pw.register(outputs="ok")
def ok(a):
    return a * 2


@pw.register(outputs="boom")
def boom(a):
    raise RuntimeError("downstream failure")


columns_before = list(pw.data.columns)

try:
    pw.run(inplace=True)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    print(f"cause : {type(exc.__cause__).__name__}: {exc.__cause__}")

print("columns before:", columns_before)
print("columns after :", list(pw.data.columns))
```

```
PipewiseExecutionError: Task 'boom' failed, all changes rolled back.
cause : RuntimeError: downstream failure
columns before: ['a']
columns after : ['a']
```

The `ok` step had already added its column — the rollback removed it.

---

## 8. Task management and single-task runs

```python
import logging
import sys

import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2, 3]})
pw = Pipewise(df)


@pw.register(outputs="b")
def step_b(a):
    return a * 2


@pw.register(outputs="c")
def step_c(b):
    return b + 1


print("tasks:")
for entry in pw.tasks:
    print("  ", entry)

logger = logging.getLogger("pipewise.core")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

print("\nplan:")
pw.plan()

logger.removeHandler(handler)

# run a single step for faster debugging
single = pw.run(task="step_b")
print("\nsingle-task columns:", list(single.columns))

print("removed step_c:", pw.remove(step_c))
pw.clear()
print("tasks after clear:", pw.tasks)
```

```
tasks:
   TaskSummary(func_name='step_b', outputs=['b'], groupby=None, vectorized=True)
   TaskSummary(func_name='step_c', outputs=['c'], groupby=None, vectorized=True)

plan:
Execution Plan:
  #  Function               Outputs                      GroupBy      Vec
---------------------------------------------------------------------------
  1. step_b                 ['b']                        -            Y
  2. step_c                 ['c']                        -            Y

single-task columns: ['a', 'b']
removed step_c: True
tasks after clear: []
```

`run(task=...)` executes only that step, but the step's own inputs still have to
exist in the frame — a step that consumes a previous step's output cannot be run
in isolation.

---

## 9. Index alignment

A vectorized step may return a `Series`. It is realigned to the frame's index by
label, so a returned series does not have to be in frame order.

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
pw = Pipewise(df)


@pw.register(outputs="reordered")
def reordered(a):
    # deliberately returned out of order
    return pd.Series([300, 100, 200], index=[30, 10, 20])


result = pw.run()
print(result.to_string())
```

```
    a  reordered
10  1        100
20  2        200
30  3        300
```

Length-mismatched results are rejected rather than broadcast — see
[Output shape guards](#11-output-shape-guards).

---

## 10. Unusual cell types

A column does not have to hold scalars. Lists, dicts, sets, tuples,
`numpy.ndarray`, class objects and class instances are all supported in both
execution modes.

```python
import numpy as np
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame(
    {
        "tags": [["urgent", "gift"], ["bulk"]],             # list
        "aspects": [{"w": 2, "h": 3}, {"w": 4, "h": 5}],    # dict
        "labels": [{"a", "b"}, {"c"}],                      # set
        "vec": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])],  # ndarray
    }
)
pw = Pipewise(df)


@pw.register(outputs="tag_count")
def tag_count(tags):
    # vectorized: the .str accessor works on list cells too
    return tags.str.len()


@pw.register(outputs=["w", "h"], vectorized=False)
def unpack(aspects):
    # row-wise: each cell arrives as a real Python object
    return aspects["w"], aspects["h"]


@pw.register(outputs="label_count", vectorized=False)
def label_count(labels):
    return len(labels)


@pw.register(outputs="vec_sum", vectorized=False)
def vec_sum(vec):
    return float(vec.sum())


print(pw.run()[["tag_count", "w", "h", "label_count", "vec_sum"]].to_string(index=False))
```

```
 tag_count  w  h  label_count  vec_sum
         2  2  3            2      6.0
         1  4  5            1      9.0
```

A vectorized step may also return a 2-D `numpy.ndarray` to fill several output
columns at once, and an `(n, 1)` array to fill a single one:

```python
import numpy as np
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
pw = Pipewise(df)


@pw.register(outputs=["sum", "product"])
def both(a, b):
    return np.column_stack([a + b, a * b])


print(pw.run().to_string(index=False))
```

```
 a  b  sum  product
 1  4    5        4
 2  5    7       10
 3  6    9       18
```

Class instances work the same way:

```python
import pandas as pd

from pipewise import Pipewise


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


points = pd.DataFrame({"pt": [Point(1, 2), Point(3, 4)]})
pw = Pipewise(points)


@pw.register(outputs=["x", "y"], vectorized=False)
def unpack_point(pt):
    return pt.x, pt.y


@pw.register(outputs="norm", vectorized=False)
def norm(pt):
    return (pt.x**2 + pt.y**2) ** 0.5


print(pw.run().to_string(index=False))
```

```
         pt  x  y     norm
Point(1, 2)  1  2 2.236068
Point(3, 4)  3  4 5.000000
```

---

## 11. Output shape guards

When a return value's shape does not match the frame, Pipewise raises
`PipewiseOutputAssignmentError` naming the function and column, instead of
letting pandas raise something opaque — or silently broadcasting a single value
across every row.

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2, 3]})
pw = Pipewise(df)


@pw.register(outputs="bad")
def bad(a):
    return [1]  # one value for three rows


try:
    pw.run()
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    print(f"cause : {type(exc.__cause__).__name__}: {exc.__cause__}")
```

```
PipewiseExecutionError: Task 'bad' failed, all changes rolled back.
cause : PipewiseOutputAssignmentError: Function 'bad' produced 1 value(s) for output column 'bad', but the DataFrame has 3 row(s). Return one value per row, or a scalar to broadcast.
```

Ragged row-wise results point at the offending row:

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [3, 12]})
pw = Pipewise(df)


@pw.register(outputs=["p", "q"], vectorized=False)
def ragged(a):
    if a >= 10:
        return a, a * 2
    return (a,)  # only one value for a two-column output


try:
    pw.run()
except Exception as exc:
    print(f"cause : {type(exc.__cause__).__name__}: {exc.__cause__}")
```

```
cause : PipewiseOutputAssignmentError: Function 'ragged' returned 1 value(s) for row 0, but outputs specifies 2 columns: ['p', 'q'].
```

A genuine scalar is still allowed, and broadcasts:

```python
import pandas as pd

from pipewise import Pipewise

df = pd.DataFrame({"a": [1, 2, 3]})
pw = Pipewise(df)


@pw.register(outputs="constant")
def constant(a):
    return 0


print(pw.run()["constant"].tolist())
```

```
[0, 0, 0]
```

---

## 12. Vectorized-hazard detection

At registration time, Pipewise parses the step's source and logs a warning for
idioms that do not behave as intended on a `Series`. Detection is limited to the
function's actual column parameters, so helpers and local variables never
produce a false positive.

| Detected | Why it is a hazard |
|---|---|
| `len(col)` | returns the row count, not the per-element length |
| `isinstance(col, ...)` | always `False` for a Series |
| `type(col)` | returns `Series`, not the element type |
| `col.split(...)` / `.strip(...)` / `.lower()` / `.upper()` | `Series` has no such method — raises `AttributeError` |
| `col[0]` / `col['key']` | label-based indexing, not per-element access |
| `if col > 0:` / `if col:` / `x if col else y` | evaluating a Series for truth raises `ValueError` |

Use the `.str` accessor, or switch the step to `vectorized=False`:

```python
import logging

import pandas as pd

from pipewise import Pipewise

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

df = pd.DataFrame({"name": ["alice", "bob"]})
pw = Pipewise(df)


# good: vectorized string handling
@pw.register(outputs="upper")
def upper_name(name):
    return name.str.upper()


# good: row-wise, so Python-string branches are legal
@pw.register(outputs="initial", vectorized=False)
def initial(name):
    return name[0].upper()


print(pw.run().to_string(index=False))
```

```
 name upper initial
alice ALICE       A
  bob   BOB       B
```

Writing `return name.upper()` in the first step instead would log:

```
WARNING Function 'upper_name' may be vectorized-incompatible: `name.upper()` raises
AttributeError on a Series — use `.str.upper()` (vectorized) or `vectorized=False` (row-wise).
```

---

## API reference

### `Pipewise(data, input_schema=None)`

Binds a `pandas.DataFrame` and optional pipeline-level input schema.

### `register(func=None, *, outputs=None, groupby=None, vectorized=True, input_schema=None, output_schema=None, fallback_on_vectorized_error=False)`

Decorator that registers a step. Returns the function unchanged, so a step can
also be called directly.

| Parameter | Default | Meaning |
|---|---|---|
| `outputs` | `None` | `None` / `"col"` / `["c1","c2"]` / `{"col": type}` / `"dict"` |
| `groupby` | `None` | column name or list of names to group by |
| `vectorized` | `True` | pass whole `Series`, or one scalar per row |
| `input_schema` | `None` | schema rules checked before the step runs |
| `output_schema` | `None` | schema rules checked after the step runs |
| `fallback_on_vectorized_error` | `False` | retry row-wise when the vectorized call fails |

Raises `PipewiseRegistrationError` for malformed `outputs`, `groupby`, a schema
with unknown keys, or an `output_schema` that references an undeclared column.

### `run(inplace=False, task=None)`

Executes every registered step in order and returns the resulting frame.
`inplace=True` writes into the bound frame. `task="name"` runs only that step and
raises `PipewiseTaskSelectionError` if the name is missing or ambiguous.

### `tasks`

A list of `TaskSummary(func_name, outputs, groupby, vectorized)` named tuples.
Comparable to plain tuples.

### `remove(func)` / `clear()` / `plan()`

`remove` deletes a step by function reference and returns whether it was found.
`clear` drops all steps. `plan` logs the execution plan at `INFO`.

---

## Exception hierarchy

```
PipewiseError
├── PipewiseRegistrationError        invalid metadata passed to register()
├── PipewiseTaskSelectionError       run(task=...) could not resolve one step
├── PipewiseOutputAssignmentError    return value cannot be written back
├── PipewiseTypeConversionError      declared output type coercion failed
├── PipewiseExecutionError           a step failed and changes were rolled back
└── PipewiseSchemaError
    ├── PipewiseInputColumnError     a required input column is missing
    ├── PipewiseInputSchemaError     input data violates declared rules
    ├── PipewiseOutputSchemaError    step output violates declared rules
    └── PipewiseGroupByError         groupby columns are not usable
```

Pipeline input schema violations are raised directly. Anything that fails
*inside* a step is raised as `PipewiseExecutionError` with the original exception
as `__cause__`.

---

## Migrating from 1.x to 2.0

2.0 makes previously implicit behaviour explicit.

| Change | What to do |
|---|---|
| `fallback_on_vectorized_error` now defaults to `False` | Pass `fallback_on_vectorized_error=True` to keep the old automatic retry, or switch the step to `vectorized=False`. The retry re-executes the function body, which can repeat side effects — that is why it is no longer implicit. |
| Row-wise values are Python scalars | Values that were numpy scalars (`np.int64`) are now `int` / `float`. `isinstance(x, int)` now behaves as expected. |
| `tasks` returns `TaskSummary` named tuples | Still plain-tuple comparable, so `tasks == [(name, outputs, groupby, vectorized)]` keeps working. |
| Grouped `groupby` keeps its original form | `groupby="g"` stays `"g"` in `tasks`; `groupby=["g"]` stays a list. |
| Class-level schema helpers removed | `_validate_frame_schema`, `_matches_dtype`, `_validate_dtype`, `_apply_types` and `_rollback` are gone from `Pipewise`. Import the `_schema` helpers directly if you depended on them. |

Everything else is additive: 2-D array returns, hazard detection for branch
conditions, the shadowed-default warning, `py.typed`, and the clearer
shape-guard errors.

---

## Package structure

```text
pipewise/
  __init__.py     public interface, version and author
  core.py         Pipewise class: registration, orchestration, rollback
  _execution.py   vectorized / row-wise / grouped execution, fallback policy
  _assign.py      output assignment, index alignment, shape guards
  _tasks.py       TaskDef metadata and signature parsing
  _schema.py      schema validation and dtype matching
  _hazards.py     AST-based vectorized-hazard detection
  errors.py       exception hierarchy
tests/
  test_pipewise.py            129 pytest cases, 85% coverage
pipewise_feature_tests.ipynb  end-to-end walkthrough, 74 assertions
```

---

## Development

```bash
python -m pytest -q
```

With coverage and lint, as CI runs them:

```bash
python -m pytest --cov=pipewise --cov-report=term-missing --cov-fail-under=80
ruff check pipewise tests
```

CI runs the suite on Python 3.9–3.13 plus a lint job. Tagging a `v*` release
triggers the publish workflow, which re-runs the tests before building and
uploading to PyPI.

`pipewise_feature_tests.ipynb` is an executable walkthrough of every feature. It
can be re-run end to end:

```bash
jupyter nbconvert --to notebook --execute --inplace ./pipewise_feature_tests.ipynb
```

## Public metadata

```python
from pipewise import __author__, __version__
```

- `__version__ = "2.0.0"`
- `__author__ = "XiaoZhouZhou"`

## License

MIT — see [LICENSE](LICENSE).
