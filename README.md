# Pipewise

`Pipewise` is a lightweight `pandas.DataFrame` pipeline library for teams that
want reusable data-processing steps without adopting a heavyweight workflow
framework.

It helps you:

- register step functions with decorators
- map function arguments to DataFrame columns automatically
- write results back to one or more columns
- validate input and output schema rules
- fall back from vectorized mode to row-wise mode when needed
- roll back all changes if any task fails

## Installation

After publishing to PyPI:

```bash
pip install pipewise
```

For local development:

```bash
pip install -r requirements.txt
pip install -e .
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

- Sequential pipeline registration with `@pipewise.register(...)`
- Multiple output modes:
  - no output
  - single-column output
  - multi-column output
  - typed output mapping
  - dynamic dict output
- Grouped execution with `groupby`
- Schema checks:
  - `dtype`
  - `nullable`
  - `allowed_values`
  - `min`
  - `max`
- Automatic rollback when execution fails
- Custom exception hierarchy for easier debugging

## Package Structure

```text
pipewise/
  __init__.py
  core.py
  errors.py
tests/
  test_pipewise.py
README.md
requirements.txt
pyproject.toml
```

## Public Metadata

```python
from pipewise import __author__, __version__
```

- `__version__ = "1.0.1"`
- `__author__ = "XiaoZhouZhou"`

## Testing

```bash
python -m unittest -v tests/test_pipewise.py
```

## Publish Checklist

1. Update `pipewise/__init__.py` version.
2. Update `pyproject.toml` version if needed.
3. Run tests.
4. Build the package:

```bash
python -m build
```

5. Upload to PyPI with your preferred workflow, for example `twine`.
