"""
Pipewise — comprehensive usage examples.

Run:  conda run -n risk python example.py
"""

# pylint: disable=missing-function-docstring
import warnings
from typing import Any, Dict

import pandas as pd

from pipewise import Pipewise, SchemaRule


# ======================================================================
# 1. Basic vectorized multi-output
# ======================================================================

def basic_vectorized_multi_output() -> pd.DataFrame:
    """Vectorized execution with multiple output columns."""
    df = pd.DataFrame({"a": [5, 15, 25], "b": [1, 2, 3]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs=["sum", "product"])
    def calc(a, b):
        return a + b, a * b

    result = pipewise.run()
    return result


# ======================================================================
# 2. Auto fallback — branching function forces row-wise
# ======================================================================

def auto_fallback_rowwise() -> pd.DataFrame:
    """Function with branching logic auto-falls back to row-wise."""
    df = pd.DataFrame({"a": [5, 15, 25], "b": [1, 2, 3]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs=["level", "doubled"])
    def classify_and_double(a, b):
        if a < 10:               # ← this branch makes vectorized fail
            return "low", a * 2
        if a < 20:
            return "medium", b * 10
        return "high", a + b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pipewise.run()
    return result


# ======================================================================
# 3. Dynamic dict output
# ======================================================================

def dynamic_dict_output() -> pd.DataFrame:
    """Dict mode: each row can produce different columns."""
    df = pd.DataFrame({"a": [5, 15, 25], "b": [1, 2, 3]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs="dict")
    def update(a, b):
        if a < 10:
            return {"level": "low", "suggest": a * 2}
        if a < 20:
            return {"level": "medium", "adjust": b + 100}
        return {"level": "high", "remark": "too large"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pipewise.run()
    return result


# ======================================================================
# 4. GroupBy execution
# ======================================================================

def groupby_execution() -> pd.DataFrame:
    """Grouped execution with split-apply-combine."""
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B", "C"],
        "value": [10, 20, 5, 15, 30],
    })
    pipewise = Pipewise(df)

    @pipewise.register(outputs="classified", groupby="group")
    def classify_group(value):
        if value > 12:
            return "high"
        if value > 8:
            return "mid"
        return "low"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pipewise.run()
    return result


# ======================================================================
# 5. Typed outputs (type conversion)
# ======================================================================

def typed_outputs() -> pd.DataFrame:
    """Output columns with declared dtype — auto conversion."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs={"quotient": float, "is_large": bool})
    def compute(a, b):
        return a / b, a > 2

    result = pipewise.run()
    return result


# ======================================================================
# 6. Input schema validation
# ======================================================================

def input_schema_validation() -> pd.DataFrame:
    """Pipeline-level input schema checks."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [10, 20, 30],
    })
    pipewise = Pipewise(
        df,
        input_schema={
            "a": {"dtype": "integer", "nullable": False, "min": 1},
            "b": {"dtype": "integer", "max": 30},
        },
    )

    @pipewise.register(outputs="c")
    def add(a, b):
        return a + b

    return pipewise.run()


# ======================================================================
# 7. Output schema validation
# ======================================================================
def output_schema_validation() -> pd.DataFrame:
    """Task-level output schema with string dtype."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs="score", output_schema={"score": {"dtype": "integer"}})
    def score(a):
        return a * 10

    return pipewise.run()


# ======================================================================
# 8. Inplace vs copy behavior
# ======================================================================

def inplace_vs_copy() -> Dict[str, pd.DataFrame]:
    """inplace=False (default) preserves original; inplace=True mutates."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs="doubled")
    def double(a):
        return a * 2

    result = pipewise.run()            # inplace=False (default) — returns copy
    original_unmodified = df.copy()    # snapshot before inplace=True

    pipewise.run(inplace=True)         # now mutates pipewise.data

    return {"result": result, "original_before_inplace": original_unmodified, "pipewise_data": pipewise.data}


# ======================================================================
# 9. Task management
# ======================================================================

def task_management() -> None:
    """List, remove, clear tasks and view execution plan."""
    pipewise = Pipewise(pd.DataFrame({"a": [1, 2, 3]}))

    @pipewise.register(outputs="b")
    def step1(a):
        return a * 2

    @pipewise.register(outputs="c")
    def step2(b):
        return b + 10

    print(f"Registered tasks: {pipewise.tasks}")
    pipewise.plan()

    pipewise.remove(step1)
    print(f"After removing step1: {pipewise.tasks}")

    pipewise.clear()
    print(f"After clear: {pipewise.tasks}")


# ======================================================================
# 10. Single task execution (debug mode)
# ======================================================================

def single_task_run() -> pd.DataFrame:
    """Run only one registered task by name — for fast debugging."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs="b")
    def step1(a):
        return a * 2

    @pipewise.register(outputs="c")
    def step2(a):
        return a + 100

    return pipewise.run(task="step2")


# ======================================================================
# 11. Rollback on failure
# ======================================================================

def rollback_on_failure() -> pd.DataFrame:
    """When a task fails, all changes from this run are rolled back."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs="b")
    def safe_step(a):
        return a * 2

    @pipewise.register(outputs="c")
    def bad_step(a):
        raise ValueError("something went wrong")

    try:
        pipewise.run(inplace=True)
    except Exception as exc:
        print(f"Caught: {exc}")

    # The data is untouched despite safe_step having run
    return pipewise.data


# ======================================================================
# 12. Complex input types — list
# ======================================================================

def list_column_operations() -> pd.DataFrame:
    """DataFrame columns containing lists work in both modes."""
    df = pd.DataFrame({"items": [[1, 2], [3, 4, 5], [6]]})
    pipewise = Pipewise(df)

    # Vectorized — uses .str accessor
    @pipewise.register(outputs="count")
    def count_items(items):
        return items.str.len()

    # Row-wise — receives individual lists
    @pipewise.register(outputs="summary", vectorized=False)
    def summarize(items):
        return f"len={len(items)}, sum={sum(items)}"

    return pipewise.run()


# ======================================================================
# 13. Complex input types — dict
# ======================================================================

def dict_column_operations() -> pd.DataFrame:
    """DataFrame columns containing dicts work row-wise."""
    df = pd.DataFrame({
        "attrs": [{"w": 1, "h": 2}, {"w": 3, "h": 4}, {"w": 5, "h": 6}],
    })
    pipewise = Pipewise(df)

    # Row-wise — safe for dict indexing
    @pipewise.register(outputs="area", vectorized=False)
    def compute_area(attrs):
        return attrs["w"] * attrs["h"]

    # Vectorized — uses .str accessor to extract keys
    @pipewise.register(outputs=["w", "h"])
    def extract(attrs):
        return attrs.str["w"], attrs.str["h"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pipewise.run()
    return result


# ======================================================================
# 14. Complex input types — set
# ======================================================================

def set_column_operations() -> pd.DataFrame:
    """DataFrame columns containing sets work in both modes."""
    df = pd.DataFrame({
        "tags": [{"x", "y"}, {"a", "b", "c"}, {"z"}],
    })
    pipewise = Pipewise(df)

    # Vectorized
    @pipewise.register(outputs="tag_count")
    def count_tags(tags):
        return tags.apply(len)

    # Row-wise
    @pipewise.register(outputs="sorted_tags", vectorized=False)
    def sort_tags(tags):
        return ",".join(sorted(tags))

    return pipewise.run()


# ======================================================================
# 15. GroupBy with list column
# ======================================================================

def groupby_with_list_column() -> pd.DataFrame:
    """GroupBy on a standard column, processing a list column within groups."""
    df = pd.DataFrame({
        "group": ["A", "A", "B"],
        "values": [[1, 2], [3], [4, 5, 6]],
    })
    pipewise = Pipewise(df)

    @pipewise.register(outputs="total", groupby="group")
    def sum_values(values):
        return sum(values)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pipewise.run()
    return result


# ======================================================================
# 16. Dict output with dict input columns
# ======================================================================

def dict_output_with_dict_input() -> pd.DataFrame:
    """Dict output mode processing dict-typed columns row-wise."""
    df = pd.DataFrame({
        "info": [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
    })
    pipewise = Pipewise(df)

    @pipewise.register(outputs="dict", vectorized=False)
    def enrich(info):
        return {"sum": info["a"] + info["b"], "product": info["a"] * info["b"]}

    result = pipewise.run()
    return result


# ======================================================================
# 17. Mixed vectorized and row-wise pipeline
# ======================================================================

def mixed_pipeline() -> pd.DataFrame:
    """Vectorized and row-wise tasks can be chained freely."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs="b")
    def v_step(a):
        return a * 10

    @pipewise.register(outputs="c")
    def r_step(b):
        if b > 15:
            return "big"
        return "small"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pipewise.run()
    return result


# ======================================================================
# 18. **kwargs row-wise mode
# ======================================================================

def kwargs_rowwise() -> pd.DataFrame:
    """Row-wise mode with **kwargs captures extra columns."""
    df = pd.DataFrame({
        "x": [1, 2],
        "y": [10, 20],
        "z": [100, 200],
    })
    pipewise = Pipewise(df)

    @pipewise.register(outputs="summary", vectorized=False)
    def summarize(x, **extra):
        extras = ", ".join(f"{k}={v}" for k, v in extra.items())
        return f"x={x}, {extras}"

    return pipewise.run()


# ======================================================================
# 19. Side-effect only tasks
# ======================================================================

def side_effect_only() -> int:
    """outputs=None means the task only runs for side effects."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    side_log: list = []
    pipewise = Pipewise(df)

    @pipewise.register(outputs=None)
    def log_input(a):
        side_log.append(sum(a))

    pipewise.run()
    return len(side_log)  # vectorized: sum(Series) yields 1 scalar call


# ======================================================================
# 20. Edge cases — empty list and None
# ======================================================================

def edge_cases() -> pd.DataFrame:
    """Empty lists and None values in complex columns."""
    df = pd.DataFrame({"data": [[1], None, [2, 3], []]})
    pipewise = Pipewise(df)

    @pipewise.register(outputs="length", vectorized=False)
    def get_length(data):
        if data is None:
            return -1
        return len(data)

    return pipewise.run()


# ======================================================================
# Run all examples and assert correctness
# ======================================================================

def main():
    passed = 0
    failed = 0

    def check(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name}")

    # 1
    df = basic_vectorized_multi_output()
    check("basic_vectorized_multi_output",
          df["sum"].tolist() == [6, 17, 28] and df["product"].tolist() == [5, 30, 75])

    # 2
    df = auto_fallback_rowwise()
    check("auto_fallback_rowwise",
          df["level"].tolist() == ["low", "medium", "high"] and df["doubled"].tolist() == [10, 20, 28])

    # 3
    df = dynamic_dict_output()
    check("dynamic_dict_output",
          df["level"].tolist() == ["low", "medium", "high"] and df["suggest"].dropna().tolist() == [10.0])

    # 4
    df = groupby_execution()
    check("groupby_execution",
          df["classified"].tolist() == ["mid", "high", "low", "high", "high"])

    # 5
    df = typed_outputs()
    check("typed_outputs",
          str(df["quotient"].dtype) == "float64" and str(df["is_large"].dtype) == "bool")

    # 6
    df = input_schema_validation()
    check("input_schema_validation", df["c"].tolist() == [11, 22, 33])

    # 7
    df = output_schema_validation()
    check("output_schema_validation", df["score"].tolist() == [10, 20, 30])

    # 8
    result = inplace_vs_copy()
    check("inplace_vs_copy (copy preserves original)",
          "doubled" not in result["original_before_inplace"].columns)
    check("inplace_vs_copy (inplace mutates)",
          "doubled" in result["pipewise_data"].columns)
    check("inplace_vs_copy (result has doubled)",
          "doubled" in result["result"].columns)

    # 9
    task_management()  # just runs without error

    # 10
    df = single_task_run()
    check("single_task_run",
          list(df.columns) == ["a", "c"] and df["c"].tolist() == [101, 102, 103])

    # 11
    df = rollback_on_failure()
    check("rollback_on_failure",
          list(df.columns) == ["a"] and df["a"].tolist() == [1, 2, 3])

    # 12
    df = list_column_operations()
    check("list_column_operations",
          df["count"].tolist() == [2, 3, 1] and df["summary"].tolist() == [
              "len=2, sum=3", "len=3, sum=12", "len=1, sum=6",
          ])

    # 13
    df = dict_column_operations()
    check("dict_column_operations",
          df["area"].tolist() == [2, 12, 30])

    # 14
    df = set_column_operations()
    check("set_column_operations",
          df["tag_count"].tolist() == [2, 3, 1] and all(
              isinstance(v, str) for v in df["sorted_tags"]
          ))

    # 15
    df = groupby_with_list_column()
    check("groupby_with_list_column", df["total"].tolist() == [3, 3, 15])

    # 16
    df = dict_output_with_dict_input()
    check("dict_output_with_dict_input",
          df["sum"].tolist() == [3, 7, 11] and df["product"].tolist() == [2, 12, 30])

    # 17
    df = mixed_pipeline()
    check("mixed_pipeline",
          df["b"].tolist() == [10, 20, 30] and df["c"].tolist() == ["small", "big", "big"])

    # 18
    df = kwargs_rowwise()
    check("kwargs_rowwise",
          df["summary"].tolist() == [
              "x=1, y=10, z=100",
              "x=2, y=20, z=200",
          ])

    # 19
    result_len = side_effect_only()
    # side_effect returns nothing so len(side_log) is 0
    # vectorized: sum(Series) once → log has 1 entry
    check("side_effect_only", result_len == 1)

    # 20
    df = edge_cases()
    check("edge_cases", df["length"].tolist() == [1, -1, 2, 0])

    print(f"\n{'='*40}")
    print(f"Total: {passed + failed}  |  Passed: {passed}  |  Failed: {failed}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
