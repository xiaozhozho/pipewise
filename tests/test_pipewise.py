"""Tests for Pipewise — a lightweight pandas DataFrame pipeline."""

from __future__ import annotations

import io
import json
import logging
import math
import statistics
import warnings

import numpy as np
import pandas as pd
import pytest

from pipewise import Pipewise, SchemaRule
from pipewise.errors import (
    PipewiseError,
    PipewiseExecutionError,
    PipewiseInputColumnError,
    PipewiseInputSchemaError,
    PipewiseOutputSchemaError,
    PipewiseRegistrationError,
    PipewiseTaskSelectionError,
)


# ======================================================================
# Basic functionality
# ======================================================================


class TestRegistrationAndExecution:
    def test_vectorized_multi_output(self):
        df = pd.DataFrame({"a": [5, 15, 25], "b": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs=["sum", "product"])
        def calc(a, b):
            return a + b, a * b

        result = pipewise.run()

        assert result["sum"].tolist() == [6, 17, 28]
        assert result["product"].tolist() == [5, 30, 75]

    def test_auto_fallback_for_branching_function(self):
        df = pd.DataFrame({"a": [5, 15, 25], "b": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs=["level", "doubled"])
        def classify_and_double(a, b):
            if a < 10:
                return "low", a * 2
            if a < 20:
                return "medium", b * 10
            return "high", a + b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        assert result["level"].tolist() == ["low", "medium", "high"]
        assert result["doubled"].tolist() == [10, 20, 28]

    def test_dynamic_dict_output(self):
        df = pd.DataFrame({"a": [5, 15, 25], "b": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="dict")
        def smart_update(a, b):
            if a < 10:
                return {"level": "low", "suggest": a * 2}
            if a < 20:
                return {"level": "medium", "adjust": b + 100}
            return {"level": "high", "remark": "too large"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        assert result["level"].tolist() == ["low", "medium", "high"]
        assert result["suggest"].dropna().tolist() == [10.0]
        assert result["adjust"].dropna().tolist() == [102.0]
        assert result["remark"].dropna().tolist() == ["too large"]

    def test_groupby_execution(self):
        df = pd.DataFrame(
            {"group": ["A", "A", "B", "B", "C"], "value": [10, 20, 5, 15, 30]}
        )
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

        assert result["classified"].tolist() == [
            "mid", "high", "low", "high", "high",
        ]

    def test_typed_outputs(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs={"quotient": float, "is_large": bool})
        def compute(a, b):
            return a / b, a > 2

        result = pipewise.run()

        assert str(result["quotient"].dtype) == "float64"
        assert str(result["is_large"].dtype) == "bool"
        assert result["is_large"].tolist() == [False, False, True]


class TestSchemaValidation:
    def test_pipeline_input_schema_validation(self):
        valid_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        valid_pipewise = Pipewise(
            valid_df,
            input_schema={
                "a": {"dtype": "integer", "nullable": False, "min": 1},
                "b": {"dtype": "integer", "max": 30},
            },
        )
        validated = valid_pipewise.run()
        assert validated.equals(valid_df)

        invalid_df = pd.DataFrame({"a": [1, None, 3]})
        invalid_pipewise = Pipewise(
            invalid_df,
            input_schema={"a": {"dtype": "number", "nullable": False}},
        )

        with pytest.raises(PipewiseInputSchemaError):
            invalid_pipewise.run()

    def test_task_output_schema_validation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(
            outputs="score", output_schema={"score": {"dtype": "integer", "max": 20}}
        )
        def score(a):
            return a * 10

        with pytest.raises(PipewiseExecutionError) as exc_info:
            pipewise.run()

        assert isinstance(exc_info.value.__cause__, PipewiseOutputSchemaError)
        assert list(df.columns) == ["a"]

    def test_invalid_output_schema_target_raises_registration_error(self):
        pipewise = Pipewise(pd.DataFrame({"a": [1, 2, 3]}))

        with pytest.raises(PipewiseRegistrationError):

            @pipewise.register(outputs="b", output_schema={"c": {"dtype": "integer"}})
            def invalid_schema(a):
                return a * 2


class TestInplace:
    def test_default_not_inplace(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="doubled")
        def double(a):
            return a * 2

        result = pipewise.run()
        assert "doubled" in result.columns
        assert "doubled" not in df.columns
        assert "doubled" not in pipewise.data.columns

    def test_inplace_flag(self):
        inplace_df = pd.DataFrame({"a": [1, 2, 3]})
        inplace_pipewise = Pipewise(inplace_df)

        @inplace_pipewise.register(outputs="doubled")
        def double_inplace(a):
            return a * 2

        inplace_pipewise.run(inplace=True)
        assert "doubled" in inplace_pipewise.data.columns


class TestTaskManagement:
    def test_task_list_and_remove_and_clear(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def step1(a):
            return a * 2

        @pipewise.register(outputs="c")
        def step2(b):
            return b + 10

        assert pipewise.tasks == [
            ("step1", ["b"], None, True),
            ("step2", ["c"], None, True),
        ]
        assert pipewise.remove(step1)
        assert pipewise.tasks == [("step2", ["c"], None, True)]
        pipewise.clear()
        assert pipewise.tasks == []

    def test_run_specific_task_only(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def step1(a):
            return a * 2

        @pipewise.register(outputs="c")
        def step2(a):
            return a + 100

        result = pipewise.run(task="step2")

        assert list(result.columns) == ["a", "c"]
        assert result["c"].tolist() == [101, 102, 103]

    def test_run_specific_task_missing_name(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def step1(a):
            return a * 2

        with pytest.raises(PipewiseTaskSelectionError):
            pipewise.run(task="step_missing")

    def test_run_specific_task_requires_unique_name(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def duplicate(a):
            return a * 2

        @pipewise.register(outputs="c")
        def duplicate(a):
            return a + 10

        with pytest.raises(PipewiseTaskSelectionError):
            pipewise.run(task="duplicate")

    def test_plan_output(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b", vectorized=False)
        def step1(a):
            return a * 2

        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        logger = logging.getLogger("pipewise.core")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            pipewise.plan()
        finally:
            logger.removeHandler(handler)

        output = buffer.getvalue()
        assert "Execution Plan:" in output
        assert "step1" in output
        assert "['b']" in output


class TestRollback:
    def test_rollback_on_failure(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def safe_step(a):
            return a * 2

        @pipewise.register(outputs="c")
        def bad_step(a):
            raise ValueError("fail")

        with pytest.raises(PipewiseExecutionError):
            pipewise.run(inplace=True)

        assert list(df.columns) == ["a"]
        assert list(pipewise.data.columns) == ["a"]

    def test_missing_input_column_uses_custom_exception(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def needs_missing_column(missing_col):
            return missing_col * 2

        with pytest.raises(PipewiseExecutionError) as exc_info:
            pipewise.run()

        assert isinstance(exc_info.value.__cause__, PipewiseInputColumnError)

    def test_non_fallback_vectorized_error_bubbles_up(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def broken(a):
            raise KeyError("boom")

        with pytest.raises(PipewiseExecutionError) as exc_info:
            pipewise.run()

        assert isinstance(exc_info.value.__cause__, KeyError)


class TestKwargsAndMixed:
    def test_kwargs_rowwise_mode(self):
        df = pd.DataFrame({"x": [1, 2], "y": [10, 20], "z": [100, 200]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="summary", vectorized=False)
        def summarize(x, **extra):
            extras = ", ".join(f"{k}={v}" for k, v in extra.items())
            return f"x={x}, {extras}"

        result = pipewise.run()

        assert result["summary"].tolist() == [
            "x=1, y=10, z=100",
            "x=2, y=20, z=200",
        ]

    def test_mixed_vectorized_and_rowwise_pipeline(self):
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

        assert result["b"].tolist() == [10, 20, 30]
        assert result["c"].tolist() == ["small", "big", "big"]


# ======================================================================
# Complex input types — list, dict, set
# ======================================================================


class TestComplexInputTypes:
    """Verify that Pipewise handles DataFrame columns containing list, dict,
    and set values correctly — both in vectorized and row-wise modes."""

    # --- List columns ---

    def test_vectorized_list_input(self):
        df = pd.DataFrame({"items": [[1, 2], [3, 4, 5], [6]]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="count")
        def count_items(items):
            return items.str.len()

        result = pipewise.run()
        assert result["count"].tolist() == [2, 3, 1]

    def test_rowwise_list_input(self):
        df = pd.DataFrame({"items": [[1, 2], [3, 4, 5], [6]]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="summary", vectorized=False)
        def summarize_list(items):
            return f"len={len(items)}, sum={sum(items)}"

        result = pipewise.run()
        assert result["summary"].tolist() == [
            "len=2, sum=3",
            "len=3, sum=12",
            "len=1, sum=6",
        ]

    def test_vectorized_list_manipulation(self):
        df = pd.DataFrame({
            "coords": [[10, 20], [30, 40], [50, 60]],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs=["x", "y"])
        def split_coords(coords):
            xs = coords.str[0]
            ys = coords.str[1]
            return xs, ys

        result = pipewise.run()
        assert result["x"].tolist() == [10, 30, 50]
        assert result["y"].tolist() == [20, 40, 60]

    # --- Dict columns ---

    def test_vectorized_dict_input(self):
        df = pd.DataFrame({
            "attrs": [{"w": 1, "h": 2}, {"w": 3, "h": 4}, {"w": 5, "h": 6}],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs=["w", "h"])
        def extract_dict(attrs):
            return attrs.str["w"], attrs.str["h"]

        result = pipewise.run()
        assert result["w"].tolist() == [1, 3, 5]
        assert result["h"].tolist() == [2, 4, 6]

    def test_rowwise_dict_input(self):
        df = pd.DataFrame({
            "attrs": [{"w": 1, "h": 2}, {"w": 3, "h": 4}, {"w": 5, "h": 6}],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="area", vectorized=False)
        def compute_area(attrs):
            return attrs["w"] * attrs["h"]

        result = pipewise.run()
        assert result["area"].tolist() == [2, 12, 30]

    def test_dict_output_with_dict_input(self):
        df = pd.DataFrame({
            "info": [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="dict", vectorized=False)
        def enrich(info):
            return {"sum": info["a"] + info["b"], "product": info["a"] * info["b"]}

        result = pipewise.run()
        assert result["sum"].tolist() == [3, 7, 11]
        assert result["product"].tolist() == [2, 12, 30]

    # --- Set columns ---

    def test_vectorized_set_input(self):
        df = pd.DataFrame({
            "tags": [{"x", "y"}, {"a", "b", "c"}, {"z"}],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="tag_count")
        def count_tags(tags):
            return tags.apply(len)

        result = pipewise.run()
        assert result["tag_count"].tolist() == [2, 3, 1]

    def test_rowwise_set_input(self):
        df = pd.DataFrame({
            "tags": [{"x", "y"}, {"a", "b", "c"}, {"z"}],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="sorted_tags", vectorized=False)
        def sort_tags(tags):
            return ",".join(sorted(tags))

        result = pipewise.run()
        assert result["sorted_tags"].tolist() == [
            "x,y",
            "a,b,c",
            "z",
        ]

    def test_groupby_with_list_column(self):
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

        assert result["total"].tolist() == [3, 3, 15]

    def test_mixed_complex_types_in_kwargs(self):
        df = pd.DataFrame({
            "name": ["a", "b"],
            "props": [{"x": 1}, {"y": 2}],
            "scores": [[10, 20], [30]],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="report", vectorized=False)
        def build_report(name, **extra):
            parts = [f"name={name}"]
            for k, v in sorted(extra.items()):
                parts.append(f"{k}={v}")
            return " | ".join(parts)

        result = pipewise.run()
        assert "report" in result.columns
        assert len(result) == 2

    # --- Edge cases ---

    def test_empty_list_column(self):
        df = pd.DataFrame({"items": [[], [1], []]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="length", vectorized=False)
        def length(items):
            return len(items)

        result = pipewise.run()
        assert result["length"].tolist() == [0, 1, 0]

    def test_mixed_type_column_with_none(self):
        df = pd.DataFrame({"data": [[1], None, [2, 3]]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="length", vectorized=False)
        def get_length(data):
            if data is None:
                return -1
            return len(data)

        result = pipewise.run()
        assert result["length"].tolist() == [1, -1, 2]

    def test_typed_output_with_complex_input(self):
        df = pd.DataFrame({"vals": [[1, 2], [3, 4], [5, 6]]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs={"avg": float})
        def compute_avg(vals):
            return (sum(vals) / len(vals)) if vals else 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        assert str(result["avg"].dtype) == "float64"
        assert result["avg"].tolist() == [1.5, 3.5, 5.5]

    def test_register_output_schema_single_col(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b", output_schema={"b": {"dtype": "float"}})
        def half(a):
            return a / 2

        result = pipewise.run()
        assert str(result["b"].dtype).startswith("float")

    def test_register_side_effect_only(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)
        side_effects = []

        @pipewise.register(outputs=None)
        def track(a):
            side_effects.append(sum(a))

        pipewise.run()
        assert side_effects == [6]

    def test_register_output_with_schema_rule(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b", output_schema={"b": "integer"})
        def double(a):
            return a * 2

        result = pipewise.run()
        assert result["b"].tolist() == [2, 4, 6]

    # --- AttributeError fallback regression ---

    def test_series_split_auto_fallback(self):
        df = pd.DataFrame({"sc": ["a,b,c", "b,2,e", "c,3,5"]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="sc_list")
        def proc_string_as_list(sc):
            return sc.split(",")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        assert result["sc_list"].tolist() == [
            ["a", "b", "c"],
            ["b", "2", "e"],
            ["c", "3", "5"],
        ]

    def test_series_strip_auto_fallback(self):
        df = pd.DataFrame({"s": ["  hello  ", "  world  "]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="cleaned")
        def trim(s):
            return s.strip()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        assert result["cleaned"].tolist() == ["hello", "world"]

    def test_attribute_error_bubbles_when_fallback_off(self):
        df = pd.DataFrame({"sc": ["a,b", "c,d"]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="sc_list", fallback_on_vectorized_error=False)
        def proc_string_as_list(sc):
            return sc.split(",")

        with pytest.raises(PipewiseExecutionError):
            pipewise.run()

    def test_string_upper_auto_fallback(self):
        df = pd.DataFrame({"name": ["alice", "bob"]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="name_upper")
        def upper(name):
            return name.str.upper()

        result = pipewise.run()
        assert result["name_upper"].tolist() == ["ALICE", "BOB"]

    def test_mixed_numeric_and_string_columns(self):
        df = pd.DataFrame({
            "ac": [1, 2, 3, 4],
            "bc": [5, 6, 7, 8],
            "lc": [[1, 2, 3], [2, 5, 1], [7, 4, 2], [20, 34, 12]],
            "sc": ["a,b,c", "b,2,e", "c,3,5", "d,2,3"],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="sc_list")
        def proc_string_as_list(sc):
            return sc.split(",")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        assert list(result.columns) == ["ac", "bc", "lc", "sc", "sc_list"]
        assert result["sc_list"].tolist() == [
            ["a", "b", "c"],
            ["b", "2", "e"],
            ["c", "3", "5"],
            ["d", "2", "3"],
        ]


# ======================================================================
# Dtype and index alignment
# ======================================================================


class TestDtypeAndAlignment:
    """Verify correct handling of non-trivial index, dtype, and Series
    returns."""

    def test_non_contiguous_index(self):
        """DataFrame with a non-default integer index should work."""
        df = pd.DataFrame({"a": [10, 20, 30]}, index=[5, 7, 9])
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def double(a):
            return a * 2

        result = pipewise.run()
        assert result["b"].tolist() == [20, 40, 60]
        assert list(result.index) == [5, 7, 9]

    def test_vectorized_returns_dataframe(self):
        """Vectorized function returns a pd.DataFrame result."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs=["sum", "diff"])
        def compute(x, y):
            return pd.DataFrame({"sum": x + y, "diff": x - y})

        result = pipewise.run()
        assert result["sum"].tolist() == [5, 7, 9]
        assert result["diff"].tolist() == [-3, -3, -3]

    def test_vectorized_returns_dataframe_misaligned_index(self):
        """DataFrame result with different index gets correctly aligned."""
        df = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
        pipewise = Pipewise(df)

        @pipewise.register(outputs=["a", "b"])
        def misaligned(x):
            # Return a DataFrame with index [2, 1, 0]
            return pd.DataFrame(
                {"a": [10, 20, 30], "b": [1, 2, 3]},
                index=[2, 1, 0],
            )

        result = pipewise.run()
        # After reindex to [0, 1, 2], row 0 gets [30, 3], row 1 gets [20, 2], row 2 gets [10, 1]
        assert result["a"].tolist() == [30, 20, 10]
        assert result["b"].tolist() == [3, 2, 1]

    def test_vectorized_single_col_returns_series(self):
        """Single-output function returning a pd.Series should work."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def double(a):
            return a * 2

        result = pipewise.run()
        assert result["b"].tolist() == [2, 4, 6]

    def test_nan_value_dtype_preservation(self):
        """NaN in float columns should not corrupt dtype."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="y")
        def add_one(x):
            return x + 1

        result = pipewise.run()
        assert np.isnan(result["y"].iloc[1])
        assert result["y"].iloc[0] == 2.0
        assert result["y"].iloc[2] == 4.0

    def test_pd_na_and_none_mixed(self):
        """pd.NA and None values mixed in a column."""
        df = pd.DataFrame({
            "val": [1, pd.NA, 3, None],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="double", vectorized=False)
        def double_val(val):
            if pd.isna(val):
                return None
            return val * 2

        result = pipewise.run()
        assert result["double"].tolist()[:1] == [2]
        assert pd.isna(result["double"].iloc[1])
        assert result["double"].iloc[2] == 6
        assert pd.isna(result["double"].iloc[3])

    def test_boolean_column_with_na(self):
        """Boolean column with pd.NA."""
        df = pd.DataFrame({"cond": [True, False, pd.NA]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="inverted", vectorized=False)
        def invert(cond):
            if pd.isna(cond):
                return None
            return not cond

        result = pipewise.run()
        assert result["inverted"].tolist() == [False, True, None]

    def test_int_column_with_none_to_float_output(self):
        """Int values with None → output coerced to float64."""
        df = pd.DataFrame({"a": [10, 20, 30]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="half", vectorized=False)
        def half(a):
            if a > 15:
                return a / 2
            return None

        result = pipewise.run()
        assert pd.isna(result["half"].iloc[0])
        assert result["half"].iloc[1] == 10.0
        assert result["half"].iloc[2] == 15.0

    def test_empty_dataframe(self):
        """0-row DataFrame should not break the pipeline."""
        df = pd.DataFrame({"a": pd.Series([], dtype=int)})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def add(a):
            return a + 1

        result = pipewise.run()
        assert len(result) == 0
        assert list(result.columns) == ["a", "b"]

    def test_single_row_dataframe(self):
        """Single-row DataFrame should work like any other."""
        df = pd.DataFrame({"x": [42]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="y")
        def compute(x):
            return x * 2

        result = pipewise.run()
        assert len(result) == 1
        assert result["y"].iloc[0] == 84


# ======================================================================
# More complex types
# ======================================================================


class TestMoreComplexTypes:
    """Verify correct handling of datetime, timedelta, categorical,
    numpy scalars, and mixed-type DataFrames."""

    def test_datetime64_column(self):
        df = pd.DataFrame({
            "d": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"]),
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="year")
        def extract_year(d):
            return d.dt.year

        result = pipewise.run()
        assert result["year"].tolist() == [2024, 2024, 2024]

    def test_datetime64_plus_timedelta(self):
        df = pd.DataFrame({
            "d": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="tomorrow")
        def next_day(d):
            return d + pd.Timedelta(days=1)

        result = pipewise.run()
        expected = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        assert (result["tomorrow"] == expected).all()

    def test_timedelta_column(self):
        df = pd.DataFrame({
            "a": pd.to_timedelta(["1 days", "2 days", "3 days"]),
            "b": pd.to_timedelta(["12 hours", "6 hours", "1 hours"]),
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="total")
        def add_time(a, b):
            return a + b

        result = pipewise.run()
        expected = pd.to_timedelta(["36 hours", "54 hours", "73 hours"])
        assert (result["total"] == expected).all()

    def test_categorical_column(self):
        df = pd.DataFrame({
            "cat": pd.Categorical(["low", "medium", "high"]),
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="code")
        def cat_codes(cat):
            return cat.cat.codes

        result = pipewise.run()
        # "high"=0, "low"=1, "medium"=2 (lexicographic)
        assert result["code"].tolist() == [1, 2, 0]

    def test_numpy_int64_input(self):
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def double(a):
            return a * 2

        result = pipewise.run()
        assert result["b"].tolist() == [2, 4, 6]
        assert result["b"].dtype == np.int64

    def test_numpy_float32_preservation(self):
        df = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def half(a):
            return a / 2

        result = pipewise.run()
        assert result["b"].dtype == np.float32
        assert result["b"].tolist() == pytest.approx([0.5, 1.0, 1.5])

    def test_mixed_int_float_str_bool_df(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="composite")
        def compose(int_col, float_col, str_col, bool_col):
            return (
                str_col + str(int_col) + str(int(float_col))
                + ("Y" if bool_col else "N")
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        assert result["composite"].tolist() == ["a11Y", "b22N", "c33Y"]

    def test_large_null_percentage(self):
        df = pd.DataFrame({
            "x": [None, None, None, 1.0, None, 2.0],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="y", vectorized=False)
        def process(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "missing"
            return f"val={x}"

        result = pipewise.run()
        assert result["y"].tolist() == [
            "missing", "missing", "missing", "val=1.0", "missing", "val=2.0",
        ]

    def test_all_null_column(self):
        df = pd.DataFrame({"x": [None, None, None]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="flag", vectorized=False)
        def flag(x):
            return "is_null" if x is None else "has_value"

        result = pipewise.run()
        assert result["flag"].tolist() == ["is_null", "is_null", "is_null"]


# ======================================================================
# Complex function logic
# ======================================================================


class TestComplexFunctionLogic:
    """Verify that complex business-logic patterns work through Pipewise."""

    def test_try_except_inside_function(self):
        df = pd.DataFrame({"raw": ["42", "not-a-number", "17"]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="parsed", vectorized=False)
        def safe_int(raw):
            try:
                return int(raw)
            except (ValueError, TypeError):
                return -1

        result = pipewise.run()
        assert result["parsed"].tolist() == [42, -1, 17]

    def test_nested_if_elif_else_tree(self):
        df = pd.DataFrame({
            "score": [45, 72, 88, 95, 60],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="grade", vectorized=False)
        def grade(score):
            if score >= 90:
                return "A"
            elif score >= 80:
                return "B"
            elif score >= 70:
                return "C"
            elif score >= 60:
                return "D"
            else:
                return "F"

        result = pipewise.run()
        assert result["grade"].tolist() == ["F", "C", "B", "A", "D"]

    def test_calling_helper_function(self):
        def helper(x):
            return x * x + 1

        df = pd.DataFrame({"n": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="result")
        def compute(n):
            return helper(n)

        result = pipewise.run()
        assert result["result"].tolist() == [2, 5, 10]

    def test_closure_with_external_variable(self):
        threshold = 10

        df = pd.DataFrame({"value": [5, 15, 25]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="label", vectorized=False)
        def label(value):
            if value > threshold:
                return "above"
            return "below"

        result = pipewise.run()
        assert result["label"].tolist() == ["below", "above", "above"]

    def test_lambda_helper(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        double = lambda x: x * 2  # noqa: E731

        @pipewise.register(outputs="b")
        def use_lambda(a):
            return double(a)

        result = pipewise.run()
        assert result["b"].tolist() == [2, 4, 6]

    def test_return_nested_structure(self):
        df = pd.DataFrame({
            "name": ["alice", "bob"],
            "scores": [[85, 90], [70, 80]],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="report", vectorized=False)
        def build_report(name, scores):
            avg = sum(scores) / len(scores)
            return {
                "name": name,
                "avg": avg,
                "min": min(scores),
                "max": max(scores),
            }

        result = pipewise.run()
        report = result["report"].tolist()
        assert report[0] == {"name": "alice", "avg": 87.5, "min": 85, "max": 90}
        assert report[1] == {"name": "bob", "avg": 75.0, "min": 70, "max": 80}

    def test_math_module_usage(self):
        df = pd.DataFrame({"angle": [0, math.pi / 2, math.pi]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="sin")
        def compute_sin(angle):
            return angle.apply(math.sin)

        result = pipewise.run()
        assert result["sin"].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-10)

    def test_statistics_module_usage(self):
        df = pd.DataFrame({"values": [[1, 2, 3], [10, 20, 30], [5, 5, 5]]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="stdev", vectorized=False)
        def compute_stdev(values):
            return statistics.stdev(values)

        result = pipewise.run()
        assert result["stdev"].tolist() == pytest.approx([1.0, 10.0, 0.0], abs=1e-10)

    def test_json_roundtrip(self):
        df = pd.DataFrame({
            "data": [
                '{"key": "value1"}',
                '{"key": "value2"}',
                '{"key": "value3"}',
            ],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="parsed", vectorized=False)
        def parse_json(data):
            return json.loads(data)

        @pipewise.register(outputs="extracted", vectorized=False)
        def extract_key(parsed):
            return parsed["key"]

        result = pipewise.run()
        assert result["extracted"].tolist() == ["value1", "value2", "value3"]

    def test_ternary_expression(self):
        df = pd.DataFrame({"val": [-5, 0, 5]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="sign")
        def compute_sign(val):
            return val.apply(lambda x: "positive" if x > 0 else ("zero" if x == 0 else "negative"))

        result = pipewise.run()
        assert result["sign"].tolist() == ["negative", "zero", "positive"]

    def test_list_comprehension_in_rowwise(self):
        df = pd.DataFrame({
            "items": [[1, 2], [3, 4, 5], [6]],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="squared", vectorized=False)
        def square_all(items):
            return [x * x for x in items]

        result = pipewise.run()
        assert result["squared"].tolist() == [[1, 4], [9, 16, 25], [36]]

    def test_groupby_multi_column(self):
        df = pd.DataFrame({
            "region": ["East", "East", "West", "West", "East"],
            "product": ["A", "B", "A", "A", "B"],
            "sales": [100, 200, 150, 300, 250],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="total_sales", groupby=["region", "product"])
        def sum_by_region_product(sales):
            return sum(sales)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        # Group sums: East-A: 100, East-B: 200+250=450, West-A: 150+300=450
        expected = [100, 450, 450, 450, 450]
        assert result["total_sales"].tolist() == expected


# ======================================================================
# Vectorized hazard detection (AST-based)
# ======================================================================


class TestVectorizedHazardDetection:
    """Verify that the AST scanner emits warnings for vectorized-incompatible
    patterns at registration time."""

    def test_len_hazard_warning(self, caplog):
        df = pd.DataFrame({"items": [[1], [2, 3]]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="count")
        def bad_count(items):
            return len(items)  # Series len → row count, not per-element

        assert any("len(" in rec.message for rec in caplog.records), (
            f"Expected len() warning, got: {[r.message for r in caplog.records]}"
        )

    def test_isinstance_hazard_warning(self, caplog):
        df = pd.DataFrame({"x": [1, "a", 3]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="label")
        def bad_type_check(x):
            if isinstance(x, int):
                return "integer"
            return "other"

        assert any("isinstance" in rec.message for rec in caplog.records)

    def test_split_hazard_warning(self, caplog):
        df = pd.DataFrame({"s": ["a,b", "c,d"]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="parts")
        def bad_split(s):
            return s.split(",")

        assert any(".split()" in rec.message for rec in caplog.records)

    def test_strip_hazard_warning(self, caplog):
        df = pd.DataFrame({"x": ["  a  ", "  b  "]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="clean")
        def bad_strip(x):
            return x.strip()

        assert any(".strip()" in rec.message for rec in caplog.records)

    def test_lower_hazard_warning(self, caplog):
        df = pd.DataFrame({"name": ["ALICE"]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="lower")
        def bad_lower(name):
            return name.lower()

        assert any(".lower()" in rec.message for rec in caplog.records)

    def test_upper_hazard_warning(self, caplog):
        df = pd.DataFrame({"name": ["alice"]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="upper")
        def bad_upper(name):
            return name.upper()

        assert any(".upper()" in rec.message for rec in caplog.records)

    def test_type_hazard_warning(self, caplog):
        df = pd.DataFrame({"x": [1, "a"]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="t")
        def bad_type(x):
            return type(x)

        assert any("type(" in rec.message for rec in caplog.records)

    def test_subscript_int_hazard_warning(self, caplog):
        df = pd.DataFrame({"items": [[1, 2], [3, 4]]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="first")
        def bad_subscript(items):
            return items[0]

        assert any("[0]" in rec.message for rec in caplog.records)

    def test_subscript_str_hazard_warning(self, caplog):
        df = pd.DataFrame({"d": [{"k": 1}, {"k": 2}]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="val")
        def bad_dict_access(d):
            return d["k"]

        assert any("'k'" in rec.message for rec in caplog.records)

    def test_str_dot_access_no_warning(self, caplog):
        """Proper .str. accessors should NOT trigger warnings."""
        df = pd.DataFrame({"s": ["a,b", "c,d"]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="parts")
        def good_str(s):
            return s.str.split(",")

        assert not any("hazard" in rec.message.lower() or ".split()" in rec.message
                       for rec in caplog.records), (
            f"Should not warn for .str.split(), got: {[r.message for r in caplog.records]}"
        )

    def test_vectorized_false_no_warning(self, caplog):
        """vectorized=False should NOT trigger AST warning scans."""
        df = pd.DataFrame({"items": [[1], [2, 3]]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="count", vectorized=False)
        def count_items(items):
            return len(items)

        # Should NOT warn because vectorized=False
        violation_msgs = [r.message for r in caplog.records
                          if "may be vectorized-incompatible" in r.message]
        assert len(violation_msgs) == 0, (
            f"Should not warn when vectorized=False, got: {violation_msgs}"
        )

    def test_no_fallback_on_vec_error_suppresses_warning(self, caplog):
        """When fallback is off, AST warnings should still fire because
        vectorized=True means the function WILL receive a Series."""
        df = pd.DataFrame({"s": ["a,b"]})
        pipewise = Pipewise(df)

        caplog.set_level(logging.WARNING, logger="pipewise.core")

        @pipewise.register(outputs="p", fallback_on_vectorized_error=False)
        def bad(s):
            return s.split(",")

        # Warning should still fire — the hazard is real
        assert any("s.split" in rec.message for rec in caplog.records)


# ======================================================================
# Package-level metadata
# ======================================================================


class TestPackageMetadata:
    def test_version_and_author(self):
        import pipewise
        assert hasattr(pipewise, "__version__")
        assert hasattr(pipewise, "__author__")

    def test_exception_hierarchy(self):
        import pipewise.errors as err

        exception_classes = [
            err.PipewiseRegistrationError,
            err.PipewiseTaskSelectionError,
            err.PipewiseInputColumnError,
            err.PipewiseInputSchemaError,
            err.PipewiseOutputSchemaError,
            err.PipewiseGroupByError,
            err.PipewiseOutputAssignmentError,
            err.PipewiseTypeConversionError,
            err.PipewiseExecutionError,
        ]
        for exc_cls in exception_classes:
            assert issubclass(exc_cls, PipewiseError), (
                f"{exc_cls.__name__} is not a subclass of PipewiseError"
            )

    def test_schema_rule_is_exported(self):
        from pipewise import SchemaRule
        assert SchemaRule is not None
