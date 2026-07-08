"""Tests for Pipewise — a lightweight pandas DataFrame pipeline."""

from __future__ import annotations

import io
import logging
import warnings

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
# New: Complex input types — list, dict, set
# ======================================================================


class TestComplexInputTypes:
    """Verify that Pipewise handles DataFrame columns containing list, dict,
    and set values correctly — both in vectorized and row-wise modes."""

    # --- List columns ---

    def test_vectorized_list_input(self):
        """Vectorized: function receives a Series of lists."""
        df = pd.DataFrame({"items": [[1, 2], [3, 4, 5], [6]]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="count")
        def count_items(items):
            return items.str.len()

        result = pipewise.run()
        assert result["count"].tolist() == [2, 3, 1]

    def test_rowwise_list_input(self):
        """Row-wise: function receives individual lists per row."""
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
        """Vectorized: extract elements from lists."""
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
        """Vectorized: function receives a Series of dicts."""
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
        """Row-wise: function receives individual dicts per row."""
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
        """dict output + dict input: compute new keys from input dict."""
        df = pd.DataFrame({
            "info": [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
        })
        pipewise = Pipewise(df)

        @pipewise.register(outputs="dict", vectorized=False)
        def enrich(info):
            return {"sum": info["a"] + info["b"], "product": info["a"] * info["b"]}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pipewise.run()

        assert result["sum"].tolist() == [3, 7, 11]
        assert result["product"].tolist() == [2, 12, 30]

    # --- Set columns ---

    def test_vectorized_set_input(self):
        """Vectorized: function receives a Series of sets."""
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
        """Row-wise: function receives individual sets per row."""
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
        """GroupBy on standard column while processing list column within groups."""
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
        """Row-wise with **kwargs capturing list and dict columns."""
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
        """Empty lists in a column should not cause errors."""
        df = pd.DataFrame({"items": [[], [1], []]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="length", vectorized=False)
        def length(items):
            return len(items)

        result = pipewise.run()
        assert result["length"].tolist() == [0, 1, 0]

    def test_mixed_type_column_with_none(self):
        """Column with None/NaN alongside lists."""
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
        """Type conversion still works when inputs are complex types."""
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
        """outputs='col' with output_schema maps the dtype."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b", output_schema={"b": {"dtype": "float"}})
        def half(a):
            return a / 2

        result = pipewise.run()
        assert str(result["b"].dtype).startswith("float")

    def test_register_side_effect_only(self):
        """outputs=None means side-effect only, no columns written."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)
        side_effects = []

        @pipewise.register(outputs=None)
        def track(a):
            side_effects.append(sum(a))

        pipewise.run()
        assert side_effects == [6]

    def test_register_output_with_schema_rule(self):
        """outputs='col' with output_schema uses string dtype."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b", output_schema={"b": "integer"})
        def double(a):
            return a * 2

        result = pipewise.run()
        assert result["b"].tolist() == [2, 4, 6]


# ======================================================================
# Package-level metadata
# ======================================================================


class TestPackageMetadata:
    def test_version_and_author(self):
        import pipewise
        assert hasattr(pipewise, "__version__")
        assert hasattr(pipewise, "__author__")

    def test_exception_hierarchy(self):
        """All custom exceptions inherit from PipewiseError."""
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
            assert issubclass(exc_cls, PipewiseError), f"{exc_cls.__name__} is not a subclass of PipewiseError"

    def test_schema_rule_is_exported(self):
        """SchemaRule type alias is part of the public API."""
        from pipewise import SchemaRule
        assert SchemaRule is not None
