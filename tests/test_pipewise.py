import io
import unittest
import warnings
from contextlib import redirect_stdout

import pandas as pd

from pipewise import Pipewise
from pipewise.errors import (
    PipewiseExecutionError,
    PipewiseInputColumnError,
    PipewiseInputSchemaError,
    PipewiseOutputSchemaError,
    PipewiseRegistrationError,
)


class PipewiseTestCase(unittest.TestCase):
    def test_vectorized_multi_output(self):
        df = pd.DataFrame({"a": [5, 15, 25], "b": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs=["sum", "product"])
        def calc(a, b):
            return a + b, a * b

        result = pipewise.run()

        self.assertEqual(result["sum"].tolist(), [6, 17, 28])
        self.assertEqual(result["product"].tolist(), [5, 30, 75])

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

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pipewise.run()

        self.assertEqual(result["level"].tolist(), ["low", "medium", "high"])
        self.assertEqual(result["doubled"].tolist(), [10, 20, 28])
        self.assertTrue(any("fell back to row-wise" in str(item.message) for item in caught))

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

        self.assertEqual(result["level"].tolist(), ["low", "medium", "high"])
        self.assertEqual(result["suggest"].dropna().tolist(), [10.0])
        self.assertEqual(result["adjust"].dropna().tolist(), [102.0])
        self.assertEqual(result["remark"].dropna().tolist(), ["too large"])

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

        self.assertEqual(
            result["classified"].tolist(),
            ["mid", "high", "low", "high", "high"],
        )

    def test_typed_outputs(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs={"quotient": float, "is_large": bool})
        def compute(a, b):
            return a / b, a > 2

        result = pipewise.run()

        self.assertEqual(str(result["quotient"].dtype), "float64")
        self.assertEqual(str(result["is_large"].dtype), "bool")
        self.assertEqual(result["is_large"].tolist(), [False, False, True])

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
        self.assertEqual(validated.equals(valid_df), True)

        invalid_df = pd.DataFrame({"a": [1, None, 3]})
        invalid_pipewise = Pipewise(
            invalid_df,
            input_schema={"a": {"dtype": "number", "nullable": False}},
        )

        with self.assertRaises(PipewiseInputSchemaError):
            invalid_pipewise.run()

    def test_task_output_schema_validation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="score", output_schema={"score": {"dtype": "integer", "max": 20}})
        def score(a):
            return a * 10

        with self.assertRaises(PipewiseExecutionError) as ctx:
            pipewise.run()

        self.assertIsInstance(ctx.exception.__cause__, PipewiseOutputSchemaError)
        self.assertEqual(list(df.columns), ["a"])

    def test_inplace_flag(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="doubled")
        def double(a):
            return a * 2

        result = pipewise.run()
        self.assertIn("doubled", result.columns)
        self.assertNotIn("doubled", df.columns)
        self.assertNotIn("doubled", pipewise.data.columns)

        inplace_df = pd.DataFrame({"a": [1, 2, 3]})
        inplace_pipewise = Pipewise(inplace_df)

        @inplace_pipewise.register(outputs="doubled")
        def double_inplace(a):
            return a * 2

        inplace_pipewise.run(inplace=True)
        self.assertIn("doubled", inplace_pipewise.data.columns)

    def test_task_management(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def step1(a):
            return a * 2

        @pipewise.register(outputs="c")
        def step2(b):
            return b + 10

        self.assertEqual(
            pipewise.tasks,
            [("step1", ["b"], None, True), ("step2", ["c"], None, True)],
        )
        self.assertTrue(pipewise.remove(step1))
        self.assertEqual(pipewise.tasks, [("step2", ["c"], None, True)])
        pipewise.clear()
        self.assertEqual(pipewise.tasks, [])

    def test_plan_output(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b", vectorized=False)
        def step1(a):
            return a * 2

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            pipewise.plan()

        output = buffer.getvalue()
        self.assertIn("Execution Plan:", output)
        self.assertIn("step1", output)
        self.assertIn("['b']", output)

    def test_rollback_on_failure(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def safe_step(a):
            return a * 2

        @pipewise.register(outputs="c")
        def bad_step(a):
            raise ValueError("fail")

        with self.assertRaises(PipewiseExecutionError):
            pipewise.run(inplace=True)

        self.assertEqual(list(df.columns), ["a"])
        self.assertEqual(list(pipewise.data.columns), ["a"])

    def test_missing_input_column_uses_custom_exception(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def needs_missing_column(missing_col):
            return missing_col * 2

        with self.assertRaises(PipewiseExecutionError) as ctx:
            pipewise.run()

        self.assertIsInstance(ctx.exception.__cause__, PipewiseInputColumnError)

    def test_kwargs_rowwise_mode(self):
        df = pd.DataFrame({"x": [1, 2], "y": [10, 20], "z": [100, 200]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="summary", vectorized=False)
        def summarize(x, **extra):
            extras = ", ".join(f"{k}={v}" for k, v in extra.items())
            return f"x={x}, {extras}"

        result = pipewise.run()

        self.assertEqual(
            result["summary"].tolist(),
            ["x=1, y=10, z=100", "x=2, y=20, z=200"],
        )

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

        self.assertEqual(result["b"].tolist(), [10, 20, 30])
        self.assertEqual(result["c"].tolist(), ["small", "big", "big"])

    def test_non_fallback_vectorized_error_bubbles_up(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pipewise = Pipewise(df)

        @pipewise.register(outputs="b")
        def broken(a):
            raise KeyError("boom")

        with self.assertRaises(PipewiseExecutionError) as ctx:
            pipewise.run()

        self.assertIsInstance(ctx.exception.__cause__, KeyError)

    def test_invalid_output_schema_target_raises_registration_error(self):
        pipewise = Pipewise(pd.DataFrame({"a": [1, 2, 3]}))

        with self.assertRaises(PipewiseRegistrationError):

            @pipewise.register(outputs="b", output_schema={"c": {"dtype": "integer"}})
            def invalid_schema(a):
                return a * 2


if __name__ == "__main__":
    unittest.main()
