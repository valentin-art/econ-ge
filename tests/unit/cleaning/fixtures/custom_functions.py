"""Toy (df, context) -> df functions at a stable dotted import path, used
only to test `type: FunctionStep` resolution in `Pipeline.from_config()`
(via `src.cleaning.steps.registry._build_function_step`). Not real
aa_clean methodology - see `tests/unit/cleaning/steps/test_function_step.py`
for that same caveat on the underlying FunctionStep.
"""

import polars as pl

from src.cleaning.context import CleaningContext


def fill_missing_weeks_with_default(
    df: pl.DataFrame, context: CleaningContext, default: float = 26.0
) -> pl.DataFrame:
    return df.with_columns(pl.col("WKSWORK1").fill_null(default).alias("WKSWORK1"))
