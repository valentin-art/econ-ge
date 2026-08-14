"""TopcodeCapStep: A Step that clips values of a column at a given ceiling."""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class TopcodeCapFilter(Step):
    def __init__(self, name: str, column: str, ceiling: float) -> None:
        super().__init__(name)
        self.column = column
        self.ceiling = ceiling
        self.required_columns = frozenset({column})
        self.produced_columns = frozenset({column})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        # Collect info for report
        n_in = len(df)
        col = pl.col(self.column)
        n_missing = df.select(col.is_null().sum()).item()
        topcoded = df.filter(col > self.ceiling).height

        # Filtering
        result = df.with_columns(col.clip(upper_bound=self.ceiling))

        return result, StepReport(
            step_name=self.name,
            n_in=n_in,
            n_out=n_in,
            branches_taken={
                "topcoded": topcoded,
                "unchanged": n_in - topcoded - n_missing,
                "missing": n_missing,
            },
        )
