"""TopcodeCapStep: clips values of a column at a given ceiling."""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class TopcodeCapStep(Step):
    """Clips a column at a fixed ceiling (`value = min(value, ceiling)`),
    unlike `TopcodeAdjuster`'s year-banded multiplier - all rows are kept,
    values above `ceiling` are pulled down to it rather than dropped.
    """

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
        try:
            topcoded = df.filter(col > self.ceiling).height
            result = df.with_columns(col.clip(upper_bound=self.ceiling))
        except pl.exceptions.PolarsError as exc:
            raise ValueError(
                f"TopcodeCapStep {self.name!r}: column {self.column!r} has dtype "
                f"{df.schema[self.column]}, not comparable against ceiling="
                f"{self.ceiling!r}: {exc}"
            ) from exc

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
