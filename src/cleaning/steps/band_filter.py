"""BandFilter: a Step that keeps rows where a given column is between min_value
and max_value.
"""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class BandFilter(Step):
    def __init__(
        self,
        name: str,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        super().__init__(name)
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.required_columns = frozenset({column})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        n_in = len(df)
        col = pl.col(self.column)

        # Collect info for statistics
        missing = df.filter(col.is_null()).height
        below_min = (
            df.filter(col < self.min_value).height if self.min_value is not None else 0
        )
        above_max = (
            df.filter(col > self.max_value).height if self.max_value is not None else 0
        )

        # Filtering
        mask = col.is_not_null()
        if self.min_value is not None:
            mask = mask & (col >= self.min_value)
        if self.max_value is not None:
            mask = mask & (col <= self.max_value)
        result = df.filter(mask)

        # Make a report
        dropped_reason_counts: dict[str, int] = {}
        if below_min:
            dropped_reason_counts["below_min"] = below_min
        if above_max:
            dropped_reason_counts["above_max"] = above_max
        if missing:
            dropped_reason_counts["missing"] = missing

        return result, StepReport(
            step_name=self.name,
            n_in=n_in,
            n_out=len(result),
            dropped_reason_counts=dropped_reason_counts,
        )
