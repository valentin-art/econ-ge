"""MembershipFilter: A Step that keeps rows where a variable is one of allowed
values."""

from collections.abc import Collection

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class MembershipFilter(Step):
    def __init__(
        self, name: str, column: str, allowed_values: Collection[object]
    ) -> None:
        super().__init__(name)
        self.column = column
        self.allowed_values = allowed_values
        self.required_columns = frozenset({column})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        # Filtering
        n_in = len(df)
        result = df.filter(pl.col(self.column).is_in(self.allowed_values))

        # Collect info for report
        dropped = n_in - len(result)
        dropped_reason_counts = {"not_in_allowed_values": dropped} if dropped else {}

        return result, StepReport(
            step_name=self.name,
            n_in=n_in,
            n_out=len(result),
            dropped_reason_counts=dropped_reason_counts,
        )
