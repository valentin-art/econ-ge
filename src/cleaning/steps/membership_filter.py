"""MembershipFilter: A Step that keeps rows where a variable is one of allowed
values."""

from collections.abc import Collection

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class MembershipFilter(Step):
    """Keeps rows where `column` is one of `allowed_values`; drops rows
    where it is null or any other value, reporting the two separately.
    """

    def __init__(
        self, name: str, column: str, allowed_values: Collection[object]
    ) -> None:
        super().__init__(name)
        self.column = column
        self.allowed_values = list(allowed_values)
        if not self.allowed_values:
            raise ValueError(
                f"MembershipFilter {name!r}: allowed_values is empty; "
                "this would drop every row"
            )
        self.required_columns = frozenset({column})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        # Filtering
        n_in = len(df)
        col = pl.col(self.column)
        n_missing = df.select(col.is_null().sum()).item()

        try:
            result = df.filter(col.is_in(self.allowed_values))
        except pl.exceptions.PolarsError as exc:
            raise ValueError(
                f"MembershipFilter {self.name!r}: column {self.column!r} has dtype "
                f"{df.schema[self.column]} but allowed_values are "
                f"{ {type(v).__name__ for v in self.allowed_values} }: {exc}"
            ) from exc
        dropped_reason_counts: dict[str, int] = {}
        if n_missing:
            dropped_reason_counts["missing"] = n_missing

        n_out_of_universe = n_in - len(result) - n_missing
        if n_out_of_universe:
            dropped_reason_counts["not_in_allowed_values"] = n_out_of_universe

        return result, StepReport(
            step_name=self.name,
            n_in=n_in,
            n_out=len(result),
            dropped_reason_counts=dropped_reason_counts,
        )
