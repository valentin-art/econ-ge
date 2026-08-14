"""TopcodeAdjuster: A Step that processes top codes for a given column.

Requires context yaml files for topcodes in config/cleaning/{source}/topcode/*
Takes topcode values from the context and applies a transformation - i.e.,
multiplies on a given number.

Example: wages or earnings with corresponding set of top codes in context files.

Classes:
    TopCodeAdjuster
"""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class TopcodeAdjuster(Step):
    """Adjusts topcodes given the context.

    Attributes:
        name (str):
            Adjuster identifier.
        column (str):
            A name of the column with topcodes which will be adjusted.
        topcode_key (str):
            A key that determines a sub-context for the adjuster.
        requred_columns (set):
            A set of columns required to apply the step.
        produced_columns (set):
            A set of columns produced by the step.

    Methods:
        validate_context(...):
            Validates that the sub-context contains all necessary fields.
            Returns a list of issues if any.
        apply(...):
            Applies a methodology described in the sub-context.
    """

    def __init__(
        self, name: str, column: str = "INCWAGE", topcode_key: str = "wage"
    ) -> None:
        super().__init__(name)
        self.column = column
        self.topcode_key = topcode_key
        self.required_columns = frozenset({column, "YEAR"})
        self.produced_columns = frozenset({column})

    def validate_context(self, context: CleaningContext) -> list[str]:
        """A simple name validation for now. Need deeper context validation."""
        if self.topcode_key in context.topcode:
            return []
        return [
            f"no topcode config named {self.topcode_key!r} in context.topcode \
            (available: {sorted(context.topcode)})"
        ]

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        n_in = len(df)
        try:
            cfg = context.topcode[self.topcode_key]
        except KeyError:
            raise ValueError(
                f"TopcodeAdjuster {self.name!r}: no topcode config named "
                f"{self.topcode_key!r} in context.topcode "
                f"(available: {sorted(context.topcode)})"
            ) from None
        income = pl.col(self.column)
        hit_expr = pl.lit(False)
        mode_expr = pl.lit(None, dtype=pl.Utf8)

        # For each raw, set a dummy if a threshold (topcode) is reached.
        for band in cfg.thresholds:
            band_match = pl.col("YEAR").is_between(band.start_year, band.end_year)
            band_hit = (
                income >= band.threshold
                if band.match_mode == "gte"
                else income == band.threshold
            )

            hit_expr = pl.when(band_match).then(band_hit).otherwise(hit_expr)
            mode_expr = (
                pl.when(band_match).then(pl.lit(band.match_mode)).otherwise(mode_expr)
            )

        df = df.with_columns(
            hit_expr.alias("_topcode_hit"), mode_expr.alias("_topcode_mode")
        )
        uncovered = df.filter(pl.col("_topcode_mode").is_null())

        # Count topcode modes
        hit = pl.col("_topcode_hit").fill_null(False)
        exact_adjusted = df.filter(hit & (pl.col("_topcode_mode") == "exact")).height
        gte_adjusted = df.filter(hit & (pl.col("_topcode_mode") == "gte")).height
        in_band_not_hit = df.filter(~hit & pl.col("_topcode_mode").is_not_null()).height
        no_threshold = uncovered.height
        if no_threshold and cfg.uncovered_years == "error":
            years = sorted(uncovered["YEAR"].unique().to_list())
            raise ValueError(
                f"TopcodeAdjuster {self.name!r}: topcode config {self.topcode_key!r} has no "
                f"threshold band for survey year(s) {years}; {no_threshold} rows would pass "
                "through unadjusted. Extend the config or set uncovered_years: skip."
            )
        # Apply multiplier
        result = df.with_columns(
            pl.when(pl.col("_topcode_hit"))
            .then(income * cfg.multiplier)
            .otherwise(income)
            .alias(self.column)
        ).drop(["_topcode_hit", "_topcode_mode"])

        return result, StepReport(
            step_name=self.name,
            n_in=n_in,
            n_out=n_in,
            branches_taken={
                "exact_match": exact_adjusted,
                "gte_match": gte_adjusted,
                "in_band_not_hit": in_band_not_hit,
                "no_threshold_for_year": no_threshold,
            },
        )
