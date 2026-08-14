"""TopcodeAdjuster: A Step that processes top codes for a given column.

Requires context yaml files for topcodes in config/cleaning/{source}/topcode/*
Takes topcode values from the context and applies a transformation - i.e.,
multiplies on a given number.

Example: wages or earnings with corresponding set of top codes in context files.

Classes:
    TopcodeAdjuster
"""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class TopcodeAdjuster(Step):
    """Adjusts topcodes given the context.

    Threshold bands are matched on the survey `YEAR` (not the income year
    `YEAR-1` that `DeflatorMergeStep` keys on) - see the comment in `apply`.
    `column` is always returned as Float64, whatever it came in as, since the
    multiplier is a float; the promotion is unconditional so that the output
    schema does not depend on whether a given batch contained a topcoded row.

    Attributes:
        name (str):
            Adjuster identifier.
        column (str):
            A name of the column with topcodes which will be adjusted.
        topcode_key (str):
            A key that determines a sub-context for the adjuster.
        required_columns (set):
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
            f"no topcode config named {self.topcode_key!r} in context.topcode "
            f"(available: {sorted(context.topcode)})"
        ]

    _RESERVED_COLUMNS = ("_topcode_hit", "_topcode_mode")

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
        clash = set(self._RESERVED_COLUMNS) & set(df.columns)
        if clash:
            raise ValueError(
                f"TopcodeAdjuster {self.name!r}: input already has reserved "
                f"scratch column(s) {sorted(clash)}; rename them before this step"
            )
        income = pl.col(self.column)
        hit_expr = pl.lit(False)
        mode_expr = pl.lit(None, dtype=pl.Utf8)

        # For each row, set a dummy if a threshold (topcode) is reached.
        # Bands are matched on the survey YEAR, NOT on the income year
        # (YEAR-1) that DeflatorMergeStep uses. That asymmetry is deliberate:
        # the thresholds are transcribed from per-survey-year blocks of
        # aa_clean/clean7909km.do (see src.harmonization.cps_tables), so
        # survey-year matching is the faithful port. "Fixing" this to YEAR-1
        # for consistency with the deflators would silently shift every
        # threshold by one year.
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
        in_band = pl.col("_topcode_mode").is_not_null()
        income_is_null = income.is_null()
        exact_adjusted = df.filter(hit & (pl.col("_topcode_mode") == "exact")).height
        gte_adjusted = df.filter(hit & (pl.col("_topcode_mode") == "gte")).height
        null_income = df.filter(in_band & income_is_null).height
        in_band_not_hit = df.filter(~hit & in_band & ~income_is_null).height
        no_threshold = uncovered.height

        # A null YEAR matches no band, so those rows land in `uncovered` too -
        # but they are a different defect from "this year has no band yet", and
        # mixing None into the year list would make `sorted` raise TypeError.
        n_null_year = df.select(pl.col("YEAR").is_null().sum()).item()
        years = sorted(
            year for year in uncovered["YEAR"].unique().to_list() if year is not None
        )

        warnings: list[str] = []
        if n_null_year:
            warnings.append(
                f"{n_null_year} rows have a null YEAR, match no threshold band "
                "and pass through unadjusted"
            )
        if years:
            message = (
                f"topcode config {self.topcode_key!r} has no threshold band for "
                f"survey year(s) {years}; {no_threshold - n_null_year} rows pass "
                "through unadjusted"
            )
            if cfg.uncovered_years == "error":
                raise ValueError(
                    f"TopcodeAdjuster {self.name!r}: {message}. Extend the config "
                    "or set uncovered_years: skip."
                )
            # Mirror DeflatorMergeStep: an uncovered year is never a silent
            # pass-through, it is always visible in the StepReport.
            warnings.append(message)

        # Apply multiplier. cfg.multiplier is a float, so the multiplied branch
        # is inherently Float64 - cast BOTH branches explicitly rather than let
        # `otherwise` silently upcast. The cast is deliberately unconditional:
        # the output dtype is then a property of the step, not of whether this
        # particular batch happened to contain a topcoded row. Emitting a
        # warning for it would fire on every run under an integer input column
        # and would make `Pipeline(fail_on_warning=True)` permanently unusable.
        result = df.with_columns(
            pl.when(pl.col("_topcode_hit"))
            .then(income.cast(pl.Float64) * cfg.multiplier)
            .otherwise(income.cast(pl.Float64))
            .alias(self.column)
        ).drop(list(self._RESERVED_COLUMNS))

        return result, StepReport(
            step_name=self.name,
            n_in=n_in,
            n_out=n_in,
            branches_taken={
                "exact_match": exact_adjusted,
                "gte_match": gte_adjusted,
                "in_band_not_hit": in_band_not_hit,
                "null_income": null_income,
                "no_threshold_for_year": no_threshold,
            },
            warnings=warnings,
        )
