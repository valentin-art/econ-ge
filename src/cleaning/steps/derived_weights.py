"""DerivedWeightsStep: derives WGT_WKS/WGT_HRS/WGT_HRS_FT from ASECWT."""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class DerivedWeightsStep(Step):
    """Derives three ASECWT-scaled weight columns: `WGT_WKS = ASECWT *
    weeks_column`, `WGT_HRS = ASECWT * weeks_column * UHRSWORKLY`, and
    `WGT_HRS_FT = ASECWT * UHRSWORKLY` (no weeks term). `weeks_column`
    defaults to `WEEKS_WORKED` (the output of `bridge_weeks_pre_1976`, not
    the raw `WKSWORK1`) so pre-1976 rows get a weight instead of a silent
    null; pass `weeks_column="WKSWORK1"` to skip the bridge deliberately.
    """

    def __init__(self, name: str, weeks_column: str = "WEEKS_WORKED") -> None:
        super().__init__(name)
        self.weeks_column = weeks_column
        self.required_columns = frozenset({"ASECWT", weeks_column, "UHRSWORKLY"})
        self.produced_columns = frozenset({"WGT_WKS", "WGT_HRS", "WGT_HRS_FT"})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        n_in = len(df)
        weeks = pl.col(self.weeks_column)
        result = df.with_columns(
            (pl.col("ASECWT") * weeks).alias("WGT_WKS"),
            (pl.col("ASECWT") * weeks * pl.col("UHRSWORKLY")).alias("WGT_HRS"),
            (pl.col("ASECWT") * pl.col("UHRSWORKLY")).alias("WGT_HRS_FT"),
        )

        warnings = [
            f"{n} rows have null {c}"
            for c, n in result.select(
                pl.col(["WGT_WKS", "WGT_HRS", "WGT_HRS_FT"]).is_null().sum()
            )
            .row(0, named=True)
            .items()
            if n
        ]
        return result, StepReport(
            step_name=self.name, n_in=n_in, n_out=n_in, warnings=warnings
        )
