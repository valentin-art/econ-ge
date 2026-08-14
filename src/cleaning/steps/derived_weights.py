"""DerivedWeightsStep: derives WGT_WKS/WGT_HRS/WGT_HRS_FT from ASECWT."""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext


class DerivedWeightsStep(Step):
    required_columns = frozenset({"ASECWT", "WKSWORK1", "UHRSWORKLY"})
    produced_columns = frozenset({"WGT_WKS", "WGT_HRS", "WGT_HRS_FT"})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        n_in = len(df)
        result = df.with_columns(
            (pl.col("ASECWT") * pl.col("WKSWORK1")).alias("WGT_WKS"),
            (pl.col("ASECWT") * pl.col("WKSWORK1") * pl.col("UHRSWORKLY")).alias(
                "WGT_HRS"
            ),
            (pl.col("ASECWT") * pl.col("UHRSWORKLY")).alias("WGT_HRS_FT"),
        )

        return result, StepReport(step_name=self.name, n_in=n_in, n_out=n_in)
