"""DeflatorMergeStep: merges CPI and GDP-PCE deflators keyed by YEAR-1."""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext
from src.harmonization.cps_tables import CPI_DEFLATOR, GDP_PCE_DEFLATOR


class DeflatorMergeStep(Step):
    required_columns = frozenset({"YEAR"})
    produced_columns = frozenset({"CPI_DEFLATOR", "GDP_DEFLATOR"})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        n_in = len(df)
        income_year = pl.col("YEAR") - 1

        result = df.with_columns(
            income_year.replace_strict(
                CPI_DEFLATOR, default=None, return_dtype=pl.Float64
            ).alias("CPI_DEFLATOR"),
            income_year.replace_strict(
                GDP_PCE_DEFLATOR, default=None, return_dtype=pl.Float64
            ).alias("GDP_DEFLATOR"),
        )

        income_years_present = (
            df.select((pl.col("YEAR") - 1).alias("income_year"))
            .unique()["income_year"]
            .to_list()
        )
        missing_years = sorted(
            year
            for year in income_years_present
            if year not in CPI_DEFLATOR or year not in GDP_PCE_DEFLATOR
        )
        warnings = (
            [f"no CPI/GDP deflator entry for income year(s) {missing_years}"]
            if missing_years
            else []
        )

        return result, StepReport(
            step_name=self.name, n_in=n_in, n_out=n_in, warnings=warnings
        )
