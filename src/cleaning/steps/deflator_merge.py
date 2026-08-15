"""DeflatorMergeStep: merges CPI and GDP-PCE deflators keyed by YEAR-1."""

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext, DeflatorTableConfig


class DeflatorMergeStep(Step):
    """Adds CPI_DEFLATOR/GDP_DEFLATOR, looked up by income year (YEAR-1,
    since ASEC income variables refer to the prior calendar year) against
    two named `DeflatorTableConfig` tables in `context.deflators` (default
    keys `"cpi"`/`"gdp_pce"` - see `config/cleaning/cps/deflators/*.yaml`,
    transcribed from `src.harmonization.cps_tables`). An income year absent
    from a table is either a raised error or a per-row warning, per that
    table's own `uncovered_years` policy - never a silent null with no
    trace, the way a bare Python dict lookup would be.
    """

    required_columns = frozenset({"YEAR"})
    produced_columns = frozenset({"CPI_DEFLATOR", "GDP_DEFLATOR"})

    def __init__(
        self, name: str, cpi_key: str = "cpi", gdp_key: str = "gdp_pce"
    ) -> None:
        super().__init__(name)
        self.cpi_key = cpi_key
        self.gdp_key = gdp_key

    def validate_context(self, context: CleaningContext) -> list[str]:
        issues = []
        for key in (self.cpi_key, self.gdp_key):
            if key not in context.deflators:
                issues.append(
                    f"no deflator table named {key!r} in context.deflators "
                    f"(available: {sorted(context.deflators)})"
                )
        return issues

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        n_in = len(df)
        try:
            cpi_cfg = context.deflators[self.cpi_key]
        except KeyError:
            raise ValueError(
                f"DeflatorMergeStep {self.name!r}: no deflator table named "
                f"{self.cpi_key!r} in context.deflators "
                f"(available: {sorted(context.deflators)})"
            ) from None
        try:
            gdp_cfg = context.deflators[self.gdp_key]
        except KeyError:
            raise ValueError(
                f"DeflatorMergeStep {self.name!r}: no deflator table named "
                f"{self.gdp_key!r} in context.deflators "
                f"(available: {sorted(context.deflators)})"
            ) from None

        income_year = pl.col("YEAR") - 1
        result = df.with_columns(
            income_year.replace_strict(
                dict(cpi_cfg.values), default=None, return_dtype=pl.Float64
            ).alias("CPI_DEFLATOR"),
            income_year.replace_strict(
                dict(gdp_cfg.values), default=None, return_dtype=pl.Float64
            ).alias("GDP_DEFLATOR"),
        )

        n_null_year = df.select(pl.col("YEAR").is_null().sum()).item()
        income_years_present = (
            df.select(income_year.alias("income_year"))
            .unique()["income_year"]
            .to_list()
        )

        warnings: list[str] = []
        if n_null_year:
            warnings.append(
                f"{n_null_year} rows have a null YEAR and get null deflators"
            )
        for label, key, cfg in (
            ("CPI", self.cpi_key, cpi_cfg),
            ("GDP", self.gdp_key, gdp_cfg),
        ):
            self._check_coverage(label, key, cfg, income_years_present, warnings)

        return result, StepReport(
            step_name=self.name, n_in=n_in, n_out=n_in, warnings=warnings
        )

    def _check_coverage(
        self,
        label: str,
        key: str,
        cfg: DeflatorTableConfig,
        income_years_present: list[int | None],
        warnings: list[str],
    ) -> None:
        missing_years = sorted(
            year
            for year in income_years_present
            if year is not None and year not in cfg.values
        )
        if not missing_years:
            return
        message = (
            f"{label} deflator table {key!r} has no entry for income "
            f"year(s) {missing_years}"
        )
        if cfg.uncovered_years == "error":
            raise ValueError(
                f"DeflatorMergeStep {self.name!r}: {message}; extend "
                f"config/cleaning/cps/deflators/{key}.yaml or set "
                "uncovered_years: warning there"
            )
        warnings.append(message)
