import polars as pl
import pytest

from src.cleaning.context import (
    CleaningContext,
    DeflatorTableConfig,
    SourceProfile,
    UncoveredYearsPolicy,
)
from src.cleaning.steps.deflator_merge import DeflatorMergeStep
from src.harmonization.cps_tables import CPI_DEFLATOR, GDP_PCE_DEFLATOR


def _context(
    cpi_uncovered_years: UncoveredYearsPolicy = "warning",
    gdp_uncovered_years: UncoveredYearsPolicy = "warning",
) -> CleaningContext:
    return CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        deflators={
            "cpi": DeflatorTableConfig(
                values=CPI_DEFLATOR, uncovered_years=cpi_uncovered_years
            ),
            "gdp_pce": DeflatorTableConfig(
                values=GDP_PCE_DEFLATOR, uncovered_years=gdp_uncovered_years
            ),
        },
    )


def test_merges_deflators_keyed_by_year_minus_one() -> None:
    df = pl.DataFrame({"YEAR": [1983]})  # income year 1982

    result, report = DeflatorMergeStep("deflator_merge").apply(df, _context())

    assert result["CPI_DEFLATOR"].to_list() == [CPI_DEFLATOR[1982]]
    assert result["GDP_DEFLATOR"].to_list() == [GDP_PCE_DEFLATOR[1982]]
    assert report.warnings == []


def test_warns_on_year_with_no_deflator_entry() -> None:
    df = pl.DataFrame({"YEAR": [1963]})  # income year 1962 - documented gap

    result, report = DeflatorMergeStep("deflator_merge").apply(df, _context())

    assert result["CPI_DEFLATOR"].to_list() == [None]
    # Both tables share the 1962 gap, so each warns separately.
    assert len(report.warnings) == 2
    assert all("1962" in warning for warning in report.warnings)


def test_null_year_is_reported_separately_from_a_missing_table_entry() -> None:
    df = pl.DataFrame({"YEAR": [None, 1983]}, schema={"YEAR": pl.Int64})

    result, report = DeflatorMergeStep("deflator_merge").apply(df, _context())

    assert result["CPI_DEFLATOR"].to_list() == [None, CPI_DEFLATOR[1982]]
    assert len(report.warnings) == 1
    assert "null YEAR" in report.warnings[0]


def test_raises_when_uncovered_years_is_error_on_the_table() -> None:
    df = pl.DataFrame({"YEAR": [1963]})  # income year 1962 - documented gap

    context = _context(cpi_uncovered_years="error")

    with pytest.raises(ValueError, match=r"CPI deflator table 'cpi'.*\[1962\]"):
        DeflatorMergeStep("deflator_merge").apply(df, context)


def test_custom_keys_select_among_multiple_named_tables() -> None:
    context = CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        deflators={
            "cpi_alt": DeflatorTableConfig(
                values={1982: 2.0}, uncovered_years="warning"
            ),
            "gdp_alt": DeflatorTableConfig(
                values={1982: 3.0}, uncovered_years="warning"
            ),
        },
    )
    df = pl.DataFrame({"YEAR": [1983]})
    step = DeflatorMergeStep("deflator_merge", cpi_key="cpi_alt", gdp_key="gdp_alt")

    result, _ = step.apply(df, context)

    assert result["CPI_DEFLATOR"].to_list() == [2.0]
    assert result["GDP_DEFLATOR"].to_list() == [3.0]


def test_missing_table_key_raises_clear_error() -> None:
    step = DeflatorMergeStep("deflator_merge", cpi_key="not_configured")
    df = pl.DataFrame({"YEAR": [1983]})

    with pytest.raises(ValueError, match="not_configured.*available.*cpi"):
        step.apply(df, _context())


def test_validate_context_passes_when_both_tables_present() -> None:
    step = DeflatorMergeStep("deflator_merge")

    assert step.validate_context(_context()) == []


def test_validate_context_flags_missing_tables_before_apply() -> None:
    step = DeflatorMergeStep("deflator_merge")
    context = CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))

    issues = step.validate_context(context)

    assert len(issues) == 2
    assert "cpi" in issues[0]
    assert "gdp_pce" in issues[1]
