import polars as pl

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.deflator_merge import DeflatorMergeStep
from src.harmonization.cps_tables import CPI_DEFLATOR, GDP_PCE_DEFLATOR


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


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
    assert len(report.warnings) == 1
    assert "1962" in report.warnings[0]
