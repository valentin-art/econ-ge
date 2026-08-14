import polars as pl

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.derived_weights import DerivedWeightsStep


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


def test_derives_all_three_weighted_products() -> None:
    df = pl.DataFrame({"ASECWT": [100.0], "WKSWORK1": [50.0], "UHRSWORKLY": [40.0]})

    result, report = DerivedWeightsStep(
        "derived_weights", weeks_column="WKSWORK1"
    ).apply(df, _context())

    assert result["WGT_WKS"].to_list() == [5000.0]
    assert result["WGT_HRS"].to_list() == [200000.0]
    assert result["WGT_HRS_FT"].to_list() == [4000.0]
    assert report.n_in == 1
    assert report.n_out == 1


def test_default_weeks_column_reads_the_bridged_column_not_wkswork1() -> None:
    # Default weeks_column is WEEKS_WORKED (the output of
    # bridge_weeks_pre_1976), not the raw WKSWORK1 - a pre-1976 row has
    # WKSWORK1 null but a bridged, non-null WEEKS_WORKED.
    df = pl.DataFrame(
        {
            "ASECWT": [100.0],
            "WKSWORK1": [None],
            "WEEKS_WORKED": [45.0],
            "UHRSWORKLY": [40.0],
        }
    )

    result, _ = DerivedWeightsStep("derived_weights").apply(df, _context())

    assert result["WGT_WKS"].to_list() == [4500.0]
    assert result["WGT_HRS"].to_list() == [180000.0]
