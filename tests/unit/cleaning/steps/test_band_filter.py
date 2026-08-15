import polars as pl

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.band_filter import BandFilter


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


def test_keeps_rows_at_or_above_min_value() -> None:
    df = pl.DataFrame({"AGE": [15, 16, 40, 64, 65]})

    result, report = BandFilter("age_band", column="AGE", min_value=16).apply(
        df, _context()
    )

    assert result["AGE"].to_list() == [16, 40, 64, 65]
    assert report.n_in == 5
    assert report.n_out == 4
    assert report.dropped_reason_counts == {"below_min": 1}


def test_respects_both_bounds() -> None:
    df = pl.DataFrame({"AGE": [15, 16, 40, 64, 65]})

    result, report = BandFilter(
        "age_band", column="AGE", min_value=16, max_value=64
    ).apply(df, _context())

    assert result["AGE"].to_list() == [16, 40, 64]
    assert report.dropped_reason_counts == {"below_min": 1, "above_max": 1}


def test_null_values_are_dropped_and_tracked_separately() -> None:
    df = pl.DataFrame({"AGE": [16, None, 40]})

    result, report = BandFilter("age_band", column="AGE", min_value=16).apply(
        df, _context()
    )

    assert result["AGE"].to_list() == [16, 40]
    assert report.dropped_reason_counts == {"missing": 1}


def test_reusable_on_a_different_column() -> None:
    # Same class, different column - the whole point of generalizing.
    df = pl.DataFrame({"WKSWORK1": [0, 1, 30, 52, 53]})

    result, report = BandFilter(
        "weeks_band", column="WKSWORK1", min_value=1, max_value=52
    ).apply(df, _context())

    assert result["WKSWORK1"].to_list() == [1, 30, 52]
    assert report.dropped_reason_counts == {"below_min": 1, "above_max": 1}


def test_required_columns_reflects_constructor_column() -> None:
    step = BandFilter("weeks_band", column="WKSWORK1", min_value=1)

    assert step.required_columns == frozenset({"WKSWORK1"})
