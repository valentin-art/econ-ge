import polars as pl

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.topcode_cap import TopcodeCapFilter


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


def test_caps_column_at_ceiling() -> None:
    df = pl.DataFrame({"AGE": [85, 90, 95, 100]})

    result, report = TopcodeCapFilter("test", column="AGE", ceiling=90).apply(
        df, _context()
    )

    assert result["AGE"].to_list() == [85, 90, 90, 90]
    assert report.n_in == 4
    assert report.n_out == 4
    assert report.branches_taken == {"topcoded": 2, "unchanged": 2, "missing": 0}


def test_reusable_on_a_different_column() -> None:
    df = pl.DataFrame({"UHRSWORKLY": [40, 98, 99]})

    result, _ = TopcodeCapFilter("test", column="UHRSWORKLY", ceiling=98).apply(
        df, _context()
    )

    assert result["UHRSWORKLY"].to_list() == [40, 98, 98]


def test_required_and_produced_columns_reflect_constructor_column() -> None:
    step = TopcodeCapFilter("test", column="AGE", ceiling=90)

    assert step.required_columns == frozenset({"AGE"})
    assert step.produced_columns == frozenset({"AGE"})
