import polars as pl
import pytest

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.topcode_cap import TopcodeCapStep


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


def test_caps_column_at_ceiling() -> None:
    df = pl.DataFrame({"AGE": [85, 90, 95, 100]})

    result, report = TopcodeCapStep("test", column="AGE", ceiling=90).apply(
        df, _context()
    )

    assert result["AGE"].to_list() == [85, 90, 90, 90]
    assert report.n_in == 4
    assert report.n_out == 4
    assert report.branches_taken == {"topcoded": 2, "unchanged": 2, "missing": 0}


def test_reusable_on_a_different_column() -> None:
    df = pl.DataFrame({"UHRSWORKLY": [40, 98, 99]})

    result, _ = TopcodeCapStep("test", column="UHRSWORKLY", ceiling=98).apply(
        df, _context()
    )

    assert result["UHRSWORKLY"].to_list() == [40, 98, 98]


def test_required_and_produced_columns_reflect_constructor_column() -> None:
    step = TopcodeCapStep("test", column="AGE", ceiling=90)

    assert step.required_columns == frozenset({"AGE"})
    assert step.produced_columns == frozenset({"AGE"})


def test_dtype_mismatch_raises_a_clear_error_naming_the_step_and_column() -> None:
    df = pl.DataFrame({"CLASSWLY": ["a", "b"]})

    with pytest.raises(ValueError, match="CLASSWLY"):
        TopcodeCapStep("test", column="CLASSWLY", ceiling=90).apply(df, _context())
