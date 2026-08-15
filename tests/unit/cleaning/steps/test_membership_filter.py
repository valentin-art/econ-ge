import polars as pl

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.membership_filter import MembershipFilter


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


def test_keeps_only_allowed_values() -> None:
    df = pl.DataFrame({"CLASSWLY": [10, 13, 14, 22, 24, 25, 27, 28, 29, 0, 99]})

    result, report = MembershipFilter(
        "test", column="CLASSWLY", allowed_values=[10, 13, 14, 22, 24, 25, 27, 28]
    ).apply(df, _context())

    assert result["CLASSWLY"].to_list() == [10, 13, 14, 22, 24, 25, 27, 28]
    assert report.n_in == 11
    assert report.n_out == 8
    assert report.dropped_reason_counts == {"not_in_allowed_values": 3}


def test_no_drops_reports_empty_dict() -> None:
    df = pl.DataFrame({"X": [1, 2]})

    _, report = MembershipFilter("test", column="X", allowed_values=[1, 2]).apply(
        df, _context()
    )

    assert report.dropped_reason_counts == {}


def test_reusable_on_a_different_column() -> None:
    df = pl.DataFrame({"POPSTAT": [1, 2, 3, 1]})

    result, _ = MembershipFilter("test", column="POPSTAT", allowed_values=[1]).apply(
        df, _context()
    )

    assert result["POPSTAT"].to_list() == [1, 1]


def test_required_columns_reflects_constructor_column() -> None:
    step = MembershipFilter("test", column="CLASSWLY", allowed_values=[10])

    assert step.required_columns == frozenset({"CLASSWLY"})
