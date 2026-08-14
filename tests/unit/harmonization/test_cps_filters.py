import pandas as pd

from src.harmonization.cps_filters import (
    exclude_filter,
    membership_filter,
    not_missing_filter,
    range_filter,
)


def test_range_filter_applies_both_bounds() -> None:
    df = pd.DataFrame({"AGE": [15, 16, 40, 64, 65]})

    result = range_filter(df, "AGE", min_value=16, max_value=64)

    assert result["AGE"].tolist() == [16, 40, 64]


def test_range_filter_one_sided_bound_and_drops_nan() -> None:
    df = pd.DataFrame({"WKSWORK1": [0, 1, 30, 52, float("nan")]})

    result = range_filter(df, "WKSWORK1", min_value=1, max_value=52)

    assert result["WKSWORK1"].tolist() == [1, 30, 52]


def test_range_filter_no_bounds_keeps_everything() -> None:
    df = pd.DataFrame({"ASECWT": [-1.0, 0.0, 100.0]})

    result = range_filter(df, "ASECWT")

    assert result["ASECWT"].tolist() == [-1.0, 0.0, 100.0]


def test_membership_filter_keeps_only_allowed_values() -> None:
    df = pd.DataFrame({"POPSTAT": [1, 2, 3, 1]})

    result = membership_filter(df, "POPSTAT", [1])

    assert result.index.tolist() == [0, 3]


def test_exclude_filter_is_complement_of_membership_filter() -> None:
    df = pd.DataFrame({"CLASSWLY": [10, 13, 14, 22, 24, 29, 99]})

    result = exclude_filter(df, "CLASSWLY", [10, 13, 14, 24])

    assert result["CLASSWLY"].tolist() == [22, 29, 99]


def test_not_missing_filter_drops_rows_with_any_nan_in_given_columns() -> None:
    df = pd.DataFrame(
        {
            "INCWAGE": [1000.0, float("nan"), 2000.0],
            "WKSWORK1": [10, 20, float("nan")],
        }
    )

    result = not_missing_filter(df, ["INCWAGE", "WKSWORK1"])

    assert result.index.tolist() == [0]


def test_not_missing_filter_accepts_single_column_as_string() -> None:
    df = pd.DataFrame({"AGE": [16, float("nan"), 40]})

    result = not_missing_filter(df, "AGE")

    assert result["AGE"].tolist() == [16, 40]
