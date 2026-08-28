import pandas as pd
import pytest

from src.schemas.bronze.ipums_long import (
    bronze_column_deviations,
    modal_columns,
    validate_ipums_long,
)


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame({"YEAR": [2006, 2006], "AGE": [34, 25]})


def test_valid_frame_passes() -> None:
    validate_ipums_long(_valid_df())  # should not raise


def test_rejects_duplicate_columns() -> None:
    df = _valid_df()
    df.columns = ["YEAR", "YEAR"]
    with pytest.raises(ValueError, match="duplicate columns"):
        validate_ipums_long(df)


def test_rejects_empty_dataframe() -> None:
    df = pd.DataFrame({"YEAR": [], "AGE": []})
    with pytest.raises(ValueError, match="no rows"):
        validate_ipums_long(df)


# The shape of the real cps bronze layer when the cps2006_09s incident was
# found: many years at one column set, one year widened by a variable-delta
# pulled for it alone, one year overwritten by an extract of the wrong grain.
_UNIFORM = {"YEAR", "AGE", "SEX", "EDUC"}
_OBSERVED = {
    1962: _UNIFORM,
    1963: _UNIFORM,
    1988: _UNIFORM,
    1989: _UNIFORM | {"OINCWAGE"},
    2006: {"YEAR", "AGE", "WTFINL"},
}


def test_modal_columns_survives_one_corrupt_year() -> None:
    assert modal_columns(_OBSERVED) == _UNIFORM


def test_modal_columns_of_no_years_is_empty() -> None:
    assert modal_columns({}) == frozenset()


def test_modal_columns_breaks_a_tie_towards_the_wider_set() -> None:
    # Two shapes, one year each: judging a year against the narrower target
    # would call a year that kept its columns conformant with one that lost
    # them, so the wider set has to win.
    observed = {2005: _UNIFORM, 2006: {"YEAR", "AGE"}}
    assert modal_columns(observed) == _UNIFORM


def test_superset_year_is_not_a_deviation_but_missing_year_is() -> None:
    deviations = bronze_column_deviations(_OBSERVED, _UNIFORM)
    assert list(deviations) == [2006]
    missing, extra = deviations[2006]
    assert missing == ("EDUC", "SEX")
    assert extra == ("WTFINL",)


def test_no_deviations_when_every_year_covers_expected() -> None:
    assert bronze_column_deviations({1962: _UNIFORM, 1989: _UNIFORM}, _UNIFORM) == {}


def test_explicit_expected_columns_can_narrow_what_counts_as_a_deviation() -> None:
    # The caller declaring a smaller contract is how a partially-pulled
    # variable stops being everyone else's problem.
    assert bronze_column_deviations(_OBSERVED, {"YEAR", "AGE"}) == {}
