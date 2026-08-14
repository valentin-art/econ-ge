import polars as pl
import pytest

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.custom_functions import bridge_weeks_pre_1976


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


def test_bridges_pre_1976_row_to_its_cell_mean_fit_from_1976_78_rows() -> None:
    df = pl.DataFrame(
        {
            "YEAR": [1962, 1976, 1977],
            "WKSWORK1": [None, 40, 50],
            "WKSWORK2": [4, 4, 4],
            "FEMALE": [0, 0, 0],
            "RACE": [1, 1, 1],
        }
    )

    result, warnings = bridge_weeks_pre_1976(df, _context())

    assert result["WEEKS_WORKED"].to_list() == [45.0, 40.0, 50.0]
    assert warnings == []


def test_group_conditioning_gives_different_groups_different_bridged_values() -> None:
    df = pl.DataFrame(
        {
            "YEAR": [1962, 1962, 1976, 1976],
            "WKSWORK1": [None, None, 40, 20],
            "WKSWORK2": [4, 4, 4, 4],
            "FEMALE": [0, 1, 0, 1],
            "RACE": [1, 1, 1, 1],
        }
    )

    result, warnings = bridge_weeks_pre_1976(df, _context())

    bridged = dict(zip(df["FEMALE"].to_list(), result["WEEKS_WORKED"].to_list()))
    assert bridged[0] == 40.0
    assert bridged[1] == 20.0
    assert bridged[0] != bridged[1]


def test_year_1976_plus_uses_wkswork1_directly_regardless_of_fit() -> None:
    df = pl.DataFrame(
        {
            "YEAR": [1980],
            "WKSWORK1": [33],
            "WKSWORK2": [4],
            "FEMALE": [0],
            "RACE": [1],
        }
    )

    result, warnings = bridge_weeks_pre_1976(df, _context())

    assert result["WEEKS_WORKED"].to_list() == [33.0]


def test_pre_1976_row_with_no_matching_cell_in_fit_population_is_null() -> None:
    df = pl.DataFrame(
        {
            "YEAR": [1962, 1976],
            "WKSWORK1": [None, 40],
            "WKSWORK2": [4, 2],
            "FEMALE": [0, 0],
            "RACE": [1, 1],
        }
    )

    result, warnings = bridge_weeks_pre_1976(df, _context())

    assert result["WEEKS_WORKED"].to_list() == [None, 40.0]


def test_no_1976_78_rows_present_leaves_every_pre_1976_row_null() -> None:
    df = pl.DataFrame(
        {
            "YEAR": [1962, 1963],
            "WKSWORK1": [None, None],
            "WKSWORK2": [4, 4],
            "FEMALE": [0, 1],
            "RACE": [1, 1],
        }
    )

    result, warnings = bridge_weeks_pre_1976(df, _context())

    assert result["WEEKS_WORKED"].to_list() == [None, None]
    assert len(warnings) == 1
    assert "2" in warnings[0]  # 2 pre-1976 rows get a null WEEKS_WORKED


def test_raises_when_input_already_has_the_reserved_scratch_column() -> None:
    df = pl.DataFrame(
        {
            "YEAR": [1976],
            "WKSWORK1": [40],
            "WKSWORK2": [4],
            "FEMALE": [0],
            "RACE": [1],
            "_bracket_mean": [1.0],
        }
    )

    with pytest.raises(ValueError, match="_bracket_mean"):
        bridge_weeks_pre_1976(df, _context())
