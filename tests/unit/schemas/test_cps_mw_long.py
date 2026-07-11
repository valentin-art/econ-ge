import pandas as pd
import pytest

from src.schemas.bronze.cps_mw_long import validate_cps_mw_long


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame({"Year": [1964, 1964], "hhid": [1, 2], "age": [34, 25]})


def test_valid_frame_passes() -> None:
    validate_cps_mw_long(_valid_df())  # should not raise


def test_rejects_missing_year_column() -> None:
    df = _valid_df().drop(columns=["Year"])
    with pytest.raises(ValueError, match="missing required 'Year'"):
        validate_cps_mw_long(df)


def test_rejects_duplicate_columns() -> None:
    df = _valid_df()
    df.columns = ["Year", "hhid", "hhid"]
    with pytest.raises(ValueError, match="duplicate columns"):
        validate_cps_mw_long(df)


def test_rejects_no_variable_columns() -> None:
    df = pd.DataFrame({"Year": [1964, 1964]})
    with pytest.raises(ValueError, match="no SPS-derived variable columns"):
        validate_cps_mw_long(df)


def test_rejects_invalid_year() -> None:
    df = _valid_df()
    df.loc[0, "Year"] = 50
    with pytest.raises(ValueError, match="invalid Year"):
        validate_cps_mw_long(df)
