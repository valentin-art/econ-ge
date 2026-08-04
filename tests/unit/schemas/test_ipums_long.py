import pandas as pd
import pytest

from src.schemas.bronze.ipums_long import validate_ipums_long


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
