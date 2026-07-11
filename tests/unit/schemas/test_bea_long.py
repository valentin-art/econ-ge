import pandas as pd
import pytest
from patito.exceptions import DataFrameValidationError

from src.schemas.bronze.bea_long import validate_bea_long


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LineNumber": [1, 2],
            "LineDescription": ["Total", "Computers"],
            "Year": [2020, 2021],
            "DataValue": [100.0, 200.0],
        }
    )


def test_valid_frame_passes() -> None:
    validate_bea_long(_valid_df())  # should not raise


def test_rejects_negative_year() -> None:
    df = _valid_df()
    df.loc[0, "Year"] = -5
    with pytest.raises(DataFrameValidationError):
        validate_bea_long(df)


def test_rejects_unexpected_extra_column() -> None:
    df = _valid_df()
    df["UnexpectedColumn"] = "oops"
    with pytest.raises(DataFrameValidationError):
        validate_bea_long(df)


def test_allows_nullable_data_value() -> None:
    df = _valid_df()
    df.loc[0, "DataValue"] = None
    validate_bea_long(df)  # should not raise — BEA suppresses some cells
