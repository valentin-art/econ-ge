import pandas as pd
import pytest
from patito.exceptions import DataFrameValidationError

from src.schemas.silver.ces_data import validate_ces_data


def _valid_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "K_IT": [1.0, 1.1],
            "K_non_IT": [1.0, 1.02],
            "p_IT": [1.0, 0.95],
            "p_non_IT": [1.0, 1.01],
            "p_IT_real": [1.0, 0.94],
            "p_non_IT_real": [1.0, 1.00],
            "delta_IT": [0.30, 0.31],
            "delta_non_IT": [0.05, 0.05],
            "r_t": [0.06, 0.061],
            "r_t_real": [0.05, 0.052],
            "Y_real": [1000.0, 1050.0],
            "Y_real_idx": [1.0, 1.05],
            "Y_nom": [1000.0, 1100.0],
            "P_output": [1.0, 1.02],
        },
        index=pd.Index([2020, 2021], name="Year"),
    )
    return df


def test_valid_ces_panel_passes() -> None:
    validate_ces_data(_valid_df())  # should not raise


def test_rejects_nonpositive_capital_services() -> None:
    df = _valid_df()
    df.loc[2020, "K_IT"] = -1.0
    with pytest.raises(DataFrameValidationError):
        validate_ces_data(df)


def test_rejects_depreciation_rate_out_of_range() -> None:
    df = _valid_df()
    df.loc[2020, "delta_IT"] = 1.5
    with pytest.raises(DataFrameValidationError):
        validate_ces_data(df)
