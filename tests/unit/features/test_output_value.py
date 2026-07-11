import pandas as pd
import pytest

from src.features.bea.output_value import build_output_value_aggregate


def _va_table(years: list[int], values: list[float], line: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LineNumber": [line] * len(years),
            "LineDescription": ["Nonfarm"] * len(years),
            "Year": years,
            "DataValue": values,
        }
    )


def test_output_value_normalized_at_ref_year() -> None:
    years = [2019, 2020, 2021]
    df_va_nom = _va_table(years, [1000.0, 1100.0, 1210.0])
    df_va_real = _va_table(
        years, [1000.0, 1050.0, 1100.0]
    )  # slower real growth -> inflation > 0

    result = build_output_value_aggregate(
        df_va_nom, df_va_real, line_nonfarm=3, ref_year=2019
    )

    assert result.P_output.loc[2019] == pytest.approx(1.0)
    assert result.Y_real_idx.loc[2019] == pytest.approx(1.0)
    assert result.Y_nom.loc[2020] == pytest.approx(1100.0)
    assert result.Y_real.loc[2020] == pytest.approx(1050.0)
    # nominal grew faster than real -> positive output-price inflation
    assert result.pi_output.loc[2020] > 0


def test_output_value_deflator_matches_nominal_over_real_ratio() -> None:
    years = [2019, 2020]
    df_va_nom = _va_table(years, [200.0, 220.0])
    df_va_real = _va_table(years, [200.0, 200.0])

    result = build_output_value_aggregate(
        df_va_nom, df_va_real, line_nonfarm=3, ref_year=2019
    )

    raw_ratio_2020 = (220.0 / 200.0) / (
        200.0 / 200.0
    )  # relative to ref-year raw ratio of 1.0
    assert result.P_output.loc[2020] == pytest.approx(raw_ratio_2020)
