import pandas as pd
import pytest

from src.features.bea.capital_services import (
    dual_price_normalized,
    effective_depreciation,
    tornqvist_index,
)


def test_tornqvist_index_equals_one_at_ref_year() -> None:
    K_df = pd.DataFrame(
        {2019: [100.0, 50.0], 2020: [110.0, 55.0], 2021: [121.0, 60.5]}, index=[1, 2]
    )
    omega_df = pd.DataFrame(
        {2019: [0.6, 0.4], 2020: [0.5, 0.5], 2021: [0.5, 0.5]}, index=[1, 2]
    )

    Ks = tornqvist_index(K_df, omega_df, ref_year=2020)

    assert Ks.loc[2020] == pytest.approx(1.0)


def test_tornqvist_index_matches_common_growth_rate() -> None:
    # Both assets grow at a constant 10%/yr — a Divisia index of components
    # growing at the same rate must grow at that same rate, regardless of weights.
    K_df = pd.DataFrame(
        {
            2019: [100.0, 50.0],
            2020: [110.0, 55.0],
            2021: [121.0, 60.5],
            2022: [133.1, 66.55],
        },
        index=[1, 2],
    )
    omega_df = pd.DataFrame(
        {2019: [0.7, 0.3], 2020: [0.5, 0.5], 2021: [0.3, 0.7], 2022: [0.5, 0.5]},
        index=[1, 2],
    )

    Ks = tornqvist_index(K_df, omega_df, ref_year=2021)

    assert Ks.loc[2022] / Ks.loc[2021] == pytest.approx(1.10, abs=1e-6)
    assert Ks.loc[2020] / Ks.loc[2021] == pytest.approx(1.0 / 1.10, abs=1e-6)


def test_dual_price_normalized_at_ref_year_and_off_year() -> None:
    RI_s = pd.Series({2020: 100.0, 2021: 121.0})
    Ks = pd.Series({2020: 1.0, 2021: 1.10})

    p = dual_price_normalized(RI_s, Ks, ref_year=2020, name="p_test")

    assert p.loc[2020] == pytest.approx(1.0)
    # raw = RI/Ks -> 100/1=100 (ref), 121/1.10=110 -> normalized = 110/100 = 1.10
    assert p.loc[2021] == pytest.approx(1.10)


def test_effective_depreciation_weighted_average() -> None:
    omega_df = pd.DataFrame({2020: [0.5, 0.5]}, index=[1, 2])
    delta_bucket = pd.Series({1: 0.30, 2: 0.10})

    delta_eff = effective_depreciation(omega_df, delta_bucket)

    assert delta_eff.loc[2020] == pytest.approx(0.20)
