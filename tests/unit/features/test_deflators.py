import pandas as pd
import pytest

from src.features.bea.deflators import (
    investment_deflator,
    real_net_stock,
    smooth_capital_gains,
)


def test_investment_deflator_matches_formula() -> None:
    W_invest = pd.DataFrame({2018: [100.0], 2019: [110.0], 2020: [121.0]}, index=[1])
    W_qinvest = pd.DataFrame({2018: [100.0], 2019: [105.0], 2020: [110.0]}, index=[1])

    P = investment_deflator(W_invest, W_qinvest, ref_year=2019)

    assert P.loc[1, 2019] == pytest.approx(1.0)
    expected_2018 = (100.0 / 110.0) / (100.0 / 105.0)
    expected_2020 = (121.0 / 110.0) / (110.0 / 105.0)
    assert P.loc[1, 2018] == pytest.approx(expected_2018)
    assert P.loc[1, 2020] == pytest.approx(expected_2020)


def test_investment_deflator_raises_when_ref_year_missing() -> None:
    W_invest = pd.DataFrame({2019: [110.0], 2020: [121.0]}, index=[1])
    W_qinvest = pd.DataFrame({2019: [90.0], 2020: [110.0]}, index=[1])

    with pytest.raises(KeyError):
        investment_deflator(W_invest, W_qinvest, ref_year=2050)


def test_smooth_capital_gains_applies_floor() -> None:
    # A single asset whose price collapses in one year — raw pct_change << floor.
    P_invest = pd.DataFrame(
        {2018: [1.0], 2019: [1.0], 2020: [0.1], 2021: [0.1], 2022: [0.1]}, index=[1]
    )

    smoothed = smooth_capital_gains(P_invest, window=3, floor=-0.40)

    # First year is NaN by construction (pct_change has no prior year) — the
    # floor only constrains values that were actually computed.
    assert (smoothed.dropna(axis=1) >= -0.40).all(axis=None)


def test_smooth_capital_gains_centred_average() -> None:
    # Constant 10% growth each year -> smoothed pi should equal 0.10 in interior years.
    values = [1.00, 1.10, 1.21, 1.331, 1.4641]
    P_invest = pd.DataFrame(
        {y: [v] for y, v in zip(range(2018, 2023), values)}, index=[1]
    )

    smoothed = smooth_capital_gains(P_invest, window=3, floor=-0.99)

    assert smoothed.loc[1, 2020] == pytest.approx(0.10, abs=1e-6)


def test_real_net_stock_identity_at_ref_year() -> None:
    W_stock = pd.DataFrame({2019: [500.0], 2020: [550.0]}, index=[1])
    W_qstock = pd.DataFrame({2019: [100.0], 2020: [104.0]}, index=[1])

    K_real = real_net_stock(W_stock, W_qstock, ref_year=2019, years=[2019, 2020])

    assert K_real.loc[1, 2019] == pytest.approx(500.0)
    assert K_real.loc[1, 2020] == pytest.approx(500.0 * 104.0 / 100.0)
