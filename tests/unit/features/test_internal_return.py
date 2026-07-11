import pandas as pd
import pytest

from src.features.bea.internal_return import solve_r_t


def test_solve_r_t_closed_form_two_assets() -> None:
    W_stock = pd.DataFrame({2020: [1000.0, 2000.0]}, index=[1, 2])
    Pi_smooth = pd.DataFrame({2020: [-0.05, 0.02]}, index=[1, 2])
    delta = pd.Series({1: 0.10, 2: 0.20})
    df_nos = pd.Series({2020: 300.0})

    result = solve_r_t(df_nos, Pi_smooth, W_stock, delta)

    assert result.sum_stock.loc[2020] == pytest.approx(3000.0)
    assert result.nos_rate.loc[2020] == pytest.approx(0.10)
    assert result.capgain_sum.loc[2020] == pytest.approx(-10.0)
    assert result.avg_delta_w.loc[2020] == pytest.approx(500.0 / 3000.0)
    assert result.r_t.loc[2020] == pytest.approx(290.0 / 3000.0)


def test_solve_r_t_positive_when_nos_dominates_capgain_drag() -> None:
    W_stock = pd.DataFrame({2020: [1000.0]}, index=[1])
    Pi_smooth = pd.DataFrame({2020: [-0.30]}, index=[1])  # heavy obsolescence
    delta = pd.Series({1: 0.10})
    df_nos = pd.Series({2020: 500.0})  # large NOS relative to stock

    result = solve_r_t(df_nos, Pi_smooth, W_stock, delta)

    assert result.r_t.loc[2020] > 0
