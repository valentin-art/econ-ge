import pandas as pd
import pytest

from src.features.bea.rental_prices import (
    compute_net_cost_rate,
    compute_rental_income,
    compute_rental_prices,
    rental_shares,
)


def test_net_cost_rate_formula() -> None:
    Pi_smooth = pd.DataFrame(
        {2020: [-0.10, 0.02]}, index=[1, 2]
    )  # asset 1 IT-like, 2 structure-like
    delta = pd.Series({1: 0.30, 2: 0.03})
    r_t = pd.Series({2020: 0.05})

    ncr = compute_net_cost_rate(Pi_smooth, delta, r_t)

    assert ncr.loc[1, 2020] == pytest.approx(0.30 - (-0.10) + 0.05)
    assert ncr.loc[2, 2020] == pytest.approx(0.03 - 0.02 + 0.05)


def test_rental_prices_and_income() -> None:
    P_invest = pd.DataFrame({2020: [2.0]}, index=[1])
    net_cost_rate = pd.DataFrame({2020: [0.25]}, index=[1])
    K_real = pd.DataFrame({2020: [100.0]}, index=[1])

    P_rental = compute_rental_prices(P_invest, net_cost_rate)
    assert P_rental.loc[1, 2020] == pytest.approx(0.5)

    rental_income = compute_rental_income(P_rental, K_real)
    assert rental_income.loc[1, 2020] == pytest.approx(50.0)


def test_rental_shares_sum_to_one_within_bucket() -> None:
    rental_income = pd.DataFrame({2020: [50.0, 30.0, 20.0]}, index=[1, 2, 3])
    bucket = pd.Series({1: "IT", 2: "IT", 3: "non_IT"})

    omega_it = rental_shares(rental_income, bucket, "IT")

    assert omega_it.loc[1, 2020] == pytest.approx(50.0 / 80.0)
    assert omega_it.loc[2, 2020] == pytest.approx(30.0 / 80.0)
    assert omega_it[2020].sum() == pytest.approx(1.0)
