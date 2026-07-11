"""Rental prices (user costs) and rental income shares."""

import pandas as pd


def compute_net_cost_rate(
    Pi_smooth: pd.DataFrame,
    delta: pd.Series,
    r_t: pd.Series,
) -> pd.DataFrame:
    """Net cost rate for each asset: r_t + delta_j − pi_bar_{j,t}.

    For IT: pi_bar < 0 (prices falling) → net cost rate is HIGH (obsolescence risk).
    For structures: pi_bar > 0 (prices rising slowly) → net cost rate is LOW.
    """
    return (
        Pi_smooth.rsub(delta, axis=0).add(  # delta_j − pi_bar_{j,t}
            r_t, axis=1
        )  # + r_t
    )


def compute_rental_prices(
    P_invest: pd.DataFrame,
    net_cost_rate: pd.DataFrame,
) -> pd.DataFrame:
    """Rental price (user cost) p^K_{j,t} = p^I_{j,t} * (r_t + delta_j − pi_bar_{j,t})."""
    return P_invest.mul(net_cost_rate)


def compute_rental_income(
    P_rental: pd.DataFrame,
    K_real: pd.DataFrame,
) -> pd.DataFrame:
    """Rental income RI_{j,t} = p^K_{j,t} * K^N_{j,t} (LineNumber × Year)."""
    return P_rental.mul(K_real)


def rental_shares(
    rental_income: pd.DataFrame,
    bucket: pd.Series,
    bkt: str,
) -> pd.DataFrame:
    """Rental income shares omega_{j,t} within bucket s.

    omega_{j,t} = RI_{j,t} / sum_{j in s} RI_{j,t}

    These are the Tornqvist weights for both quantity and price aggregation.
    """
    idx = bucket[bucket == bkt].index
    ri = rental_income.loc[idx]
    return ri.div(ri.sum(axis=0), axis=1)
