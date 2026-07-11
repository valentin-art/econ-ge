"""Investment price deflator, capital-gains smoothing, and real net stock.

Derived economic series from already-parsed wide tables - bronze->silver
feature engineering, not bronze parsing (see parsers/ for that).
"""

import pandas as pd


def investment_deflator(
    W_invest: pd.DataFrame,
    W_qinvest: pd.DataFrame,
    ref_year: int,
) -> pd.DataFrame:
    """Implicit investment price deflator p^I_{j,t}.

    Derived as:
        p^I_{j,t} = (X_{j,t} / X_{j,ref}) / (Z_{j,t} / Z_{j,ref})

    Equals 1.0 in ref_year by construction.  Asserts this on return.

    Parameters
    ----------
    W_invest  : current-cost investment (LineNumber × Year)
    W_qinvest : chain-quantity index for investment (LineNumber × Year)
    ref_year  : reference year; deflator = 1.0 here for all assets
    """
    X_ref = W_invest[ref_year]
    Z_ref = W_qinvest[ref_year]
    P = W_invest.div(X_ref, axis=0).div(W_qinvest.div(Z_ref, axis=0))
    assert (P[ref_year].round(6) == 1.0).all(), (
        f"Investment deflator != 1.0 in ref_year={ref_year} - "
        "check Table 2.5/2.6 year alignment."
    )
    return P


def smooth_capital_gains(
    P_invest: pd.DataFrame,
    window: int,
    floor: float,
) -> pd.DataFrame:
    """Smoothed investment-price growth rate pi_bar_{j,t}.

    Raw pi_{j,t} = pct_change of P_invest.
    Smoothed with centred MA of width `window` (min_periods=2 at edges).
    Clipped at `floor` to prevent extreme early-sample values.

    NOTE:
    For IT assets pi_bar is large and negative (prices fall ~15-25%/yr)
    which raises the net cost rate - economically correct (obsolescence risk).
    """
    Pi_raw = P_invest.pct_change(axis=1)
    Pi_smooth = Pi_raw.T.rolling(window=window, center=True, min_periods=2).mean().T
    return Pi_smooth.clip(lower=floor)


def real_net_stock(
    W_stock: pd.DataFrame,
    W_qstock: pd.DataFrame,
    ref_year: int,
    years: list[int],
) -> pd.DataFrame:
    """Real net stock K^N_{j,t} in millions of ref_year USD.

    K^N_{j,t} = V_{j,ref} * Q^N_{j,t} / Q^N_{j,ref}

    In ref_year:
    p^I_{j,ref} = 1, so V_{j,ref} = p^I * K = K (real = nominal).

    Parameters
    ----------
    W_stock  : current-cost net stock V_{j,t} (LineNumber × Year)
    W_qstock : chain-quantity index Q^N_{j,t} (LineNumber × Year)
    ref_year : anchor year
    years    : list of years to populate (typically YEARS from config)
    """
    V_ref = W_stock[ref_year]
    QN_ref = W_qstock[ref_year]
    K_real = pd.DataFrame(index=W_stock.index, columns=years, dtype=float)
    for yr in years:
        K_real[yr] = V_ref * W_qstock[yr] / QN_ref
    return K_real
