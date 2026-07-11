"""Diagnostic and sanity-check functions for the BEA capital pipeline.

All functions return DataFrames/Series for the caller (notebook, job) to
inspect or display; they log structured summaries via structlog instead of
printing directly.
"""

import numpy as np
import pandas as pd
import structlog

from src.config.sources import EFF_START, YEAR_END, YEAR_START

log = structlog.get_logger(__name__)


# ── Pre-solve decomposition ────────────────────────────────────────────────


def log_r_t_decomp(
    nos_rate: pd.Series,
    avg_delta_w: pd.Series,
    capgain_rate: pd.Series,
    year_end: int = YEAR_END,
) -> None:
    """Log decade-by-decade r_t pre-solve decomposition: NOS/ΣV, avg_delta, avg_pi.

    r_t(fix) = NOS/sum(V) + avg_pi              (NOS already net of CFC)
    r_t(bug) = NOS/sum(V) - avg_delta + avg_pi  (delta double-subtracted)
    """
    for yr in [1990, 2000, 2010, 2020, min(2024, year_end)]:
        if yr in nos_rate.index:
            n, d, pi = nos_rate.loc[yr], avg_delta_w.loc[yr], capgain_rate.loc[yr]
            log.info(
                "r_t_decomp",
                year=yr,
                nos_over_sum_v=round(n, 4),
                avg_delta=round(d, 4),
                avg_pi=round(pi, 4),
                r_t_fix=round(n + pi, 4),
                r_t_bug=round(n - d + pi, 4),
            )


def log_r_t_diagnostic(
    r_t: pd.Series,
    nos_rate: pd.Series,
    capgain_rate: pd.Series,
    year_end: int = YEAR_END,
) -> None:
    """Log r_t summary stats and warn if non-positive in any year."""
    last = min(2024, year_end)
    log.info(
        "r_t_summary",
        nos_over_sum_v_last=round(nos_rate.loc[last], 4),
        capgain_correction_last=round(capgain_rate.loc[last], 4),
        mean=round(r_t.mean(), 4),
        min=round(r_t.min(), 4),
        max=round(r_t.max(), 4),
    )

    if (r_t <= 0).any():
        neg_years = r_t[r_t <= 0].index.tolist()
        log.warning(
            "r_t_nonpositive",
            years=neg_years,
            possible_causes=[
                "ASSET_DIM incomplete — missing lines inflate avg_delta relative to NOS "
                "(expect nonres_coverage ~0.55-0.65 of private stock)",
                "NOS_LINE_TOTAL wrong in T11600 — check scope_ratio (expect 1.3-1.8x)",
                "Residential lines not fully excluded — check is_residential in ASSET_DIM",
            ],
        )
    else:
        log.info("r_t_positive_all_years")


# ── Rental-price decomposition ─────────────────────────────────────────────


def decompose_bucket(
    bkt: str,
    omega_df: pd.DataFrame,
    p_index: pd.Series,
    delta_eff: pd.Series,
    bucket: pd.Series,
    P_invest: pd.DataFrame,
    Pi_smooth: pd.DataFrame,
    net_cost_rate: pd.DataFrame,
    r_t: pd.Series,
) -> pd.DataFrame:
    """Decompose rental price growth into asset-price and net-cost-rate channels.

    Tornqvist dual:  d ln p^s ≈ sum_j w_bar_{j,t} * d ln p^K_{j,t}
        d ln p^K_j = d ln p^I_j  +  d ln(r + delta_j - pi_j)
        => c_price (asset-price channel) + c_rate (net-cost-rate channel)

    Net-cost-rate channel (first order):
        c_r  =  d r_t · sum_j (w_bar_j / rate_j)     (return channel)
        c_pi = -sum_j w_bar_j · d pi_{j,t} / rate_j   (capital-gains channel)

    Returns DataFrame with columns:
        dln_p_s, c_price, c_rate, c_r, c_pi, resid,
        r (level), delta_s (level), pi_s (level), ncr_s (level).
    """
    idx = bucket[bucket == bkt].index
    w_bar = 0.5 * (omega_df + omega_df.shift(1, axis=1))
    pI = P_invest.loc[idx]
    rate = net_cost_rate.loc[idx]
    pi_b = Pi_smooth.loc[idx]

    dln_pI = np.log(pI).diff(axis=1)  # type: ignore
    dln_rate = np.log(rate).diff(axis=1)  # type: ignore
    dr = r_t.diff()
    dpi = pi_b.diff(axis=1)

    c_price = (w_bar * dln_pI).sum(axis=0)
    c_rate = (w_bar * dln_rate).sum(axis=0)
    c_r = dr * (w_bar / rate).sum(axis=0)
    c_pi = -(w_bar * dpi / rate).sum(axis=0)

    dln_p_s = np.log(p_index).diff()  # type: ignore
    resid = dln_p_s - (c_price + c_rate)

    pi_s = (omega_df * pi_b).sum(axis=0)
    ncr_s = (omega_df * rate).sum(axis=0)

    return pd.DataFrame(
        {
            "dln_p_s": dln_p_s,
            "c_price": c_price,
            "c_rate": c_rate,
            "c_r": c_r,
            "c_pi": c_pi,
            "resid": resid,
            "r": r_t,
            "delta_s": delta_eff,
            "pi_s": pi_s,
            "ncr_s": ncr_s,
        }
    )


def log_decomp(name: str, dec: pd.DataFrame) -> None:
    """Log decade-mean rental-price decomposition for one capital bucket."""
    for start in range(YEAR_START, YEAR_END, 10):
        end = min(start + 9, YEAR_END)
        m = dec.loc[max(start, EFF_START) : end].mean()
        log.info(
            "rental_price_decomp",
            bucket=name,
            decade=f"{start}-{str(end)[-2:]}",
            dln_p=round(m["dln_p_s"], 4),
            c_price=round(m["c_price"], 4),
            c_rate=round(m["c_rate"], 4),
            c_r=round(m["c_r"], 4),
            c_pi=round(m["c_pi"], 4),
            resid=round(m["resid"], 4),
            r_level=round(m["r"], 3),
            delta_level=round(m["delta_s"], 3),
            pi_level=round(m["pi_s"], 3),
            ncr_level=round(m["ncr_s"], 3),
        )


# ── Sanity checks ──────────────────────────────────────────────────────────


def _agg_invest_deflator(
    bkt: str,
    bucket: pd.Series,
    W_invest: pd.DataFrame,
    P_invest: pd.DataFrame,
) -> pd.Series:
    mask = bucket == bkt
    nominal = W_invest.loc[mask].sum(axis=0)
    real = W_invest.loc[mask].div(P_invest.loc[mask]).sum(axis=0)
    return (nominal / real).rename(f"p_inv_{bkt}")


def _implied_delta(K_sum: pd.Series, I_real: pd.Series) -> pd.Series:
    """Implied delta from perpetual inventory: K_t = (1-d)*K_{t-1} + I_t.

    Both K_sum and I_real must be in the same dollar-level units (not indices).
    """
    return (1 - (K_sum - I_real) / K_sum.shift(1)).rename("delta_implied")


def run_sanity_checks(
    r_t: pd.Series,
    bucket: pd.Series,
    W_invest: pd.DataFrame,
    P_invest: pd.DataFrame,
    K_real: pd.DataFrame,
    Ks_IT: pd.Series,
    Ks_non_IT: pd.Series,
    delta_IT: pd.Series,
    delta_non_IT: pd.Series,
    Ps_IT: pd.Series,
    Ps_non_IT: pd.Series,
) -> None:
    """Run all three sanity checks and log the results.

    Check A — perpetual inventory consistency: implied delta vs. constructed.
              Uses dollar-level real stock (K_real bucket sum), not the
              Tornqvist index, so units match real investment.
    Check B — r_t magnitude and trend by decade.
    Check C — IT/non-IT rental price ratio trend.
              Note: with IT bucket = hardware + software + R&D, rental prices
              for IT can RISE over time (high and increasing delta_IT, software
              dominance) even as hardware asset prices fall.  Rising ratio is
              not necessarily a bug; interpret in conjunction with decomposition.
    """
    ps_inv_IT = _agg_invest_deflator("IT", bucket, W_invest, P_invest)
    ps_inv_non_IT = _agg_invest_deflator("non_IT", bucket, W_invest, P_invest)

    I_real_IT = W_invest.loc[bucket == "IT"].sum(axis=0) / ps_inv_IT
    I_real_non_IT = W_invest.loc[bucket == "non_IT"].sum(axis=0) / ps_inv_non_IT

    K_sum_IT = K_real.loc[bucket == "IT"].sum(axis=0)
    K_sum_non_IT = K_real.loc[bucket == "non_IT"].sum(axis=0)

    delta_IT_check = pd.concat(
        [delta_IT, _implied_delta(K_sum_IT, I_real_IT)], axis=1
    ).dropna()
    delta_IT_check["residual"] = (
        delta_IT_check["delta_IT"] - delta_IT_check["delta_implied"]
    )

    delta_non_IT_check = pd.concat(
        [delta_non_IT, _implied_delta(K_sum_non_IT, I_real_non_IT)], axis=1
    ).dropna()
    delta_non_IT_check["residual"] = (
        delta_non_IT_check["delta_non_IT"] - delta_non_IT_check["delta_implied"]
    )

    max_resid = max(
        delta_IT_check["residual"].abs().max(),
        delta_non_IT_check["residual"].abs().max(),
    )
    log.info(
        "check_a_delta_sanity",
        max_residual=round(max_resid, 4),
        within_tolerance=max_resid <= 0.05,
    )
    if max_resid > 0.05:
        log.warning("check_a_delta_residual_large", max_residual=round(max_resid, 4))

    log.info(
        "check_b_r_t_summary",
        mean=round(r_t.mean(), 4),
        min=round(r_t.min(), 4),
        max=round(r_t.max(), 4),
    )
    for start in range(YEAR_START, YEAR_END, 10):
        end = min(start + 9, YEAR_END)
        sl = r_t.loc[start:end]
        log.info(
            "check_b_r_t_by_decade",
            decade=f"{start}-{end}",
            mean=round(sl.mean(), 4),
            min=round(sl.min(), 4),
            max=round(sl.max(), 4),
        )

    price_ratio = (Ps_IT / Ps_non_IT).rename("p_IT_over_p_nonIT")
    pr = price_ratio.dropna()
    trend_dir = "rises" if pr.iloc[-1] > pr.iloc[0] else "falls"
    log.info(
        "check_c_it_price_ratio",
        first=round(pr.iloc[0], 3),
        last=round(pr.iloc[-1], 3),
        trend=trend_dir,
    )
