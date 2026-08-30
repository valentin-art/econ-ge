"""Pure bronze -> silver transform for BEA capital/output data.

Alternative to `pipelines.bea_pipeline`: that module runs the *full* ETL
(BEA API extract -> bronze parse -> Hall-Jorgenson silver panel used as CES
model input). This module skips extraction entirely and is a pure function
of already-bronzed parquet (`data/bronze/bea/{FixedAssets,NIPA}/*.parquet`)
— no BEA API key, no network call. It reuses the same feature functions
(`features.bea.*`) so the two pipelines never disagree on methodology.

Produces one row per year with nominal/real levels *and* indices for the
IT / non-IT capital buckets, rental prices, investment, the internal rate of
return, and the output-value aggregate. See docs/bea_silver_derivations.md
for the derivation of every column, with formulas.
"""

from pathlib import Path

import pandas as pd
import structlog

from src.config import sources
from src.features.bea.capital_services import effective_depreciation, tornqvist_index
from src.features.bea.deflators import (
    investment_deflator,
    real_net_stock,
    smooth_capital_gains,
)
from src.features.bea.internal_return import solve_r_t
from src.features.bea.output_value import build_output_value_aggregate
from src.features.bea.rental_prices import (
    compute_net_cost_rate,
    compute_rental_income,
    compute_rental_prices,
    rental_shares,
)
from src.input_output.parquet import read_parquet
from src.parsers.bea.asset_dim import build_asset_dim
from src.parsers.bea.parser_bea import bronze_path
from src.parsers.bea.wide import to_wide
from src.schemas.silver.bea_silver import validate_bea_silver

log = structlog.get_logger(__name__)


def run_bea_silver_pipeline(bronze_dir: Path) -> pd.DataFrame:
    """Build the BEA silver panel from bronze parquet.

    Parameters
    ----------
    bronze_dir : the `bea` bronze root, e.g. settings.paths.bronze / "bea"
                 (containing FixedAssets/ and NIPA/ subdirectories).

    Returns
    -------
    DataFrame indexed by Year, columns per schemas.silver.bea_silver.BeaSilverRow.
    """
    log.info("bea_silver_pipeline_start", bronze_dir=str(bronze_dir))

    asset_dim, asset_dim_nonres = build_asset_dim()
    delta = asset_dim_nonres.set_index("LineNumber")["delta_j"]
    bucket = asset_dim_nonres.set_index("LineNumber")["bucket"]
    idx_IT = bucket[bucket == "IT"].index
    idx_non_IT = bucket[bucket == "non_IT"].index

    df_net_stock = read_parquet(
        bronze_path(bronze_dir, "FixedAssets", sources.FA_TABLE_21)
    )
    df_stock_idx = read_parquet(
        bronze_path(bronze_dir, "FixedAssets", sources.FA_TABLE_24)
    )
    df_investment = read_parquet(
        bronze_path(bronze_dir, "FixedAssets", sources.FA_TABLE_25)
    )
    df_invest_idx = read_parquet(
        bronze_path(bronze_dir, "FixedAssets", sources.FA_TABLE_26)
    )
    df_nos = read_parquet(bronze_path(bronze_dir, "NIPA", sources.NIPA_TABLE_1_16))
    df_va_nom = read_parquet(bronze_path(bronze_dir, "NIPA", sources.VA_TABLE_NOMINAL))
    df_va_real = read_parquet(bronze_path(bronze_dir, "NIPA", sources.VA_TABLE_REAL))

    W_stock = to_wide(df_net_stock, asset_dim_nonres)
    W_qstock = to_wide(df_stock_idx, asset_dim_nonres)
    W_invest = to_wide(df_investment, asset_dim_nonres)
    W_qinvest = to_wide(df_invest_idx, asset_dim_nonres)

    P_invest = investment_deflator(W_invest, W_qinvest, sources.REF_YEAR)
    K_real = real_net_stock(W_stock, W_qstock, sources.REF_YEAR, sources.YEARS)
    Pi_smooth = smooth_capital_gains(
        P_invest, sources.PI_SMOOTH_WINDOW, sources.PI_FLOOR
    )

    # T11600 and FAAt201 are both UNIT_MULT=6 (millions USD) in all current BEA
    # vintages (verified live via extractors.bea_api.verify_unit_scale in
    # pipelines.bea_pipeline). This pipeline is bronze-only / offline, so that
    # network check is not repeated here — units are assumed aligned.
    nos_t = (
        df_nos[df_nos["LineNumber"] == sources.NOS_LINE_TOTAL]
        .set_index("Year")["DataValue"]
        .rename("Pi_t")
    )

    r_t = solve_r_t(nos_t, Pi_smooth, W_stock, delta).r_t

    net_cost_rate_df = compute_net_cost_rate(Pi_smooth, delta, r_t)
    P_rental = compute_rental_prices(P_invest, net_cost_rate_df)
    rental_inc = compute_rental_income(P_rental, K_real)

    omega_IT = rental_shares(rental_inc, bucket, "IT")
    omega_non_IT = rental_shares(rental_inc, bucket, "non_IT")
    RI_IT = rental_inc.loc[idx_IT].sum(axis=0)
    RI_non_IT = rental_inc.loc[idx_non_IT].sum(axis=0)

    # Real capital-services index: Tornqvist (superlative, rental-weighted) —
    # avoids the fixed-weight aggregation bias of simply summing heterogeneous
    # assets. Same series bea_pipeline.run_capital_pipeline uses as the CES
    # capital-services input (there named K_IT / K_non_IT).
    Ks_IT = tornqvist_index(K_real.loc[idx_IT], omega_IT, sources.REF_YEAR)
    Ks_non_IT = tornqvist_index(K_real.loc[idx_non_IT], omega_non_IT, sources.REF_YEAR)

    delta_IT = effective_depreciation(omega_IT, delta.loc[idx_IT])
    delta_non_IT = effective_depreciation(omega_non_IT, delta.loc[idx_non_IT])

    # ---- Capital stock: nominal & real levels + indices -----------------
    cap_IT_nom = W_stock.loc[idx_IT].sum(axis=0)
    cap_non_IT_nom = W_stock.loc[idx_non_IT].sum(axis=0)
    cap_IT_real = K_real.loc[idx_IT].sum(axis=0)
    cap_non_IT_real = K_real.loc[idx_non_IT].sum(axis=0)

    cap_IT_nom_idx = cap_IT_nom / cap_IT_nom.loc[sources.REF_YEAR]
    cap_non_IT_nom_idx = cap_non_IT_nom / cap_non_IT_nom.loc[sources.REF_YEAR]
    cap_IT_real_idx = Ks_IT
    cap_non_IT_real_idx = Ks_non_IT

    # ---- Rental price: nominal & real ------------------------------------
    # Raw (un-normalized) dual price = total rental income / Tornqvist
    # quantity index. Dollar-denominated because Ks_* = 1.0 in ref_year, so
    # nominal == real there (same anchoring as real_net_stock).
    rent_IT_nom = RI_IT / Ks_IT
    rent_non_IT_nom = RI_non_IT / Ks_non_IT

    # ---- Output value aggregate -------------------------------------------
    output_value = build_output_value_aggregate(
        df_va_nom, df_va_real, sources.VA_LINE_NONFARM, sources.REF_YEAR
    )
    Y_nom_idx = output_value.Y_nom / output_value.Y_nom.loc[sources.REF_YEAR]

    rent_IT_real = rent_IT_nom / output_value.P_output
    rent_non_IT_real = rent_non_IT_nom / output_value.P_output

    # ---- Capital income shares of output -----------------------------------
    share_IT = RI_IT / output_value.Y_nom
    share_non_IT = RI_non_IT / output_value.Y_nom

    # ---- Investment: nominal & real ----------------------------------------
    inv_IT_nom = W_invest.loc[idx_IT].sum(axis=0)
    inv_non_IT_nom = W_invest.loc[idx_non_IT].sum(axis=0)
    inv_IT_real = W_invest.loc[idx_IT].div(P_invest.loc[idx_IT]).sum(axis=0)
    inv_non_IT_real = W_invest.loc[idx_non_IT].div(P_invest.loc[idx_non_IT]).sum(axis=0)

    silver = pd.DataFrame(
        {
            "cap_it_nom": cap_IT_nom,
            "cap_nonit_nom": cap_non_IT_nom,
            "cap_it_real": cap_IT_real,
            "cap_nonit_real": cap_non_IT_real,
            "cap_it_nom_idx": cap_IT_nom_idx,
            "cap_nonit_nom_idx": cap_non_IT_nom_idx,
            "cap_it_real_idx": cap_IT_real_idx,
            "cap_nonit_real_idx": cap_non_IT_real_idx,
            "rent_it_nom": rent_IT_nom,
            "rent_nonit_nom": rent_non_IT_nom,
            "rent_it_real": rent_IT_real,
            "rent_nonit_real": rent_non_IT_real,
            "share_it": share_IT,
            "share_nonit": share_non_IT,
            "delta_it": delta_IT,
            "delta_nonit": delta_non_IT,
            "inv_it_nom": inv_IT_nom,
            "inv_nonit_nom": inv_non_IT_nom,
            "inv_it_real": inv_IT_real,
            "inv_nonit_real": inv_non_IT_real,
            "r_t": r_t,
            "y_nom": output_value.Y_nom,
            "y_real": output_value.Y_real,
            "y_nom_idx": Y_nom_idx,
            "y_real_idx": output_value.Y_real_idx,
            "p_output": output_value.P_output,
            "pi_output": output_value.pi_output,
        }
    )
    silver.index.name = "year"
    silver = silver.loc[sources.EFF_START : sources.YEAR_END]
    validate_bea_silver(silver)

    log.info("bea_silver_pipeline_complete", n_years=len(silver))
    return silver
