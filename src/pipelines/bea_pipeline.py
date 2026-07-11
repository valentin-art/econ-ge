"""Core orchestration: BEA capital-services pipeline (Hall-Jorgenson framework).

Constructs asset-level rental prices (user costs) for IT / non-IT capital
aggregates, for use as inputs to a nested CES production function. Pure
function of `beakey` — no file I/O, no settings-singleton dependency — so it's
usable from the notebook, the CLI job (jobs/build_ces_capital_inputs.py), or a
test with a mocked extractor.

Key design decisions vs. naive implementation:
  - Restricted to NONRESIDENTIAL assets only: residential structures/equipment
    are excluded from both the capital stock and the r_t identity, making the
    stock denominator consistent with corporate net operating surplus, which
    excludes imputed rent on owner-occupied housing.
  - Internal rate of return r_t solved from the capital income exhaustion
    identity each year — not estimated from the CES, not taken from a market
    rate. This is pre-model arithmetic, keeping the CES identified.
  - Time-varying effective depreciation delta^s_t aggregated from detailed
    Hulten-Wykoff rates via rental-share-weighted Tornqvist — enters the
    aggregate accumulation equation as data, not as a calibrated constant.
  - Investment price deflator pi_{j,t} smoothed with a 3-year centred MA
    before entering the rental price formula — mandatory for IT assets where
    raw year-over-year changes can be extreme.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

from src.config import sources
from src.config.settings import settings
from src.extractors.bea_api import BEAExtractor, verify_unit_scale
from src.features.bea import diagnostics
from src.features.bea.capital_services import (
    dual_price_normalized,
    effective_depreciation,
    tornqvist_index,
)
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
from src.io.parquet import read_parquet
from src.parsers.asset_dim import build_asset_dim
from src.parsers.bea_bronze import parse_to_bronze
from src.parsers.wide import to_wide
from src.schemas.silver.ces_data import validate_ces_data

log = structlog.get_logger(__name__)


@dataclass
class CapitalPipelineResult:
    asset_dim: pd.DataFrame
    asset_dim_nonres: pd.DataFrame
    P_invest: pd.DataFrame
    K_real: pd.DataFrame
    Pi_smooth: pd.DataFrame
    r_t: pd.Series
    ces_inputs: pd.DataFrame
    ces_data: pd.DataFrame  # validated against schemas.silver.ces_data
    diagnostics: pd.DataFrame
    decomposition_it: pd.DataFrame
    decomposition_non_it: pd.DataFrame
    nonres_coverage: pd.Series
    scope_ratio: pd.Series


def _extract_and_parse(
    extractor: BEAExtractor, bronze_dir: Path, dataset: str, table: str
) -> pd.DataFrame:
    """extract() the raw JSON to external, parse+persist it to bronze, read it back."""
    record = extractor.extract(dataset=dataset, table=table)
    bronze_file = parse_to_bronze(
        record.file_path,
        dataset,
        table,
        bronze_dir,
        sources.YEAR_START,
        sources.YEAR_END,
    )
    return read_parquet(bronze_file)


def run_capital_pipeline(beakey: str) -> CapitalPipelineResult:
    """Run the full BEA capital-services pipeline end to end.

    Parameters
    ----------
    beakey : BEA API registration key (settings.bea_api_key)
    """
    log.info(
        "pipeline_start",
        ref_year=sources.REF_YEAR,
        year_start=sources.YEAR_START,
        year_end=sources.YEAR_END,
    )
    if beakey == "":
        raise RuntimeError(
            "BEA_API_KEY is empty - cannot run capital-services pipeline"
        )
    # ── 3. Fixed Assets tables ────────────────────────────────────────────
    extractor = BEAExtractor(api_key=beakey)
    bronze_dir = settings.paths.bronze / "bea"
    df_net_stock_raw = _extract_and_parse(
        extractor, bronze_dir, "FixedAssets", sources.FA_TABLE_21
    )
    df_stock_idx_raw = _extract_and_parse(
        extractor, bronze_dir, "FixedAssets", sources.FA_TABLE_24
    )
    df_investment_raw = _extract_and_parse(
        extractor, bronze_dir, "FixedAssets", sources.FA_TABLE_25
    )
    df_invest_idx_raw = _extract_and_parse(
        extractor, bronze_dir, "FixedAssets", sources.FA_TABLE_26
    )

    # ── 4. Asset dimension table ─────────────────────────────────────────
    asset_dim, asset_dim_nonres = build_asset_dim()
    delta = asset_dim_nonres.set_index("LineNumber")["delta_j"]
    bucket = asset_dim_nonres.set_index("LineNumber")["bucket"]

    total_stock = df_net_stock_raw[df_net_stock_raw["LineNumber"] == 1].set_index(
        "Year"
    )["DataValue"]
    nonres_stock = (
        df_net_stock_raw[
            df_net_stock_raw["LineNumber"].isin(asset_dim_nonres["LineNumber"])
        ]
        .groupby("Year")["DataValue"]
        .sum()
    )
    nonres_coverage = nonres_stock / total_stock
    log.info(
        "asset_dim_built",
        n_total=len(asset_dim),
        n_nonres=len(asset_dim_nonres),
        coverage_last=round(nonres_coverage.iloc[-1], 3),
    )

    # ── 5. Wide matrices (LineNumber x Year) ─────────────────────────────
    W_stock = to_wide(df_net_stock_raw, asset_dim_nonres)
    W_qstock = to_wide(df_stock_idx_raw, asset_dim_nonres)
    W_invest = to_wide(df_investment_raw, asset_dim_nonres)
    W_qinvest = to_wide(df_invest_idx_raw, asset_dim_nonres)

    # ── 6-8. Deflator, real stock, smoothed capital gains ────────────────
    P_invest = investment_deflator(W_invest, W_qinvest, sources.REF_YEAR)
    K_real = real_net_stock(W_stock, W_qstock, sources.REF_YEAR, sources.YEARS)
    Pi_smooth = smooth_capital_gains(
        P_invest, sources.PI_SMOOTH_WINDOW, sources.PI_FLOOR
    )

    # ── 9. NIPA capital income ────────────────────────────────────────────
    df_nipa_114 = _extract_and_parse(
        extractor, bronze_dir, "NIPA", sources.NIPA_TABLE_1_14
    )
    unit_scale = verify_unit_scale(beakey, sources.NIPA_TABLE_1_14, sources.FA_TABLE_21)
    df_nos_corp = (
        df_nipa_114[df_nipa_114["LineNumber"] == sources.NOS_LINE_CORP]
        .set_index("Year")["DataValue"]
        .mul(unit_scale)
        .rename("Pi_t_corp")
    )
    df_nipa_116 = _extract_and_parse(
        extractor, bronze_dir, "NIPA", sources.NIPA_TABLE_1_16
    )
    df_nos_total = (
        df_nipa_116[df_nipa_116["LineNumber"] == sources.NOS_LINE_TOTAL]
        .set_index("Year")["DataValue"]
        .mul(unit_scale)
        .rename("Pi_t_total")
    )
    scope_ratio = (df_nos_total / df_nos_corp).dropna()
    df_nos = df_nos_total.rename("Pi_t")
    log.info(
        "nos_scope_ratio",
        mean=round(scope_ratio.mean(), 3),
        min=round(scope_ratio.min(), 3),
        max=round(scope_ratio.max(), 3),
    )

    # ── 10. Internal rate of return r_t ──────────────────────────────────
    rt_result = solve_r_t(df_nos, Pi_smooth, W_stock, delta)
    r_t = rt_result.r_t
    sum_stock = rt_result.sum_stock
    nos_rate = rt_result.nos_rate

    diagnostics.log_r_t_decomp(
        rt_result.nos_rate, rt_result.avg_delta_w, rt_result.capgain_rate
    )
    diagnostics.log_r_t_diagnostic(r_t, rt_result.nos_rate, rt_result.capgain_rate)

    # ── 11. Rental prices p^K_{j,t} ──────────────────────────────────────
    net_cost_rate_df = compute_net_cost_rate(Pi_smooth, delta, r_t)
    neg_ncr = net_cost_rate_df[net_cost_rate_df <= 0].stack().dropna()
    if len(neg_ncr) > 0:
        log.warning("net_cost_rate_negative", n_cells=len(neg_ncr))
    P_rental = compute_rental_prices(P_invest, net_cost_rate_df)

    # ── 12. Rental income and shares omega_{j,t} ─────────────────────────
    rental_inc = compute_rental_income(P_rental, K_real)
    omega_IT = rental_shares(rental_inc, bucket, "IT")
    omega_non_IT = rental_shares(rental_inc, bucket, "non_IT")
    RI_IT = rental_inc.loc[bucket[bucket == "IT"].index].sum(axis=0)
    RI_non_IT = rental_inc.loc[bucket[bucket == "non_IT"].index].sum(axis=0)

    # ── 13. Tornqvist capital services index K^s_t ───────────────────────
    Ks_IT = tornqvist_index(
        K_real.loc[bucket[bucket == "IT"].index], omega_IT, sources.REF_YEAR
    ).rename("K_IT")
    Ks_non_IT = tornqvist_index(
        K_real.loc[bucket[bucket == "non_IT"].index], omega_non_IT, sources.REF_YEAR
    ).rename("K_non_IT")

    # ── 14. Dual rental price index p^s_t ────────────────────────────────
    Ps_IT = dual_price_normalized(RI_IT, Ks_IT, sources.REF_YEAR, "p_IT")
    Ps_non_IT = dual_price_normalized(
        RI_non_IT, Ks_non_IT, sources.REF_YEAR, "p_non_IT"
    )

    assert abs(Ps_IT.loc[sources.REF_YEAR] - 1.0) < 1e-9, "p_IT not 1.0 in REF_YEAR"
    assert (
        abs(Ps_non_IT.loc[sources.REF_YEAR] - 1.0) < 1e-9
    ), "p_non_IT not 1.0 in REF_YEAR"

    ri_ratio_IT = RI_IT / RI_IT.loc[sources.REF_YEAR]
    ri_ratio_non_IT = RI_non_IT / RI_non_IT.loc[sources.REF_YEAR]
    tol = 1e-6
    assert ((Ps_IT * Ks_IT) - ri_ratio_IT).abs().max() < tol, "Adding-up failed IT"
    assert (
        (Ps_non_IT * Ks_non_IT) - ri_ratio_non_IT
    ).abs().max() < tol, "Adding-up failed non-IT"
    log.info("adding_up_condition_verified")

    # ── 15. Effective depreciation delta^s_t ─────────────────────────────
    delta_IT = effective_depreciation(
        omega_IT, delta.loc[bucket[bucket == "IT"].index]
    ).rename("delta_IT")
    delta_non_IT = effective_depreciation(
        omega_non_IT, delta.loc[bucket[bucket == "non_IT"].index]
    ).rename("delta_non_IT")

    # ── 16. Rental-price decomposition + sanity checks ───────────────────
    decomposition_it = diagnostics.decompose_bucket(
        "IT",
        omega_IT,
        Ps_IT,
        delta_IT,
        bucket,
        P_invest,
        Pi_smooth,
        net_cost_rate_df,
        r_t,
    )
    decomposition_non_it = diagnostics.decompose_bucket(
        "non_IT",
        omega_non_IT,
        Ps_non_IT,
        delta_non_IT,
        bucket,
        P_invest,
        Pi_smooth,
        net_cost_rate_df,
        r_t,
    )
    diagnostics.log_decomp("IT", decomposition_it)
    diagnostics.log_decomp("non_IT", decomposition_non_it)

    diagnostics.run_sanity_checks(
        r_t,
        bucket,
        W_invest,
        P_invest,
        K_real,
        Ks_IT,
        Ks_non_IT,
        delta_IT,
        delta_non_IT,
        Ps_IT,
        Ps_non_IT,
    )

    # ── 17. CES inputs + diagnostics tables ──────────────────────────────
    ces_inputs = pd.concat(
        [Ks_IT, Ks_non_IT, Ps_IT, Ps_non_IT, delta_IT, delta_non_IT, r_t], axis=1
    )
    ces_inputs.index.name = "Year"
    ces_inputs = ces_inputs.loc[sources.EFF_START : sources.YEAR_END]

    diagnostics_table = pd.concat(
        [
            r_t,
            nos_rate.rename("NOS_over_sumV"),
            (df_nos_corp / sum_stock).rename("r_approx_corp"),
            delta_IT,
            delta_non_IT,
        ],
        axis=1,
    )
    diagnostics_table.index.name = "Year"

    # ── 18. Output value aggregate ───────────────────────────────────────
    df_va_nom = _extract_and_parse(
        extractor, bronze_dir, "NIPA", sources.VA_TABLE_NOMINAL
    )
    df_va_real = _extract_and_parse(
        extractor, bronze_dir, "NIPA", sources.VA_TABLE_REAL
    )
    output_value = build_output_value_aggregate(
        df_va_nom, df_va_real, sources.VA_LINE_NONFARM, sources.REF_YEAR
    )

    # ── 19. Full CES data panel ───────────────────────────────────────────
    ces_data = pd.concat(
        [
            Ks_IT,
            Ks_non_IT,
            Ps_IT,
            Ps_non_IT,
            (Ps_IT / output_value.P_output).rename("p_IT_real"),
            (Ps_non_IT / output_value.P_output).rename("p_non_IT_real"),
            delta_IT,
            delta_non_IT,
            r_t,
            (r_t - output_value.pi_output).rename("r_t_real"),
            output_value.Y_real,
            output_value.Y_real_idx,
            output_value.Y_nom,
            output_value.P_output,
        ],
        axis=1,
    )
    ces_data.index.name = "Year"
    ces_data = ces_data.loc[sources.EFF_START : sources.YEAR_END]
    validate_ces_data(ces_data)

    log.info("pipeline_complete", n_years=len(ces_data))

    return CapitalPipelineResult(
        asset_dim=asset_dim,
        asset_dim_nonres=asset_dim_nonres,
        P_invest=P_invest,
        K_real=K_real,
        Pi_smooth=Pi_smooth,
        r_t=r_t,
        ces_inputs=ces_inputs,
        ces_data=ces_data,
        diagnostics=diagnostics_table,
        decomposition_it=decomposition_it,
        decomposition_non_it=decomposition_non_it,
        nonres_coverage=nonres_coverage,
        scope_ratio=scope_ratio,
    )
