"""Output-value aggregate: nonfarm business gross value added.

Scope-matches T11600 NOS + nonfarm-nonfinancial capital - the same sector
generates the output, the capital income, and the stock - so this is the
correct numeraire for writing the CES FOCs against p_IT/p_non_IT/r_t.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class OutputValueAggregate:
    """Nonfarm business gross value added, nominal/real/deflator/inflation."""

    Y_nom: pd.Series  # nominal GVA, $ millions
    Y_real: pd.Series  # real (chained) GVA, $ millions
    Y_real_idx: pd.Series  # real GVA index, = 1.0 in ref_year
    P_output: pd.Series  # implicit output-price deflator, = 1.0 in ref_year
    pi_output: pd.Series  # output-price inflation, pct_change of P_output


def build_output_value_aggregate(
    df_va_nom: pd.DataFrame,
    df_va_real: pd.DataFrame,
    line_nonfarm: int,
    ref_year: int,
) -> OutputValueAggregate:
    """Build the nonfarm business value-added aggregate from raw VA tables.

    Parameters
    ----------
    df_va_nom    : tidy long DataFrame from parsers.bea_bronze.parse_bea_table (T10305)
    df_va_real   : tidy long DataFrame from parsers.bea_bronze.parse_bea_table (T10306)
    line_nonfarm : LineNumber for "Nonfarm" business (config.sources.VA_LINE_NONFARM)
    ref_year     : normalization year (config.sources.REF_YEAR) - same base as Ps_*, Ks_*

    Notes
    -----
    Published T10304 price index = nominal/real × 100, so the implicit deflator
    (nominal/real) is exact; we renormalize it to ref_year = 1.0.
    """
    Y_nom = (
        df_va_nom[df_va_nom["LineNumber"] == line_nonfarm]
        .set_index("Year")["DataValue"]
        .rename("Y_nom")
    )
    Y_real = (
        df_va_real[df_va_real["LineNumber"] == line_nonfarm]
        .set_index("Year")["DataValue"]
        .rename("Y_real")
    )

    P_output = Y_nom / Y_real
    P_output = (P_output / P_output.loc[ref_year]).rename("P_output")
    pi_output = P_output.pct_change().rename("pi_output")
    Y_real_idx = (Y_real / Y_real.loc[ref_year]).rename("Y_real_idx")

    return OutputValueAggregate(
        Y_nom=Y_nom,
        Y_real=Y_real,
        Y_real_idx=Y_real_idx,
        P_output=P_output,
        pi_output=pi_output,
    )
