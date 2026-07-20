"""Tornqvist capital services index, dual rental price, effective depreciation."""

import numpy as np
import pandas as pd


def tornqvist_index(
    K_df: pd.DataFrame,
    omega_df: pd.DataFrame,
    ref_year: int,
) -> pd.Series:
    """Discrete-time Divisia (Tornqvist) capital services index K^s_t.

    Growth rate:
        d ln K^s_t = sum_j omega_bar_{j,t} * d ln K^N_{j,t}
    where omega_bar_{j,t} = 0.5 * (omega_{j,t} + omega_{j,t-1}).

    Rental-weighted aggregation gives capital *services* (the productive
    input to a production function).

    Parameters
    ----------
    K_df     : real net stock (LineNumber X Year) - asset subset for this bucket
    omega_df : rental shares within bucket (LineNumber X Year)
    ref_year : level anchor; index value = 1.0 in ref_year

    Returns
    -------
    pd.Series indexed by Year, equals 1.0 in ref_year.
    """
    log_K = np.log(K_df)
    dlog_K = log_K.diff(axis=1)  # type: ignore
    omega_bar = 0.5 * (omega_df + omega_df.shift(1, axis=1))
    dlog_Ks = (omega_bar * dlog_K).sum(axis=0)
    log_Ks = dlog_Ks.cumsum()
    log_Ks -= log_Ks[ref_year]
    return np.exp(log_Ks)


def dual_price_normalized(
    RI_s: pd.Series,
    Ks: pd.Series,
    ref_year: int,
    name: str,
) -> pd.Series:
    """Dual rental price index p^s_t, normalized to 1.0 in ref_year.

    Construction:
        raw p^s_t = RI_s_t / K^s_t
        p^s_t     = raw / raw[ref_year]   (dimensionless index)

    The normalized form is the correct object for CES estimation: relative
    movements matter, not nominal levels.  Adding-up in index form:
        p^s_t * K^s_t = RI_s_t / RI_s_ref  (ratio of rental incomes).

    Parameters
    ----------
    RI_s     : total rental income of bucket s (Series, Year index)
    Ks       : Tornqvist capital services index (= 1.0 in ref_year)
    ref_year : normalization year
    name     : output series name
    """
    p_raw = RI_s / Ks
    return (p_raw / p_raw.loc[ref_year]).rename(name)


def effective_depreciation(
    omega_df: pd.DataFrame,
    delta_bucket: pd.Series,
) -> pd.Series:
    """Rental-share-weighted effective depreciation rate delta^s_t.

    delta^s_t = sum_{j in s} omega_{j,t} * delta_j

    Time-varying because within-bucket composition shifts over time
    (e.g. software's growing share of IT capital raises delta^IT_t even
    though each delta_j is static).  Enters the aggregate accumulation
    equation as data, not a calibrated constant.

    Parameters
    ----------
    omega_df     : rental shares for the bucket (LineNumber X Year)
    delta_bucket : depreciation rates delta_j for assets in this bucket
    """
    return omega_df.mul(delta_bucket, axis=0).sum(axis=0)
