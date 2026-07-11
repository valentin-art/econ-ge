"""Internal rate of return r_t solved from capital income exhaustion identity."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class InternalReturn:
    """All outputs of solve_r_t - core result plus diagnostic components."""

    r_t: pd.Series  # internal rate of return
    sum_stock: pd.Series  # sum V_{j,t}  (nonresidential denominator)
    nos_rate: pd.Series  # NOS_t / sum(V)  (raw rate before capgain adj)
    capgain_sum: pd.Series  # sum pi_bar_{j,t} · V_{j,t}
    capgain_rate: pd.Series  # capgain_sum / sum_stock  (weighted-avg pi)
    avg_delta_w: pd.Series  # sum delta_j · V_{j,t} / sum_stock  (memo: CFC size)


def solve_r_t(
    df_nos: pd.Series,
    Pi_smooth: pd.DataFrame,
    W_stock: pd.DataFrame,
    delta: pd.Series,
) -> InternalReturn:
    """Solve for r_t from the NOS-consistent capital income exhaustion identity.

    Identity (nonresidential private assets):

        NOS_t = Σ_j ( r_t − pi_bar_{j,t} ) · V_{j,t}

    NOS already has CFC subtracted by BEA, so delta does not appear on the RHS.
    Using the GOS form (+ delta_j) with NOS on the LHS double-counts CFC and
    pushes r_t negative.

    Solving for r_t:

        r_t = ( NOS_t + sum_j pi_bar_{j,t} · V_{j,t} ) / sum_j V_{j,t}

    Parameters
    ----------
    df_nos    : NOS series (Year index), millions USD - use T11600 primary
    Pi_smooth : smoothed capital-gains rates pi_bar_{j,t} (LineNumber × Year)
    W_stock   : current-cost net stock V_{j,t} (LineNumber × Year)
    delta     : geometric depreciation rates delta_j (LineNumber index)

    Returns
    -------
    InternalReturn dataclass with r_t and diagnostic components.
    """
    sum_stock = W_stock.sum(axis=0)
    nos_rate = df_nos / sum_stock
    capgain_sum = Pi_smooth.mul(W_stock).sum(axis=0)
    capgain_rate = capgain_sum / sum_stock
    # NB: must broadcast delta (indexed by LineNumber) along axis=0 - plain
    # `delta * W_stock` aligns a Series against DataFrame *columns* (Year),
    # which don't overlap with LineNumber, silently producing all-NaN/0.
    avg_delta_w = W_stock.mul(delta, axis=0).sum(axis=0) / sum_stock

    r_t = (df_nos + capgain_sum) / sum_stock
    r_t.name = "r_t"

    return InternalReturn(
        r_t=r_t,
        sum_stock=sum_stock,
        nos_rate=nos_rate,
        capgain_sum=capgain_sum,
        capgain_rate=capgain_rate,
        avg_delta_w=avg_delta_w,
    )
