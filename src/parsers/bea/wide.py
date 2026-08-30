import pandas as pd


# TODO: make it more generic or move to BEA utils module
def to_wide(df_raw: pd.DataFrame, dim_table: pd.DataFrame) -> pd.DataFrame:
    """Pivot long BEA table to wide format (LineNumber × Year).

    Only lines present in dim_table are retained — dim_table is the join key.

    Parameters
    ----------
    df_raw    : tidy long DataFrame from parsers.bea_bronze.parse_bea_table
    dim_table : ASSET_DIM_NONRES (or ASSET_DIM) with a LineNumber column

    Returns
    -------
    DataFrame with index=LineNumber, columns=Year (int), no names on axes.
    """
    keep = df_raw[df_raw["LineNumber"].isin(dim_table["LineNumber"])].copy()
    wide = keep.pivot(index="LineNumber", columns="Year", values="DataValue")
    wide.columns.name = None
    wide.index.name = "LineNumber"
    return wide
