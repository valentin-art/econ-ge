"""Schema for the tidy DataFrame returned by parsers.cps_mw.

Unlike schemas.bronze.bea_long, the column set here is driven by whichever
SPS dictionary covers the extracted year (parsers.cps_mw.parse_sps_dictionary)
rather than being fixed in advance, so this validates structure — a Year
column plus at least one SPS-derived variable, no duplicate names — rather
than a fixed per-column patito model.
"""

import pandas as pd


def validate_cps_mw_long(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a parsed CPS Mare-Winship DataFrame; raises ValueError on failure."""
    if "Year" not in df.columns:
        raise ValueError("CPS-MW DataFrame missing required 'Year' column")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"CPS-MW DataFrame has duplicate columns: {dupes}")
    if len(df.columns) < 2:
        raise ValueError("CPS-MW DataFrame has no SPS-derived variable columns")
    if df["Year"].isna().any() or not (df["Year"].between(1900, 2100)).all():
        raise ValueError("CPS-MW DataFrame has invalid Year values")
    return df
