"""Schema for the tidy DataFrame returned by CPS parsers.

Basic validation of the CPS data.
"""

import pandas as pd


def validate_cps_long(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a parsed CPS DataFrame; raises ValueError on failure."""
    if "Year" not in df.columns:
        raise ValueError("CPS DataFrame missing required 'Year' column")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"CPS DataFrame has duplicate columns: {dupes}")
    if len(df.columns) < 2:
        raise ValueError("CPS DataFrame has no SPS-derived variable columns")
    if df["Year"].isna().any() or not (df["Year"].between(1900, 2100)).all():
        raise ValueError("CPS DataFrame has invalid Year values")
    return df
