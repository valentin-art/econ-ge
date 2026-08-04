"""Schema for the tidy DataFrame returned by IPUMS parsers.

Basic validation of the IPUMS data. No fixed column set: the columns depend
on which variables were requested in the extract, so this only checks shape,
not specific columns.
"""

import pandas as pd


def check_no_duplicate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if `df` has duplicate column names."""
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"IPUMS DataFrame has duplicate columns: {dupes}")


def validate_ipums_long(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a parsed IPUMS DataFrame; raises ValueError on failure."""
    check_no_duplicate_columns(df)
    if df.empty:
        raise ValueError("IPUMS DataFrame has no rows")
    return df
