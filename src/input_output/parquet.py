"""Thin write adapter for pipeline output tables. No business logic."""

from pathlib import Path

import pandas as pd


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame, returning an empty DataFrame if the file doesn't exist."""
    if not path.exists():
        ValueError(f"File {path} does not exist, returning empty DataFrame")
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to Parquet, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
