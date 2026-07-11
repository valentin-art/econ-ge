"""Thin write adapter for pipeline output tables. No business logic."""

from pathlib import Path

import pandas as pd


def read_csv(path: Path, index_col: str | None = None) -> pd.DataFrame:
    """Read a CSV into a DataFrame, returning an empty DataFrame if the file doesn't exist."""
    if not path.exists():
        ValueError(f"File {path} does not exist, returning empty DataFrame")
    return pd.read_csv(path, index_col=index_col)


def write_csv(df: pd.DataFrame, path: Path, index: bool = True) -> Path:
    """Write a DataFrame to CSV, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path
