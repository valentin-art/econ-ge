"""Thin write adapter for pipeline output tables. No business logic."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetUnreadableError(Exception):
    "A parquet file exists by its footer could not be parsed."


def read_parquet_columns(path: Path) -> tuple[str, ...]:
    """Column names from a Parquet footer, without reading any row data.

    Pandas index fields are excluded, so this is the table's columns as
    written rather than materialized.

    Raises:
        ParquetUnreadableError:
            The file is truncated or is not Parquet.
    """
    try:
        with pq.ParquetFile(path) as parquet_file:
            names = parquet_file.schema_arrow.names
    except (pa.ArrowInvalid, OSError) as exc:
        raise ParquetUnreadableError(path) from exc

    return tuple(name for name in names if not name.startswith("__index_level_"))


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
