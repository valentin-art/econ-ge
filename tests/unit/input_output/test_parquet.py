from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.input_output.parquet import ParquetUnreadableError, read_parquet_columns


def test_read_parquet_columns_reads_names_from_the_footer(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    pd.DataFrame({"YEAR": [2006], "AGE": [42]}).to_parquet(path, index=False)

    assert read_parquet_columns(path) == ("YEAR", "AGE")


def test_read_parquet_columns_excludes_the_pandas_index(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    # A default RangeIndex is stored as parquet metadata, never as a column -
    # slicing forces pandas to materialise it as __index_level_0__, which is
    # the case the filter exists for.
    df = pd.DataFrame({"YEAR": [2006, 2007]})
    df[df["YEAR"] > 2006].to_parquet(path)

    assert tuple(pq.ParquetFile(path).schema_arrow.names) == (
        "YEAR",
        "__index_level_0__",
    )
    assert read_parquet_columns(path) == ("YEAR",)


def test_read_parquet_columns_raises_on_a_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.parquet"

    path.write_bytes(b"not a parquet file")

    with pytest.raises(ParquetUnreadableError):
        read_parquet_columns(path)


def test_read_parquet_columns_raises_on_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        read_parquet_columns(path)


def test_read_parquet_columns_handles_a_range_index(tmp_path: Path) -> None:
    # pandas records a RangeIndex as a descriptor dict, not a column name -
    # it must be ignored, not fed to set().
    path = tmp_path / "range.parquet"
    pd.DataFrame({"YEAR": [2006, 2007]}).to_parquet(path)  # index=True default

    assert read_parquet_columns(path) == ("YEAR",)
