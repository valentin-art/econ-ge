from pathlib import Path

import pandas as pd
import pytest

from src.input_output.parquet import ParquetUnreadableError, read_parquet_columns


def test_read_parquet_columns_reads_names_from_the_footer(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    pd.DataFrame({"YEAR": [2006], "AGE": [42]}).to_parquet(path, index=False)

    assert read_parquet_columns(path) == ("YEAR", "AGE")


def test_read_parquet_columns_excludes_the_pandas_index(tmp_path: Path) -> None:
    path = tmp_path / "data_parquet"

    pd.DataFrame({"YEAR": [2006]}).to_parquet(path)

    assert read_parquet_columns(path) == ("YEAR",)


def test_read_parquet_column_raises_on_a_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.parquet"

    path.write_bytes(b"not a parquet file")

    with pytest.raises(ParquetUnreadableError):
        read_parquet_columns(path)
