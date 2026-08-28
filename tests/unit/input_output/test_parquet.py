from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.input_output.parquet import (
    ParquetUnreadableError,
    read_parquet,
    read_parquet_columns,
    write_parquet,
)


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


def test_read_parquet_raises_file_not_found_for_a_missing_file(tmp_path: Path) -> None:
    # Callers in bea_/cps_ pipelines have always seen FileNotFoundError here;
    # this pins the type, which has changed twice.
    with pytest.raises(FileNotFoundError):
        read_parquet(tmp_path / "missing.parquet")


def test_read_parquet_round_trips_a_frame(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    df = pd.DataFrame({"YEAR": [2006, 2007], "AGE": [42, 43]})
    write_parquet(df, path)

    pd.testing.assert_frame_equal(read_parquet(path), df)


def test_read_parquet_columns_rejects_a_file_with_unparseable_pandas_metadata(
    tmp_path: Path,
) -> None:
    # A foreign writer can leave a b'pandas' blob that is not JSON; that is a
    # footer we cannot describe, not a crash.
    path = tmp_path / "badmeta.parquet"
    table = pa.table({"YEAR": [2006]}).replace_schema_metadata(
        {b"pandas": b"{not json"}
    )
    pq.write_table(table, path)

    with pytest.raises(ParquetUnreadableError):
        read_parquet_columns(path)
