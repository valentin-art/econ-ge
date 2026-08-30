import json
from pathlib import Path

import pandas as pd

from src.parsers.bea.parser_bea import (
    bronze_path,
    load_bea_json,
    parse_bea_json,
    parse_bea_table,
    parse_to_bronze,
)


def _raw_df() -> pd.DataFrame:
    # Shape matches what BEAExtractor persists: beaapi's raw columns, DataValue
    # as comma-formatted strings, TimePeriod as string year.
    return pd.DataFrame(
        {
            "TableName": ["FAAt201"] * 4,
            "SeriesCode": ["k1ptotl1es00"] * 4,
            "LineNumber": [1, 1, 2, 2],
            "LineDescription": ["Private fixed assets"] * 2 + ["Equipment"] * 2,
            "TimePeriod": ["1974", "1975", "1974", "1975"],
            "CL_UNIT": ["Level"] * 4,
            "UNIT_MULT": [6] * 4,
            "METRIC_NAME": ["Current Dollars"] * 4,
            "DataValue": ["223,353", "231,574", "1,000", ""],
            "NoteRef": ["FAAt201"] * 4,
        }
    )


def test_parses_to_tidy_long_shape() -> None:
    out = parse_bea_table(_raw_df(), year_start=1975, year_end=2024)

    assert list(out.columns) == ["LineNumber", "LineDescription", "Year", "DataValue"]
    assert out["Year"].tolist() == [1975, 1975]  # 1974 filtered out by year_start


def test_strips_commas_from_datavalue() -> None:
    out = parse_bea_table(_raw_df(), year_start=1974, year_end=2024)
    row = out[(out["LineNumber"] == 1) & (out["Year"] == 1974)].iloc[0]
    assert row["DataValue"] == 223353.0


def test_blank_datavalue_becomes_nan() -> None:
    out = parse_bea_table(_raw_df(), year_start=1974, year_end=2024)
    row = out[(out["LineNumber"] == 2) & (out["Year"] == 1975)].iloc[0]
    assert pd.isna(row["DataValue"])


def test_year_window_filters_out_of_range_rows() -> None:
    out = parse_bea_table(_raw_df(), year_start=1975, year_end=1975)
    assert set(out["Year"].unique()) == {1975}
    assert len(out) == 2


def _write_bea_json(tmp_path: Path) -> Path:
    payload = {
        "BEAAPI": {
            "Request": {"RequestParam": []},
            "Results": {
                "Data": [
                    {
                        "LineNumber": "1",
                        "LineDescription": "Private fixed assets",
                        "TimePeriod": "1975",
                        "DataValue": "223,353",
                    },
                    {
                        "LineNumber": "1",
                        "LineDescription": "Private fixed assets",
                        "TimePeriod": "1976",
                        "DataValue": "231,574",
                    },
                ],
                "Notes": [],
            },
        }
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "FAAt201.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_bea_json_returns_raw_shaped_dataframe(tmp_path: Path) -> None:
    json_path = _write_bea_json(tmp_path)

    out = load_bea_json(json_path)

    assert list(out.columns) == [
        "LineNumber",
        "LineDescription",
        "TimePeriod",
        "DataValue",
    ]
    assert out["TimePeriod"].tolist() == ["1975", "1976"]


def test_parse_bea_json_returns_tidy_long_shape(tmp_path: Path) -> None:
    json_path = _write_bea_json(tmp_path)

    out = parse_bea_json(json_path, year_start=1975, year_end=2024)

    assert list(out.columns) == ["LineNumber", "LineDescription", "Year", "DataValue"]
    assert out["DataValue"].tolist() == [223353.0, 231574.0]


def test_bronze_path_builds_dataset_table_parquet_path(tmp_path: Path) -> None:
    path = bronze_path(tmp_path, "FixedAssets", "FAAt201")
    assert path == tmp_path / "FixedAssets" / "FAAt201.parquet"


def test_parse_to_bronze_writes_parquet_for_one_source_table(tmp_path: Path) -> None:
    json_path = _write_bea_json(tmp_path / "external")
    bronze_dir = tmp_path / "bronze"

    out_path = parse_to_bronze(
        json_path, "FixedAssets", "FAAt201", bronze_dir, 1975, 2024
    )

    assert out_path == bronze_dir / "FixedAssets" / "FAAt201.parquet"
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert written["DataValue"].tolist() == [223353.0, 231574.0]
