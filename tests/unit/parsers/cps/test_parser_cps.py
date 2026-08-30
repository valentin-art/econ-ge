import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.parsers.cps.parser_cps import (
    CPSVariable,
    apply_value_labels,
    bronze_path,
    build_and_save_variable_dictionary,
    build_variable_dictionary,
    load_variable_dictionary,
    parse_cps_zip,
    parse_fixed_width,
    parse_sps_dictionary,
    parse_to_bronze,
    parse_value_labels,
    parse_variable_labels,
    save_variable_dictionary,
    variable_dictionary_path,
    variables_from_dictionary,
)

# Mirrors the real NBER dictionary syntax (confirmed against the actual
# cpsmw64_88.sps): `data list file=... /` block, plain "start-end" or a
# single "col" for one-column fields, terminated by a lone ".", with `(a)`
# marking alphanumeric fields and no marker meaning numeric. The `value
# labels` block mirrors the real file too: one variable's code/label pairs
# per `/`-separated group, terminated by a lone ".".
_SPS_TEXT = """\
*Change the input file location as needed.
input program.
data list file='c:\\cpsmw64.raw' /
            hhid       1-5
            state      6-7         (a)
            age        8-9
            wgt        10-17
.
variable labels
            hhid       "Household ID"
            state      "State"
            age        "Age"
            wgt        "Weight"
.
value labels
            state
                   1    "California"
                   2    "Texas"
            /age
                   99   "missing/invalid"
            .
"""

# hhid=12345, state="CA", age=34, wgt=00123.45
_DAT_TEXT = "12345CA3400123.45\n12346TX2500098.70\n"


def test_parse_sps_dictionary_extracts_variables() -> None:
    variables = parse_sps_dictionary(_SPS_TEXT)

    assert variables == [
        CPSVariable(name="hhid", start=1, end=5, numeric=True),
        CPSVariable(name="state", start=6, end=7, numeric=False),
        CPSVariable(name="age", start=8, end=9, numeric=True),
        CPSVariable(name="wgt", start=10, end=17, numeric=True),
    ]


def test_parse_sps_dictionary_single_column_field() -> None:
    sps_text = "data list file='x' /\n    flag 3\n.\n"
    variables = parse_sps_dictionary(sps_text)
    assert variables == [CPSVariable(name="flag", start=3, end=3, numeric=True)]


def test_parse_sps_dictionary_ignores_variable_labels_block() -> None:
    variables = parse_sps_dictionary(_SPS_TEXT)
    assert [v.name for v in variables] == ["hhid", "state", "age", "wgt"]


def test_parse_sps_dictionary_raises_on_no_variables() -> None:
    with pytest.raises(ValueError, match="No variable definitions"):
        parse_sps_dictionary("not a dictionary at all")


def test_parse_variable_labels_extracts_descriptions() -> None:
    descriptions = parse_variable_labels(_SPS_TEXT)

    assert descriptions == {
        "hhid": "Household ID",
        "state": "State",
        "age": "Age",
        "wgt": "Weight",
    }


def test_parse_variable_labels_returns_empty_dict_when_no_block() -> None:
    assert parse_variable_labels("data list file='x' /\n flag 3\n.\n") == {}


def test_parse_value_labels_extracts_code_to_label_maps() -> None:
    value_labels = parse_value_labels(_SPS_TEXT)

    assert value_labels == {
        "state": {"1": "California", "2": "Texas"},
        "age": {"99": "missing/invalid"},
    }


def test_parse_value_labels_returns_empty_dict_when_no_block() -> None:
    assert parse_value_labels("data list file='x' /\n flag 3\n.\n") == {}


def test_build_variable_dictionary_combines_positions_descriptions_and_values() -> None:
    variable_dictionary = build_variable_dictionary(_SPS_TEXT)

    assert variable_dictionary == {
        "hhid": {
            "start": 1,
            "end": 5,
            "numeric": True,
            "Description": "Household ID",
            "Values": {},
        },
        "state": {
            "start": 6,
            "end": 7,
            "numeric": False,
            "Description": "State",
            "Values": {"1": "California", "2": "Texas"},
        },
        "age": {
            "start": 8,
            "end": 9,
            "numeric": True,
            "Description": "Age",
            "Values": {"99": "missing/invalid"},
        },
        "wgt": {
            "start": 10,
            "end": 17,
            "numeric": True,
            "Description": "Weight",
            "Values": {},
        },
    }


def test_variables_from_dictionary_reconstructs_specs_sorted_by_start() -> None:
    variable_dictionary = {
        "wgt": {"start": 10, "end": 17, "numeric": True},
        "hhid": {"start": 1, "end": 5, "numeric": True},
        "state": {"start": 6, "end": 7, "numeric": False},
    }

    variables = variables_from_dictionary(variable_dictionary)

    assert variables == [
        CPSVariable(name="hhid", start=1, end=5, numeric=True),
        CPSVariable(name="state", start=6, end=7, numeric=False),
        CPSVariable(name="wgt", start=10, end=17, numeric=True),
    ]


def test_variable_dictionary_path_builds_year_json_path(tmp_path: Path) -> None:
    assert variable_dictionary_path(tmp_path, 1964) == tmp_path / "1964.json"


def test_variable_dictionary_path_includes_month(tmp_path: Path) -> None:
    assert variable_dictionary_path(tmp_path, 1991, 2) == tmp_path / "199102.json"


def test_save_and_load_variable_dictionary_round_trip(tmp_path: Path) -> None:
    variable_dictionary = {
        "state": {
            "start": 6,
            "end": 7,
            "numeric": False,
            "Description": "State",
            "Values": {"1": "California", "2": "Texas"},
        }
    }

    out_path = save_variable_dictionary(variable_dictionary, tmp_path, 1964)

    assert out_path == tmp_path / "1964.json"
    assert json.loads(out_path.read_text()) == variable_dictionary
    assert load_variable_dictionary(tmp_path, 1964) == variable_dictionary


def test_build_and_save_variable_dictionary_parses_sps_file_and_saves(
    tmp_path: Path,
) -> None:
    sps_path = tmp_path / "cpsmw64_88.sps"
    sps_path.write_text(_SPS_TEXT)
    dictionaries_dir = tmp_path / "dictionaries"

    out_path = build_and_save_variable_dictionary(
        sps_path, 1964, None, dictionaries_dir
    )

    assert out_path == dictionaries_dir / "1964.json"
    loaded = load_variable_dictionary(dictionaries_dir, 1964)
    assert loaded["state"] == {
        "start": 6,
        "end": 7,
        "numeric": False,
        "Description": "State",
        "Values": {"1": "California", "2": "Texas"},
    }


def test_apply_value_labels_replaces_codes_and_returns_new_frame() -> None:
    df = pd.DataFrame({"state": [1, 2, 1], "age": [34, 99, 25], "wgt": [1.0, 2.0, 3.0]})
    variable_dictionary = {
        "state": {"Description": "State", "Values": {"1": "California", "2": "Texas"}},
        "age": {"Description": "Age", "Values": {"99": "missing/invalid"}},
        "wgt": {"Description": "Weight", "Values": {}},
    }

    out = apply_value_labels(df, variable_dictionary)

    assert out["state"].tolist() == ["California", "Texas", "California"]
    assert out["age"].tolist() == [34, "missing/invalid", 25]
    assert out["wgt"].tolist() == [1.0, 2.0, 3.0]  # untouched, no dictionary entry
    assert df["state"].tolist() == [1, 2, 1]  # original frame untouched


def test_apply_value_labels_leaves_unmapped_codes_and_missing_values_as_is() -> None:
    df = pd.DataFrame({"state": [1, 3, None]})
    variable_dictionary = {
        "state": {"Description": "State", "Values": {"1": "California"}}
    }

    out = apply_value_labels(df, variable_dictionary)

    assert out["state"].iloc[0] == "California"
    assert out["state"].iloc[1] == 3.0  # code 3 not in dictionary — unchanged
    assert pd.isna(out["state"].iloc[2])


def test_apply_value_labels_handles_float_coerced_whole_number_codes() -> None:
    # A column with a missing value forces float64 (1.0 not 1) — must still match.
    df = pd.DataFrame({"state": [1.0, None, 2.0]})
    variable_dictionary = {
        "state": {
            "Description": "State",
            "Values": {"1": "California", "2": "Texas"},
        }
    }

    out = apply_value_labels(df, variable_dictionary)

    assert out["state"].iloc[0] == "California"
    assert pd.isna(out["state"].iloc[1])
    assert out["state"].iloc[2] == "Texas"


def test_parse_fixed_width_reads_columns_by_spec() -> None:
    variables = parse_sps_dictionary(_SPS_TEXT)

    df = parse_fixed_width(_DAT_TEXT, variables)

    assert list(df.columns) == ["hhid", "state", "age", "wgt"]
    assert df["hhid"].tolist() == [12345, 12346]
    assert df["state"].tolist() == ["CA", "TX"]
    assert df["age"].tolist() == [34, 25]
    assert df["wgt"].tolist() == [123.45, 98.70]


def _write_zip_and_dictionary(tmp_path: Path) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_path / "cpsmw64.zip"
    # NBER's real archives contain one arbitrarily-named member (e.g.
    # "cpsmw64", no extension) — mirror that instead of "*.dat" naming.
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("cpsmw64", _DAT_TEXT)
    variable_dictionary = build_variable_dictionary(_SPS_TEXT)
    return zip_path, variable_dictionary


def test_parse_cps_zip_returns_tidy_frame_with_year(tmp_path: Path) -> None:
    zip_path, variable_dictionary = _write_zip_and_dictionary(tmp_path)

    out = parse_cps_zip(zip_path, variable_dictionary, year=1964)

    assert list(out.columns) == ["Year", "hhid", "state", "age", "wgt"]
    assert (out["Year"] == 1964).all()
    assert out["hhid"].tolist() == [12345, 12346]


def test_bronze_path_builds_annual_parquet_path(tmp_path: Path) -> None:
    path = bronze_path(tmp_path, 1964, None)
    assert path == tmp_path / "1964.parquet"


def test_bronze_path_builds_monthly_nested_parquet_path(tmp_path: Path) -> None:
    path = bronze_path(tmp_path, 1991, 2)
    assert path == tmp_path / "1991" / "199102.parquet"


def test_parse_to_bronze_writes_parquet_for_one_period(tmp_path: Path) -> None:
    zip_path, variable_dictionary = _write_zip_and_dictionary(tmp_path / "external")
    bronze_dir = tmp_path / "bronze"

    out_path = parse_to_bronze(zip_path, variable_dictionary, 1964, None, bronze_dir)

    assert out_path == bronze_dir / "1964.parquet"
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert written["hhid"].tolist() == [12345, 12346]
