import json
from pathlib import Path

import pytest

from src.parsers.dictionaries import get_variable_info


def _write_dictionary(
    dictionaries_root: Path, source: str, year: int, content: dict
) -> None:
    source_dir = dictionaries_root / source
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"{source}_{year}.json").write_text(json.dumps(content))


def test_get_variable_info_returns_description_and_values(tmp_path: Path) -> None:
    _write_dictionary(
        tmp_path,
        "cpsmw",
        1964,
        {"adc": {"Description": "ADC Recipiency", "Values": {"1": "yes", "2": "no"}}},
    )

    info = get_variable_info("cpsmw", "adc", tmp_path)

    assert info == {"Description": "ADC Recipiency", "Values": {"1": "yes", "2": "no"}}


def test_get_variable_info_checks_multiple_year_files(tmp_path: Path) -> None:
    _write_dictionary(
        tmp_path, "cpsmw", 1964, {"hhid": {"Description": "ID", "Values": {}}}
    )
    _write_dictionary(
        tmp_path, "cpsmw", 1989, {"newvar": {"Description": "New", "Values": {}}}
    )

    info = get_variable_info("cpsmw", "newvar", tmp_path)

    assert info == {"Description": "New", "Values": {}}


def test_get_variable_info_raises_when_source_missing(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="No dictionaries found for source 'ipums'"):
        get_variable_info("ipums", "adc", tmp_path)


def test_get_variable_info_raises_when_variable_missing(tmp_path: Path) -> None:
    _write_dictionary(
        tmp_path, "cpsmw", 1964, {"hhid": {"Description": "ID", "Values": {}}}
    )

    with pytest.raises(KeyError, match="'nonexistent' not found"):
        get_variable_info("cpsmw", "nonexistent", tmp_path)
