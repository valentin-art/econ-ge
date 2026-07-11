import json
from pathlib import Path

import pytest

from src.extractors.bea_json import read_bea_results


def _write_json(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "FAAt201.json"
    path.write_text(json.dumps(payload))
    return path


def test_read_bea_results_returns_results_node(tmp_path: Path) -> None:
    payload = {
        "BEAAPI": {
            "Request": {"RequestParam": []},
            "Results": {"Data": [{"LineNumber": "1"}], "Notes": []},
        }
    }
    path = _write_json(tmp_path, payload)

    results = read_bea_results(path)

    assert results["Data"] == [{"LineNumber": "1"}]


def test_read_bea_results_unwraps_single_item_results_list(tmp_path: Path) -> None:
    payload = {
        "BEAAPI": {
            "Request": {"RequestParam": []},
            "Results": [{"Data": [{"LineNumber": "1"}]}],
        }
    }
    path = _write_json(tmp_path, payload)

    results = read_bea_results(path)

    assert results["Data"] == [{"LineNumber": "1"}]


def test_read_bea_results_raises_on_top_level_error(tmp_path: Path) -> None:
    payload = {"BEAAPI": {"Error": {"APIErrorDescription": "bad request"}}}
    path = _write_json(tmp_path, payload)

    with pytest.raises(RuntimeError, match="BEA API error"):
        read_bea_results(path)


def test_read_bea_results_raises_on_results_level_error(tmp_path: Path) -> None:
    payload = {
        "BEAAPI": {
            "Request": {"RequestParam": []},
            "Results": {"Error": {"APIErrorDescription": "no data"}},
        }
    }
    path = _write_json(tmp_path, payload)

    with pytest.raises(RuntimeError, match="BEA API error"):
        read_bea_results(path)
