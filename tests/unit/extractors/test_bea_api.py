import json
from pathlib import Path

import pytest

from extractors.bea.bea_api import BEAExtractor
from src.extractors.manifest import read_manifest


def _fake_raw_json(table: str = "FAAt201", n_rows: int = 2) -> str:
    payload = {
        "BEAAPI": {
            "Request": {"RequestParam": []},
            "Results": {
                "Data": [
                    {
                        "TableName": table,
                        "LineNumber": "1",
                        "LineDescription": "Private fixed assets",
                        "TimePeriod": "1975",
                        "DataValue": "223,353",
                    },
                    {
                        "TableName": table,
                        "LineNumber": "1",
                        "LineDescription": "Private fixed assets",
                        "TimePeriod": "1976",
                        "DataValue": "231,574",
                    },
                ][:n_rows],
                "Notes": [
                    {
                        "NoteRef": table.upper(),
                        "NoteText": (
                            "Table 2.1. Current-Cost Net Stock - LastRevised: "
                            "June 25, 2026"
                        ),
                    }
                ],
            },
        }
    }
    return json.dumps(payload)


def test_extract_writes_json_and_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.extractors.bea.bea_api.beaapi.api_request",
        lambda *args, **kwargs: _fake_raw_json("FAAt201"),
    )
    extractor = BEAExtractor(api_key="fake-key", storage_dir=tmp_path)

    record = extractor.extract(dataset="FixedAssets", table="FAAt201")

    assert record.file_path == tmp_path / "FixedAssets" / "FAAt201.json"
    assert record.file_path.exists()
    assert record.metadata == {
        "dataset": "FixedAssets",
        "table": "FAAt201",
        "n_rows": 2,
        "release_date": "2026-06-25",
    }
    assert json.loads(record.file_path.read_text())["BEAAPI"]["Results"]["Data"]

    manifest_entries = read_manifest(tmp_path)
    assert len(manifest_entries) == 1
    assert manifest_entries[0]["extraction_id"] == record.extraction_id
    assert manifest_entries[0]["sha256"] == record.sha256


def test_extract_passes_frequency_for_nipa_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_api_request(beaspec, *args, **kwargs):
        calls.append(beaspec)
        return _fake_raw_json("T11400")

    monkeypatch.setattr(
        "src.extractors.bea.bea_api.beaapi.api_request", fake_api_request
    )
    extractor = BEAExtractor(api_key="fake-key", storage_dir=tmp_path)

    extractor.extract(dataset="FixedAssets", table="FAAt201")
    extractor.extract(dataset="NIPA", table="T11400")

    assert "Frequency" not in calls[0]
    assert calls[1]["Frequency"] == "A"


def test_extract_raises_on_bea_api_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error_payload = json.dumps(
        {"BEAAPI": {"Error": {"APIErrorDescription": "Invalid table name"}}}
    )
    monkeypatch.setattr(
        "src.extractors.bea.bea_api.beaapi.api_request",
        lambda *args, **kwargs: error_payload,
    )
    extractor = BEAExtractor(api_key="fake-key", storage_dir=tmp_path)

    with pytest.raises(RuntimeError, match="BEA API error"):
        extractor.extract(dataset="FixedAssets", table="BadTable")
