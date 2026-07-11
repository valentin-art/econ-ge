from pathlib import Path

from src.extractors.base import build_extraction_record
from src.extractors.manifest import append_to_manifest, read_manifest


def _record(tmp_path: Path, name: str):
    file_path = tmp_path / name
    file_path.write_bytes(b"raw data")
    return build_extraction_record(
        source="bea_api",
        extraction_id=name,
        file_path=file_path,
        metadata={"table": name},
    )


def test_read_manifest_returns_empty_list_when_missing(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) == []


def test_append_to_manifest_creates_and_appends(tmp_path: Path) -> None:
    record1 = _record(tmp_path, "FAAt201.parquet")
    append_to_manifest(tmp_path, record1)

    entries = read_manifest(tmp_path)
    assert len(entries) == 1
    assert entries[0]["extraction_id"] == "FAAt201.parquet"
    assert entries[0]["sha256"] == record1.sha256

    record2 = _record(tmp_path, "T11400.parquet")
    append_to_manifest(tmp_path, record2)

    entries = read_manifest(tmp_path)
    assert len(entries) == 2
    assert [e["extraction_id"] for e in entries] == [
        "FAAt201.parquet",
        "T11400.parquet",
    ]


def test_append_to_manifest_creates_missing_parent_dir(tmp_path: Path) -> None:
    nested_dir = tmp_path / "bea" / "FixedAssets"
    record = _record(tmp_path, "FAAt201.parquet")

    manifest_path = append_to_manifest(nested_dir, record)

    assert manifest_path.exists()
    assert read_manifest(nested_dir) != []
