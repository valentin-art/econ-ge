import hashlib
from pathlib import Path

from src.extractors.base import build_extraction_record


def test_build_extraction_record_computes_size_and_checksum(tmp_path: Path) -> None:
    file_path = tmp_path / "raw.parquet"
    file_path.write_bytes(b"some raw bytes")

    record = build_extraction_record(
        source="bea_api",
        extraction_id="FixedAssets_FAAt201_20260101T000000Z",
        file_path=file_path,
        metadata={"dataset": "FixedAssets", "table": "FAAt201"},
    )

    assert record.source == "bea_api"
    assert record.file_path == file_path
    assert record.size_bytes == len(b"some raw bytes")
    assert record.sha256 == hashlib.sha256(b"some raw bytes").hexdigest()
    assert record.metadata == {"dataset": "FixedAssets", "table": "FAAt201"}


def test_build_extraction_record_checksum_changes_with_content(tmp_path: Path) -> None:
    file_a = tmp_path / "a.parquet"
    file_b = tmp_path / "b.parquet"
    file_a.write_bytes(b"content A")
    file_b.write_bytes(b"content B")

    record_a = build_extraction_record("bea_api", "id-a", file_a, {})
    record_b = build_extraction_record("bea_api", "id-b", file_b, {})

    assert record_a.sha256 != record_b.sha256
