"""Append-only manifest of ExtractionRecords for one external-source directory."""

from pathlib import Path

import yaml

from src.extractors.base import ExtractionRecord

MANIFEST_FILENAME = "_MANIFEST.yaml"


def read_manifest(source_dir: Path) -> list[dict]:
    """Read source_dir/_MANIFEST.yaml; returns [] if it doesn't exist yet."""
    manifest_path = source_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return []
    return yaml.safe_load(manifest_path.read_text()) or []


def append_to_manifest(source_dir: Path, record: ExtractionRecord) -> Path:
    """Append one ExtractionRecord to source_dir/_MANIFEST.yaml, creating it if needed."""
    manifest_path = source_dir / MANIFEST_FILENAME
    entries = read_manifest(source_dir)
    entries.append(
        {
            "extraction_id": record.extraction_id,
            "source": record.source,
            "extracted_at": record.extracted_at.isoformat(),
            "file_path": str(record.file_path),
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "metadata": record.metadata,
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(entries, sort_keys=False))
    return manifest_path
