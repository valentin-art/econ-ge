"""Read and append manifests of ExtractRecord."""

from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

import structlog
import yaml

from src.extractors.base import ExtractionRecord

log = structlog.get_logger(__name__)

MANIFEST_FILENAME = "_MANIFEST.yaml"


def as_name_list(value: object) -> list[str] | None:
    """A list of names, or None if the value is any other shape."""
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def read_manifest(source_dir: Path) -> list[dict[str, Any]]:
    """Read source_dir/_MANIFEST.yaml; returns [] if it doesn't exist yet."""
    manifest_path = source_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return []
    return yaml.safe_load(manifest_path.read_text()) or []


def iter_valid_entries(
    source_dir: Path,
    required_entry_keys: Collection[str] = (),
    required_metadata_keys: Collection[str] = (),
    entries: Collection[dict[str, Any]] | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield (entry, metadata) for each well-formed manifest entry, warning once
    per skipped entry. `entries` reuses an already-read manifest.
    """
    rows = list(entries) if entries is not None else read_manifest(source_dir)
    for entry in rows:
        metadata = entry.get("metadata") if isinstance(entry, dict) else None
        if not isinstance(metadata, dict):
            log.warning(
                "manifest_entry_skipped",
                reason="metadata_not_a_mapping",
                source_dir=str(source_dir),
                entry=str(entry)[:200],
            )
        elif not set(required_metadata_keys) <= metadata.keys():
            log.warning(
                "manifest_entry_skipped",
                reason="missing_metadata_keys",
                source_dir=str(source_dir),
                entry=str(entry)[:200],
            )
        elif not set(required_entry_keys) <= entry.keys():
            log.warning(
                "manifest_entry_skipped",
                reason="missing_entry_keys",
                source_dir=str(source_dir),
                entry=str(entry)[:200],
            )
        else:
            yield entry, metadata


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
