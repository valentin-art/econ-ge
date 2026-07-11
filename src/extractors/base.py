"""Base extractor protocol:
 - what was downloaded
 - when
 - where
 - checksum

Extractors don't parse or transform data. They only save the raw
pull as-is and hand back a record of what they saved.
"""

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class ExtractionRecord(BaseModel):
    """What was downloaded, when, size, checksum - one manifest entry."""

    source: str
    extraction_id: str
    extracted_at: datetime
    file_path: Path
    size_bytes: int
    sha256: str
    metadata: dict


class Extractor(ABC):
    @abstractmethod
    def extract(self, **params) -> ExtractionRecord: ...


def build_extraction_record(
    source: str,
    extraction_id: str,
    file_path: Path,
    metadata: dict,
) -> ExtractionRecord:
    """Compute size/checksum for a just-written file and build its ExtractionRecord."""
    data = file_path.read_bytes()
    return ExtractionRecord(
        source=source,
        extraction_id=extraction_id,
        extracted_at=datetime.now(timezone.utc),
        file_path=file_path,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        metadata=metadata,
    )
