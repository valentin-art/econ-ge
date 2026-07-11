"""CPS Mare-Winship extractor: NBER -> raw ZIP file on disk
(settings.paths.external/cpsmw/...).

Downloads a Mare-Winship March CPS extract (cpsmw{YY}.zip) into
storage_dir/, and the SPS dictionary that describes its fixed-width layout
into storage_dir/dictionaries/ — saving both as-is, no unpacking, no column
parsing, that's parsers.cps_mw's job.

Unlike BEA's tables (revised between vintages, so bea_api.BEAExtractor always
re-downloads), NBER's Mare-Winship extracts are a fixed historical archive —
cpsmw64.zip published in 2000 never changes. So extract() skips the network
call entirely for any file already present in storage_dir, checking the zip
and the SPS dictionary independently (a dictionary covering a multi-year
range may already be on disk from a previous year's extract() call).

NBER's data.nber.org sits behind Akamai bot management, which 403s bare
requests (no User-Agent) outright and intermittently 403s even browser-UA
`requests` calls (empirically: a plain `curl` is blocked every time
regardless of headers sent — TLS/HTTP2 fingerprint, not header content — while
Python's `requests`/`httpx` usually succeed but occasionally still get a
transient 403). extract() always sends a browser User-Agent and retries
403s with backoff before giving up — only relevant when a file must actually
be downloaded.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import structlog

from src.config.settings import settings
from src.config.sources import cps_mw_sps_filename
from src.extractors.base import ExtractionRecord, Extractor, build_extraction_record
from src.extractors.manifest import append_to_manifest

log = structlog.get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
_MAX_RETRIES = 4
_RETRY_BACKOFF_SECONDS = 20.0


class CPSMWExtractor(Extractor):
    """Downloads NBER Mare-Winship CPS zip/SPS files and persists them as-is.

    base_url/storage_dir default to settings but can be injected — keeps this
    testable/composable without going through the settings singleton.
    """

    def __init__(
        self, base_url: str | None = None, storage_dir: Path | None = None
    ) -> None:
        self.base_url = base_url if base_url is not None else settings.cps_mw_base_url
        self.storage_dir = (
            storage_dir
            if storage_dir is not None
            else settings.paths.external / "cpsmw"
        )
        self.dictionaries_dir = self.storage_dir / "dictionaries"

    def _download(self, filename: str) -> tuple[bytes, str]:
        url = f"{self.base_url}/{filename}"
        for attempt in range(1, _MAX_RETRIES + 1):
            response = requests.get(url, headers=_HEADERS, timeout=60)
            if response.status_code == 403 and attempt < _MAX_RETRIES:
                log.warning(
                    "cps_mw_download_403_retrying",
                    filename=filename,
                    attempt=attempt,
                )
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
            response.raise_for_status()
            return response.content, url
        raise RuntimeError(f"unreachable: retry loop exited without return for {url}")

    def _ensure_local(self, path: Path, filename: str) -> str | None:
        """Download `filename` to `path` unless it's already on disk.

        Returns the source URL if downloaded, None if served from the local
        cache (cheaper than fetching, and NBER's archives don't change).
        """
        if path.exists():
            log.info("cps_mw_file_cached", filename=filename, file_path=str(path))
            return None
        content, url = self._download(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return url

    def extract(self, year: int) -> ExtractionRecord:
        """Pull the cpsmw{YY}.zip extract and its covering SPS dictionary for `year`.

        Skips the download for either file if it's already present locally.

        Parameters
        ----------
        year : four-digit year, e.g. 1964
        """
        yy = f"{year % 100:02d}"
        zip_filename = f"cpsmw{yy}.zip"
        sps_filename = cps_mw_sps_filename(year)
        log.info("cps_mw_extract_start", year=year, zip_filename=zip_filename)

        zip_path = self.storage_dir / zip_filename
        zip_url = self._ensure_local(zip_path, zip_filename)

        sps_path = self.dictionaries_dir / sps_filename
        self._ensure_local(sps_path, sps_filename)

        metadata = {
            "year": year,
            "zip_filename": zip_filename,
            "sps_filename": sps_filename,
            "zip_url": zip_url,
            "sps_path": str(sps_path),
            "cached": zip_url is None,
        }
        extraction_id = f"cps_mw_{year}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        record = build_extraction_record(
            source="cps_mw",
            extraction_id=extraction_id,
            file_path=zip_path,
            metadata=metadata,
        )
        append_to_manifest(self.storage_dir, record)
        log.info(
            "cps_mw_extract_complete",
            year=year,
            file_path=str(zip_path),
            sps_path=str(sps_path),
        )
        return record
