"""CPS Basic and Mare-Winship extractors: NBER -> raw ZIP file on disk
(settings.paths.cps_external_dir("basic"|"mw")).

NBER's Mare-Winship extracts and CPS Basic are a fixed historical archive.

The module provides following functionality:
- Downloads a CPS extracts (cpsb{year}{month}_dat.zip or cpsmw{YY}.zip)
- Downloads SPS-files (as dictionaries) that describe its fixed-width layout
- Stores files into storage_dir/ and torage_dir/dictionaries/ as-is
- Files are not pulled from external resources if they already present in
  `storage_dir/`.
"""

import time
from abc import abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import requests
import structlog

from src.config.settings import settings
from src.config.sources import cps_basic_sps_filename, cps_mw_sps_filename
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


class CPSExtractor(Extractor):
    """General machinery to download/extract NBER files and dictionaries."""

    _source: str

    def __init__(self, base_url: str, storage_dir: Path) -> None:
        self.base_url = base_url
        self.storage_dir = storage_dir
        self.dictionaries_dir = storage_dir / "dictionaries"

    def _build_url(self, filename: str, year: int | None, month: int | None) -> str:
        return f"{self.base_url.rstrip('/')}/{filename}"

    @abstractmethod
    def _zip_filename(self, year: int, month: int | None) -> str: ...

    @abstractmethod
    def _sps_filename(self, year: int, month: int | None) -> str: ...

    def _download(
        self, filename: str, year: int | None = None, month: int | None = None
    ) -> tuple[bytes, str]:
        url = self._build_url(filename, year, month)
        for attempt in range(1, _MAX_RETRIES + 1):
            response = requests.get(url, headers=_HEADERS, timeout=60)
            if response.status_code == 403 and attempt < _MAX_RETRIES:
                log.warning(
                    f"{self._source}_download_403_retrying",
                    filename=filename,
                    attempt=attempt,
                )
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
            response.raise_for_status()
            return response.content, url
        raise RuntimeError(f"unreachable: retry loop exited without return for {url}")

    def _ensure_local(
        self,
        path: Path,
        filename: str,
        year: int | None = None,
        month: int | None = None,
    ) -> str | None:
        """Download `filename` to `path` unless it's already on disk.

        Returns the source URL if downloaded, None if served from the local
        cache (cheaper than fetching, and NBER's archives don't change).
        """
        if path.exists():
            log.info(
                f"{self._source}_file_cached", filename=filename, file_path=str(path)
            )
            return None
        content, url = self._download(filename, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return url

    def extract(self, year: int, month: int | None = None) -> ExtractionRecord:
        """Pull the zip extract and its covering SPS dictionary for `year`.

        CPS Basic additionally scopes this to `month`; Mare-Winship is
        annual, so `month` stays None and is omitted from the resulting
        metadata/log fields entirely (rather than appearing as `None`).

        Skips the download for either file if it's already present locally.

        Parameters
        ----------
        year : four-digit year, e.g. 1964
        month : two-digit month, e.g. 1 (CPS Basic only)
        """
        zip_filename = self._zip_filename(year, month)
        sps_filename = self._sps_filename(year, month)
        fields: dict[str, int] = {"year": year}
        if month is not None:
            fields["month"] = month

        log.info(f"{self._source}_extract_start", zip_filename=zip_filename, **fields)

        zip_path = self.storage_dir / zip_filename
        zip_url = self._ensure_local(zip_path, zip_filename, year, month)

        sps_path = self.dictionaries_dir / sps_filename
        self._ensure_local(sps_path, sps_filename, year, month)

        metadata = {
            **fields,
            "zip_filename": zip_filename,
            "sps_filename": sps_filename,
            "zip_url": zip_url,
            "sps_path": str(sps_path),
            "cached": zip_url is None,
        }
        month_suffix = f"_{month}" if month is not None else ""
        extraction_id = (
            f"{self._source}_{year}{month_suffix}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        record = build_extraction_record(
            source=self._source,
            extraction_id=extraction_id,
            file_path=zip_path,
            metadata=metadata,
        )
        append_to_manifest(self.storage_dir, record)
        log.info(
            f"{self._source}_extract_complete",
            file_path=str(zip_path),
            sps_path=str(sps_path),
            **fields,
        )
        return record


class CPSBasicExtractor(CPSExtractor):
    """Downloads NBER CPS Basic zip/SPS files and persists them as-is.

    base_url/storage_dir/sps_base_url default to settings but can be
    injected - keeps this testable/composable without going through the
    settings singleton.
    """

    _source = "basic"

    def __init__(
        self,
        base_url: str | None = None,
        storage_dir: Path | None = None,
        sps_base_url: str | None = None,
    ) -> None:
        super().__init__(
            base_url=(
                base_url if base_url is not None else settings.cps_basic_base_url
            ),
            storage_dir=(
                storage_dir
                if storage_dir is not None
                else settings.paths.cps_external_dir("basic")
            ),
        )
        self.sps_base_url = (
            sps_base_url
            if sps_base_url is not None
            else settings.cps_basic_sps_base_url
        )

    def _build_url(self, filename: str, year: int | None, month: int | None) -> str:
        if filename.endswith(".sps"):
            return f"{self.sps_base_url.rstrip('/')}/{filename}"
        return f"{self.base_url.rstrip('/')}/{year}/{filename}"

    def _zip_filename(self, year: int, month: int | None) -> str:
        if month is None:
            raise ValueError("CPSBasicExtractor.extract requires month")
        return f"cpsb{year}{month:02d}_dat.zip"

    def _sps_filename(self, year: int, month: int | None) -> str:
        if month is None:
            raise ValueError("CPSBasicExtractor.extract requires month")
        return cps_basic_sps_filename(year, month)


class CPSMWExtractor(CPSExtractor):
    """Downloads NBER Mare-Winship CPS zip/SPS files and persists them as-is.

    base_url/storage_dir default to settings but can be injected - keeps this
    testable/composable without going through the settings singleton.
    """

    _source = "mw"

    def __init__(
        self, base_url: str | None = None, storage_dir: Path | None = None
    ) -> None:
        super().__init__(
            base_url=base_url if base_url is not None else settings.cps_mw_base_url,
            storage_dir=(
                storage_dir
                if storage_dir is not None
                else settings.paths.cps_external_dir("mw")
            ),
        )

    def _zip_filename(self, year: int, month: int | None) -> str:
        return f"cpsmw{year % 100:02d}.zip"

    def _sps_filename(self, year: int, month: int | None) -> str:
        return cps_mw_sps_filename(year)
