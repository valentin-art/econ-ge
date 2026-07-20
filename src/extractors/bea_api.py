"""BEA API extractor: external -> raw JSON file on disk.

Downloads a FixedAssets/NIPA table and saves BEA's raw JSON response as-is.
Every extract() re-downloads and re-saves rather than skipping on a
cache hit: BEA revises published tables between vintages, so silently
reusing a stale file would be worse than the extra network call.
"""

import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import beaapi
import pandas as pd
import structlog

from src.config.settings import settings
from src.extractors.base import (
    ExtractionRecord,
    Extractor,
    build_extraction_record,
)
from src.extractors.bea_json import read_bea_results
from src.extractors.manifest import append_to_manifest

log = structlog.get_logger(__name__)

# BEA's NIPA backend intermittently 503s on otherwise-valid requests (observed:
# identical request succeeds seconds later, unrelated to request shape) --
# beaapi's own throttle=True only backs off for HTTP 429, not generic 5xx.
_RETRYABLE_STATUS_CODES = {502, 503, 504}
_MAX_ATTEMPTS = 4
_RETRY_SLEEP_SECONDS = 5


def _api_request_with_retry(beaspec: dict) -> str:
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return beaapi.api_request(beaspec, as_string=True, throttle=True)  # type: ignore
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_STATUS_CODES or attempt == _MAX_ATTEMPTS:
                raise
            log.warning(
                "bea_extract_retry",
                status_code=e.code,
                attempt=attempt,
                max_attempts=_MAX_ATTEMPTS,
            )
            time.sleep(_RETRY_SLEEP_SECONDS)
    raise RuntimeError("unreachable")


def _find_release_date(notes: list[dict] | None, table: str) -> str | None:
    """Pull the 'LastRevised' date BEA embeds in the table's title note, e.g.
    "Table 2.1. ... - LastRevised: September 26, 2025" under NoteRef == table."""
    if not notes:
        return None
    for note in notes:
        if note.get("NoteRef", "").upper() == table.upper():
            text = note.get("NoteText", "")
            if " - LastRevised: " in text:
                _, _, date_str = text.rpartition(" - LastRevised: ")
                return datetime.strptime(date_str, "%B %d, %Y").date().isoformat()
    return None


class BEAExtractor(Extractor):
    """Downloads BEA FixedAssets/NIPA tables and persists BEA's raw JSON response."""

    def __init__(
        self, api_key: str | None = None, storage_dir: Path | None = None
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.bea_api_key
        self.storage_dir = (
            storage_dir if storage_dir is not None else settings.paths.external / "bea"
        )

    def extract(self, dataset: str, table: str) -> ExtractionRecord:
        """Pull `dataset`/`table` (all years) from the BEA API and save the raw JSON as-is.

        Parameters
        ----------
        dataset : BEA dataset name, e.g. "FixedAssets" or "NIPA"
        table   : BEA TableName, e.g. "FAAt201" or "T11400"
        """
        log.info("bea_extract_start", dataset=dataset, table=table)
        frequency_kwargs = {"Frequency": "A"} if dataset == "NIPA" else {}
        beaspec = {
            "UserID": self.api_key,
            "method": "GetData",
            "datasetname": dataset.lower(),
            "TableName": table,
            "Year": "X",
            "ResultFormat": "json",
            **frequency_kwargs,
        }
        raw_json = _api_request_with_retry(beaspec)

        file_path = self.storage_dir / dataset / f"{table}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(raw_json, encoding="utf-8")  # type: ignore

        results = read_bea_results(file_path)
        metadata = {
            "dataset": dataset,
            "table": table,
            "n_rows": len(results["Data"]),
            "release_date": _find_release_date(results.get("Notes"), table),
        }

        extraction_id = f"{dataset}_{table}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        record = build_extraction_record(
            source="bea_api",
            extraction_id=extraction_id,
            file_path=file_path,
            metadata=metadata,
        )
        append_to_manifest(self.storage_dir, record)
        log.info(
            "bea_extract_complete",
            dataset=dataset,
            table=table,
            n_rows=metadata["n_rows"],
            file_path=str(file_path),
        )
        return record


def verify_unit_scale(
    beakey: str,
    nipa_table: str,
    faa_table: str,
    year: int = 2017,
) -> float:
    """Return 10^(nipa_unit_mult - faa_unit_mult); warn if != 1.

    Call once after fetching NIPA and FAA tables to confirm units are
    compatible. Both T11400/T11600 and FAAt201 are UNIT_MULT=6 (millions USD)
    in current BEA vintages, so scale should be 1.0. A tiny single-year
    diagnostic lookup, not a dataset pull - doesn't go through the extractor.
    """
    nipa_unit = int(
        beaapi.get_data(
            beakey,
            datasetname="NIPA",
            TableName=nipa_table,
            Frequency="A",
            Year=str(year),
        )["UNIT_MULT"].iloc[0]
    )
    faa_unit = int(
        beaapi.get_data(
            beakey, datasetname="FixedAssets", TableName=faa_table, Year=str(year)
        )["UNIT_MULT"].iloc[0]
    )
    scale = 10 ** (nipa_unit - faa_unit)
    if scale != 1:
        log.warning(
            "bea_unit_mismatch",
            nipa_unit_mult=nipa_unit,
            faa_unit_mult=faa_unit,
            scale_factor=scale,
        )
    return float(scale)


def discover_table_names(
    beakey: str,
    dataset: str,
    keyword: str = "",
) -> pd.DataFrame:
    """List valid TableName values for a BEA dataset, optionally filtered.

    Examples:
        discover_table_names(beakey, "FixedAssets", "net stock")
        discover_table_names(beakey, "NIPA", "nonfarm")
    """
    params = beaapi.get_parameter_values(beakey, dataset, "TableName")
    if keyword:
        mask = params["Description"].str.contains(keyword, case=False, na=False)
        params = params[mask]
    return params
