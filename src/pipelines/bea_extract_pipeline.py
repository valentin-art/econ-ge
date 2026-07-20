"""External stage: BEA API -> raw JSON on disk (data/external/bea/).

Companion to `pipelines.bea_parse_pipeline` (external -> bronze) and
`pipelines.bea_silver_pipeline` (bronze -> silver) — the three stages
`jobs/extract_bea.py`, `jobs/parse_bea_bronze.py`, and
`jobs/transform_bea_silver.py` each thinly wrap. `pipelines.bea_pipeline`
(the fused full-ETL pipeline) also calls `extract_bea_tables` directly
rather than re-implementing the per-table extraction loop.
"""

import structlog

from src.config import sources
from src.extractors.base import ExtractionRecord
from src.extractors.bea_api import BEAExtractor

log = structlog.get_logger(__name__)


def extract_bea_tables(
    beakey: str, tables: list[tuple[str, str]] | None = None
) -> list[ExtractionRecord]:
    """Download every (dataset, table) pair via the BEA API.

    Saves each table's raw JSON response as-is to data/external/bea/{dataset}/
    {table}.json (via extractors.bea_api.BEAExtractor) and appends a record to
    that directory's _MANIFEST.yaml. Always re-downloads (BEA revises
    published tables between vintages), never skips on a cache hit.

    Parameters
    ----------
    beakey : BEA API registration key (settings.bea_api_key)
    tables : (dataset, table) pairs to pull; defaults to sources.BEA_TABLES
    """
    if beakey == "":
        raise RuntimeError("BEA_API_KEY is empty - cannot extract BEA tables")
    tables = tables if tables is not None else sources.BEA_TABLES

    log.info("bea_extract_pipeline_start", n_tables=len(tables))
    extractor = BEAExtractor(api_key=beakey)
    records = [
        extractor.extract(dataset=dataset, table=table) for dataset, table in tables
    ]
    log.info("bea_extract_pipeline_complete", n_tables=len(records))
    return records
