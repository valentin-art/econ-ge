"""Bronze stage: raw BEA JSON (external) -> tidy bronze parquet (data/bronze/bea/).

Companion to `pipelines.bea_extract_pipeline` (BEA API -> external) and
`pipelines.bea_silver_pipeline` (bronze -> silver) — the three stages
`jobs/extract_bea.py`, `jobs/parse_bea_bronze.py`, and
`jobs/transform_bea_silver.py` each thinly wrap. Pure function of
already-downloaded external JSON — no network. `pipelines.bea_pipeline` (the
fused full-ETL pipeline) also calls `parse_bea_tables` directly rather than
re-implementing the per-table parsing loop.
"""

from pathlib import Path

import structlog

from src.config import sources
from src.parsers.bea.parser_bea import parse_to_bronze

log = structlog.get_logger(__name__)


def parse_bea_tables(
    external_dir: Path,
    bronze_dir: Path,
    tables: list[tuple[str, str]] | None = None,
    year_start: int = sources.YEAR_START,
    year_end: int = sources.YEAR_END,
) -> list[Path]:
    """Parse every (dataset, table)'s already-downloaded external JSON into bronze parquet.

    Parameters
    ----------
    external_dir : the `bea` external root, e.g. settings.paths.external / "bea"
                   (containing {dataset}/{table}.json files written by
                   extract_bea_tables/BEAExtractor)
    bronze_dir    : the `bea` bronze root, e.g. settings.paths.bronze / "bea"
    tables        : (dataset, table) pairs to parse; defaults to sources.BEA_TABLES
    year_start, year_end : sample window passed through to parse_to_bronze
    """
    tables = tables if tables is not None else sources.BEA_TABLES

    log.info("bea_parse_pipeline_start", n_tables=len(tables))
    bronze_paths = [
        parse_to_bronze(
            external_dir / dataset / f"{table}.json",
            dataset,
            table,
            bronze_dir,
            year_start,
            year_end,
        )
        for dataset, table in tables
    ]
    log.info("bea_parse_pipeline_complete", n_tables=len(bronze_paths))
    return bronze_paths
