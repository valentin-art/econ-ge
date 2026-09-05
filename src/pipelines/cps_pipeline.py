"""A collection of one-liners which do basic ETL for CPS data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

from extractors.cps.cps import CPSBasicExtractor, CPSExtractor, CPSMWExtractor
from src.config import sources
from src.config.settings import settings
from src.config.sources import CPSSource
from src.input_output.parquet import read_parquet
from src.parsers.cps.parser_cps import (
    build_and_save_variable_dictionary,
    load_variable_dictionary,
    parse_to_bronze,
)

log = structlog.get_logger(__name__)


@dataclass
class CPSPipelineResult:
    frame: pd.DataFrame  # all extracted periods, concatenated


def _extract_and_parse(
    source: CPSSource,
    extractor: CPSExtractor,
    year: int,
    month: int | None,
) -> pd.DataFrame:
    """ETL block that processes one-period CPS data (year or year-month)."""
    record = extractor.extract(year=year, month=month)
    sps_path = Path(record.metadata["sps_path"])
    dictionaries_dir = settings.paths.cps_clean_dictionaries_dir(source)
    build_and_save_variable_dictionary(sps_path, year, month, dictionaries_dir)
    variable_dictionary = load_variable_dictionary(dictionaries_dir, year, month)

    bronze_dir = settings.paths.cps_bronze_dir(source)
    bronze_file = parse_to_bronze(
        record.file_path, variable_dictionary, year, month, bronze_dir
    )
    return read_parquet(bronze_file)


def _run_cps_pipeline(
    source: CPSSource,
    extractor: CPSExtractor,
    periods: list[tuple[int, int | None]],
) -> pd.DataFrame:
    """ETL block that pricesses multiple CPS periods."""
    frames = [
        _extract_and_parse(source, extractor, year, month) for year, month in periods
    ]
    return pd.concat(frames, ignore_index=True)


def run_cpsmw_pipeline(years: list[int] | None = None) -> CPSPipelineResult:
    """Run the CPS Mare-Winship extract-and-parse pipeline for `years`.

    Parameters
    ----------
    years : years to pull, defaults to sources.CPS_MW_YEARS
    """
    years = years if years is not None else sources.CPS_MW_YEARS
    log.info("cps_mw_pipeline_start", years=years)

    periods: list[tuple[int, int | None]] = [(year, None) for year in years]
    frame = _run_cps_pipeline("mw", CPSMWExtractor(), periods)

    log.info("cps_mw_pipeline_complete", n_periods=len(periods), n_rows=len(frame))
    return CPSPipelineResult(frame=frame)


def run_cps_basic_pipeline(
    periods: list[tuple[int, int]] | None = None,
) -> CPSPipelineResult:
    """Run the CPS Basic extract-and-parse pipeline for `periods`.

    Parameters
    ----------
    periods : (year, month) pairs to pull, defaults to sources.CPS_BASIC_PERIODS
    """
    periods = periods if periods is not None else sources.CPS_BASIC_PERIODS
    log.info("cps_basic_pipeline_start", periods=periods)

    frame = _run_cps_pipeline("basic", CPSBasicExtractor(), list(periods))

    log.info("cps_basic_pipeline_complete", n_periods=len(periods), n_rows=len(frame))
    return CPSPipelineResult(frame=frame)
