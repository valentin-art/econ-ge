"""Core orchestration: CPS Mare-Winship extraction + bronze parsing.

Runs extractors.cps_mw.CPSMWExtractor (NBER -> external zip/SPS, skipping the
download for files already cached under data/external/cpsmw/) followed by
parsers.cps_mw.parse_to_bronze (zip/SPS -> tidy bronze parquet) and
parsers.cps_mw.build_and_save_variable_dictionary (SPS -> parsers/dictionaries/
cpsmw/cpsmw_{year}.json, variable descriptions + value labels) for every year
in sources.CPS_MW_YEARS, and concatenates the per-year bronze frames into one
panel. No file I/O beyond what the extractor/parser already do, no
settings-singleton dependency beyond what CPSMWExtractor defaults to — usable
from a notebook, a CLI job, or a test with a mocked extractor.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

from src.config import sources
from src.config.settings import settings
from src.extractors.cps_mw import CPSMWExtractor
from src.input_output.parquet import read_parquet
from src.parsers.cps_mw import build_and_save_variable_dictionary, parse_to_bronze

log = structlog.get_logger(__name__)


@dataclass
class CPSMWPipelineResult:
    cps_mw: pd.DataFrame  # all extracted years, concatenated


def _extract_and_parse(
    extractor: CPSMWExtractor, bronze_dir: Path, year: int
) -> pd.DataFrame:
    """extract() the raw zip/SPS to external, parse+persist to bronze, read it back.

    Also (re)builds the year's variable-dictionary JSON from the same SPS
    file — cheap pure-text parsing, so it's not worth caching/skipping like
    the network download is.
    """
    record = extractor.extract(year=year)
    sps_path = Path(record.metadata["sps_path"])
    build_and_save_variable_dictionary(sps_path, year)
    bronze_file = parse_to_bronze(record.file_path, sps_path, year, bronze_dir)
    return read_parquet(bronze_file)


def run_cpsmw_pipeline(years: list[int] | None = None) -> CPSMWPipelineResult:
    """Run the CPS Mare-Winship extract-and-parse pipeline for `years`.

    Parameters
    ----------
    years : years to pull, defaults to sources.CPS_MW_YEARS
    """
    years = years if years is not None else sources.CPS_MW_YEARS
    log.info("cpsmw_pipeline_start", years=years)

    extractor = CPSMWExtractor()
    bronze_dir = settings.paths.bronze / "cps"
    frames = [_extract_and_parse(extractor, bronze_dir, year) for year in years]
    cps_mw = pd.concat(frames, ignore_index=True)

    log.info("cpsmw_pipeline_complete", n_years=len(years), n_rows=len(cps_mw))
    return CPSMWPipelineResult(cps_mw=cps_mw)
