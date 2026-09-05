"""Bronze-layer parsing of raw BEA extractor JSON into a tidy long DataFrame,
persisted source-by-source as bronze parquet.

 - Reads the as-saved JSON from extractors.bea_api.BEAExtractor (still has BEA's
original TimePeriod/DataValue formatting)
 - Produces the (LineNumber, LineDescription, Year, DataValue) shape that
 the rest of the pipeline uses.
 - Kept separate from extraction so it's testable on small in-memory
DataFrames/JSON payloads without hitting the network
 - Re-runnable without re-downloading if a parsing bug turns up.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from extractors.bea.bea_json import read_bea_results
from src.input_output.parquet import write_parquet
from src.schemas.bronze.bea_long import validate_bea_long


def _clean_datavalue(series: pd.Series) -> pd.Series:
    """Strip commas, convert to float, coerce blanks to NaN."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .replace("", np.nan)
        .astype(float)
    )


def parse_bea_table(raw: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    """Parse a raw BEA extractor DataFrame (FixedAssets or NIPA) into tidy long form.

    Columns out: LineNumber (int), LineDescription (str), Year (int), DataValue (float).
    All values in millions USD (UNIT_MULT = 6) for the tables this pipeline uses.
    """
    df = raw.copy()
    df["DataValue"] = _clean_datavalue(df["DataValue"])
    df["Year"] = df["TimePeriod"].astype(int)
    df = df[df["Year"].between(year_start, year_end)].copy()
    df["LineNumber"] = df["LineNumber"].astype(int)
    df = df[["LineNumber", "LineDescription", "Year", "DataValue"]]
    return validate_bea_long(df)


def load_bea_json(json_path: Path) -> pd.DataFrame:
    """Load a raw BEA API JSON response into the beaapi-shaped raw DataFrame:
    all-string columns, BEA's native TimePeriod/DataValue formatting."""
    results = read_bea_results(json_path)
    return pd.DataFrame(results["Data"], dtype="string")


def parse_bea_json(json_path: Path, year_start: int, year_end: int) -> pd.DataFrame:
    """load_bea_json + parse_bea_table: the external(JSON) -> tidy long path."""
    return parse_bea_table(load_bea_json(json_path), year_start, year_end)


def bronze_path(bronze_dir: Path, dataset: str, table: str) -> Path:
    """A path to bronze parquet: {bronze_dir}/{dataset}/{table}.parquet."""
    return bronze_dir / dataset / f"{table}.parquet"


def parse_to_bronze(
    json_path: Path,
    dataset: str,
    table: str,
    bronze_dir: Path,
    year_start: int,
    year_end: int,
) -> Path:
    """Parse one external JSON source file and persist it as its own bronze parquet.

    One JSON file in -> one parquet file out (source-by-source): a parsing bug
    or re-run for one table doesn't touch the other tables' bronze files.
    """
    tidy = parse_bea_json(json_path, year_start, year_end)
    out_path = bronze_path(bronze_dir, dataset, table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(tidy, out_path)
    return out_path
