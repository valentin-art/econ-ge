"""Bronze-layer parsing of raw IPUMS extractor files (.dat.gz + DDI .xml) into
a tidy DataFrame, persisted as bronze parquet, plus conversion of the DDI
codebook into a cleaned JSON variable dictionary (mirrors parsers.cps's
{variable: {start, end, numeric, Description, Values}} shape).
"""

import json
import tempfile
from pathlib import Path
from typing import Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ipumspy import readers
from ipumspy.ddi import Codebook

from src.input_output.parquet import write_parquet
from src.schemas.bronze.ipums_long import (
    check_no_duplicate_columns,
    validate_ipums_long,
)

_NUMERIC_VARTYPES = {"numeric", "integer", "float"}


def build_variable_dictionary(ddi_codebook: Codebook) -> dict[str, dict[str, object]]:
    """Convert a parsed DDI codebook into the JSON shape persisted by
    save_variable_dictionary:

        {variable_name: {"start": int, "end": int, "numeric": bool,
                          "Description": str, "Values": {code: label}}}
    """
    return {
        v.name: {
            "start": v.start,
            "end": v.end,
            "numeric": v.vartype in _NUMERIC_VARTYPES,
            "Description": v.label,
            "Values": {str(code): label for label, code in v.codes.items()},
        }
        for v in ddi_codebook.data_description
    }


def variable_dictionary_path(dictionaries_dir: Path, year: int) -> Path:
    """Path to store a year's dictionary: {dictionaries_dir}/{year}.json.

    dictionaries_dir is already collection-scoped
    (settings.paths.ipums_clean_dictionaries_dir(collection)), so no
    collection/extract_id in the filename - matches parsers.cps's own
    {year}{month}.json convention, and keeps dictionary_lookup's plain
    *.json glob working unchanged.
    """
    return dictionaries_dir / f"{year}.json"


def save_variable_dictionary(
    variable_dictionary: dict[str, dict[str, object]],
    dictionaries_dir: Path,
    year: int,
) -> Path:
    """Union `variable_dictionary` onto whatever's already saved for `year`
    and save - new variable keys are added, already-known keys are kept -
    rather than overwriting. A single year accumulates variable definitions
    across multiple extracts over time (an initial "new_samples" pull, then
    zero or more later "variable_delta" merges), the same way
    merge_variables_into_bronze accumulates bronze columns rather than
    overwriting the file wholesale.
    """
    out_path = variable_dictionary_path(dictionaries_dir, year)
    existing = (
        load_variable_dictionary(dictionaries_dir, year) if out_path.exists() else {}
    )
    merged = {**existing, **variable_dictionary}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_variable_dictionary(
    dictionaries_dir: Path, year: int
) -> dict[str, dict[str, object]]:
    """Loads a JSON-dictionary."""
    path = variable_dictionary_path(dictionaries_dir, year)
    result = json.loads(path.read_text(encoding="utf-8"))
    return result


def build_and_save_variable_dictionary(
    ddi_path: Path,
    dictionaries_dir: Path,
    years: list[int],
) -> list[Path]:
    """Build once from the DDI, save (merge) into every year in `years`.

    `years` should be the years actually just written/updated in bronze -
    the stems of parse_to_bronze's/merge_variables_into_bronze's own return
    value - ground truth from the data itself, not sample-name parsing.
    """
    ddi_codebook = readers.read_ipums_ddi(ddi_path)
    variable_dictionary = build_variable_dictionary(ddi_codebook)
    return [
        save_variable_dictionary(variable_dictionary, dictionaries_dir, year)
        for year in years
    ]


def bronze_coverage(dictionaries_dir: Path) -> dict[int, set[str]]:
    """{year: set(variable_names)} already reflected in bronze, read straight
    off dictionaries_dir/*.json's top-level keys.

    This is the "reference dir as source of truth for what's in bronze"
    check: a year's dictionary file is only ever written/updated at the same
    time parse_to_bronze/merge_variables_into_bronze touch that year, so its
    keys are exactly the columns bronze has for that year. Non-numeric
    filenames (e.g. leftover legacy {collection}_{extract_id}.json files)
    are ignored.
    """
    coverage: dict[int, set[str]] = {}
    for json_path in dictionaries_dir.glob("*.json"):
        try:
            year = int(json_path.stem)
        except ValueError:
            continue
        coverage[year] = set(json.loads(json_path.read_text(encoding="utf-8")))
    return coverage


def parse_ipums_extract(
    data_path: Path, ddi_path: Path, chunksize: int = 100_000
) -> pd.DataFrame:
    """Parse one raw IPUMS extract (.dat.gz + DDI .xml) into a tidy DataFrame."""
    ddi_codebook = readers.read_ipums_ddi(ddi_path)
    iter_microdata = readers.read_microdata_chunked(
        ddi_codebook, data_path, chunksize=chunksize
    )
    df = pd.concat([df for df in iter_microdata])
    return validate_ipums_long(df)


def bronze_path(bronze_dir: Path, collection: str, year: int) -> Path:
    """Location of bronze data files: {bronze_dir}/{collection}/{year}.parquet."""
    return bronze_dir / collection / f"{year}.parquet"


def parse_to_bronze(
    data_path: Path,
    ddi_path: Path,
    collection: str,
    bronze_dir: Path,
    chunksize: int = 100_000,
) -> list[Path]:
    """Stream-parse one raw IPUMS extract straight to bronze parquet, split by
    YEAR, without ever holding the full extract in memory.

    A chunk may span several years, and a single year's rows may arrive
    across several non-contiguous chunks, so a ParquetWriter is opened
    lazily per year on first sight of that year and kept open (accumulating
    row groups) until every chunk has been processed.
    """
    ddi_codebook = readers.read_ipums_ddi(ddi_path)
    iter_microdata = readers.read_microdata_chunked(
        ddi_codebook, data_path, chunksize=chunksize
    )

    writers: dict[int, pq.ParquetWriter] = {}
    out_paths: dict[int, Tuple[Path, Path]] = {}
    total_rows = 0
    try:
        for chunk in iter_microdata:
            check_no_duplicate_columns(chunk)
            if chunk.empty:
                continue
            total_rows += len(chunk)
            for year, year_df in chunk.groupby("YEAR"):
                year = int(year)  # type: ignore[arg-type]
                table = pa.Table.from_pandas(year_df, preserve_index=False)
                if year not in writers:
                    out_path = bronze_path(bronze_dir, collection, year)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = out_path.with_suffix(".tmp.parquet")
                    writers[year] = pq.ParquetWriter(tmp_path, table.schema)
                    out_paths[year] = (tmp_path, out_path)
                writers[year].write_table(table)
    finally:
        errors = []
        for writer in writers.values():
            try:
                writer.close()
            except Exception as e:
                errors.append(e)
        if errors:
            raise RuntimeError(
                f"Failed to close {len(errors)} ParquetWriter(s) - "
                f"check for partial .tmp.parquet files in {bronze_dir}"
            ) from errors[0]
        for year, (tmp_path, out_path) in out_paths.items():
            tmp_path.rename(out_path)

    if total_rows == 0:
        raise ValueError("IPUMS extract has no rows")

    return [out_paths[year][1] for year in sorted(out_paths)]


def merge_variables_into_bronze(
    data_path: Path,
    ddi_path: Path,
    collection: str,
    bronze_dir: Path,
    new_variables: list[str],
    merge_keys: tuple[str, ...] = ("YEAR", "MONTH", "SERIAL", "PERNUM"),
    chunksize: int = 1_000_000,
) -> list[Path]:
    """Merge a variable-delta extract (new_variables pulled for samples whose
    other variables are already in bronze) into the existing per-year bronze
    parquet files, joining on merge_keys.

    Stages the delta extract through parse_to_bronze (reusing its chunked,
    split-by-YEAR writer so this stays memory-bounded too) into a temp
    directory, then for each staged year: keeps only merge_keys +
    new_variables (dropping any other columns IPUMS auto-included, e.g.
    technical/weight variables that weren't actually requested, so they don't
    collide with what's already in bronze), left-joins onto the existing
    bronze file for that year, and overwrites it.

    Raises RuntimeError if a staged year has no existing bronze file to merge
    into - that means the request this delta was planned from doesn't
    actually correspond to a year already parsed to bronze, which points at
    a coverage-tracking bug rather than something to silently paper over.
    """
    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        staged_paths = parse_to_bronze(
            data_path,
            ddi_path,
            collection,
            bronze_dir=staging_dir,
            chunksize=chunksize,
        )

        updated_paths: list[Path] = []
        for staged_path in staged_paths:
            year = int(staged_path.stem)
            out_path = bronze_path(bronze_dir, collection, year)
            if not out_path.exists():
                raise RuntimeError(
                    f"Cannot merge variable-delta extract into missing bronze "
                    f"file {out_path} for year {year} - expected it to already "
                    f"exist from a prior new_samples extract"
                )
            staged_df = pd.read_parquet(staged_path)
            existing_df = pd.read_parquet(out_path)
            merge_columns = [k for k in merge_keys if k in staged_df.columns]
            if not merge_columns:
                raise RuntimeError(
                    f"Staged variable-delta extract for year {year} has no "
                    f"columns in common with merge_keys {merge_keys} - cannot "
                    f"merge into existing bronze file {out_path}"
                )
            add_columns = [
                v
                for v in new_variables
                if v in staged_df.columns and v not in existing_df.columns
            ]
            merged = existing_df.merge(
                staged_df[merge_columns + add_columns], on=merge_columns, how="left"
            )
            if len(merged) != len(existing_df):
                raise RuntimeError(
                    f"Merging variable-delta extract for year {year} into "
                    f"existing bronze file {out_path} changed the number of "
                    f"rows from {len(existing_df)} to {len(merged)} - this "
                    f"should never happen, check merge_keys {merge_keys} and "
                    f"the staged extract's columns"
                )
            write_parquet(merged, out_path)
            updated_paths.append(out_path)

    return updated_paths
