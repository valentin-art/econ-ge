"""A module contains pipeline functions for parsing. It takes takes row data,
transforms and stores it into bronze-layer parquet files.

Also builds and saves each extract's DDI-derived JSON variable dictionary.
"""

from pathlib import Path

import structlog

from src.config.settings import settings
from src.extractors.ipums_coverage import parse_sample_year
from src.extractors.ipums_ddi import (
    merge_column_names,
    summary_from_metadata,
    try_summarize_ddi,
)
from src.extractors.manifest import read_manifest
from src.parsers.ipums import (
    bronze_coverage,
    build_and_save_variable_dictionary,
    merge_variables_into_bronze,
    parse_to_bronze,
)

log = structlog.get_logger(__name__)

_REQUIRED_METADATA = frozenset({"samples", "variables", "ddi_path", "extract_id"})
_REQUIRED_ENTRY = frozenset({"file_path"})


def _collection_manifest_entries(external_dir: Path, collection: str) -> list[dict]:
    """Manifest entries for `collection` whose data file and DDI codebook still
    exist on disk.

    "new_samples" entries ordered before "variable_delta" ones (a delta merge needs
    an existing bronze file to merge into) while preserving relative extraction
    order within each group. A missing request_kind (manifest entries written before
    incremental extraction existed) is treated as "new_samples".

    NOTE: A hand-edited/truncated manifest can carry a malformed entry - missing
    keys entirely, or `metadata:` empty/scalar. Skipped with a warning rather
    than raised, so one bad entry does not abort parsing for the whole collection.

    Args:
        external_dir (Path):
            A path containing raw data.
        collection (str):
            A collection name.

    Return:
        list[dict]:
            A list of manifest entries corresponding to raw data for given collection.
    """
    collection_dir = external_dir / collection
    entries = []
    for entry in read_manifest(collection_dir):
        metadata = entry.get("metadata") if isinstance(entry, dict) else None
        if not isinstance(metadata, dict) or not _REQUIRED_METADATA <= metadata.keys():
            log.warning(
                "ipums_manifest_entry_skipped",
                reason="missing_required_metadata_keys",
                entry=str(entry)[:200],
            )
            continue
        if not _REQUIRED_ENTRY <= entry.keys():
            log.warning(
                "ipums_manifest_entry_skipped",
                reason="no_file_path",
                entry=str(entry)[:200],
            )
            continue
        data_path = Path(entry["file_path"])
        ddi_path = Path(metadata["ddi_path"])
        if data_path.exists() and ddi_path.exists():
            entries.append(entry)
    entries.sort(
        key=lambda e: e["metadata"].get("request_kind", "new_samples") != "new_samples"
    )
    return entries


def _entry_needs_processing(
    coverage: dict[int, set[str]],
    sample_years: set[int],
    variables: set[str],
    force: bool = False,
) -> bool:
    """Checks if a manifest entry needs to be processed (parsed/merged).

    False only when every entry's year is already in `coverage` with all
    `variables` already present - i.e. bronze already fully reflects this
    entry. It's safe to skip re-parsing/re-merging it.

    If no year could be parsed from any of the entry's sample ids, always
    process (fail safe rather than silently skip on an ambiguous entry).

    Args:
        coverage (dict[int, set[str]]):
            Data coverage dictionary (year, <set of variables>)
        sample_years (set[int]):
            A set of sample years that need to be parsed.
        variables (set[str]):
            A set of variables that need to be parsed.
        force (bool):
            Should the processing be forcely parsed.

    Returns:
        bool:
            Whether to parse data or not (given coverage and parse request).
    """
    if force:
        return True
    if not sample_years:
        return True
    return not all(variables <= coverage.get(year, set()) for year in sample_years)


def parse_ipums_extracts(
    external_dir: Path,
    bronze_dir: Path,
    collection: str,
    dictionaries_dir: Path | None = None,
) -> list[Path]:
    """Parse every already-downloaded IPUMS extract not yet reflected in bronze.

    Walks every manifest entry for `collection` still backed by files on disk.

    Args:
        external_dir (Path):
            The path to raw data that needs to be parsed.
        bronze_dir (Path):
            The path to bronze data that has already been parsed before.
        collection (str):
            The IPUMS collection to parse (e.g. "cps").
        dictionaries_dir (Path | None):

    Returns:
        list[Path]:
            A list of paths with bronze data for a given collection.

    Raises:
        RuntimeError:
            No usable manifest entry for `collection` is backed by files on
            disk. Nothing is written.
    """
    log.info("ipums_parse_pipeline_start", collection=collection)
    if dictionaries_dir is None:
        dictionaries_dir = settings.paths.ipums_clean_dictionaries_dir(collection)

    bronze_paths: list[Path] = []
    entries = _collection_manifest_entries(external_dir, collection)
    if not entries:
        raise RuntimeError(
            f"No downloaded IPUMS extract found for collection "
            f"{collection!r} in {external_dir / collection} - run "
            f"extract_ipums_extracts first"
        )
    coverage = bronze_coverage(dictionaries_dir)

    for entry in entries:
        metadata = entry["metadata"]
        data_path = Path(entry["file_path"])
        ddi_path = Path(metadata["ddi_path"])
        extract_id = metadata["extract_id"]
        request_kind = metadata.get("request_kind", "new_samples")
        requested = list(metadata["variables"])
        # The columns this entry actually contributes to bronze.
        #
        # A "new_samples" pull is written whole by parse_to_bronze, so that
        # is every column in the file.
        # A "variable_delta" merge keeps only its own columns - which must
        # include the flag columns IPUMS attached to the requested variables,
        # or they are silently dropped even though they are in raw files.
        summary = try_summarize_ddi(ddi_path) or summary_from_metadata(metadata)
        if request_kind == "variable_delta":
            entry_columns = merge_column_names(summary, requested)
        else:
            entry_columns = (
                list(summary.variables) if summary is not None else requested
            )
        variables = set(entry_columns)
        sample_years = {
            year
            for sample in metadata["samples"]
            if (year := parse_sample_year(sample)) is not None
        }

        force = metadata.get("force", False)
        if not _entry_needs_processing(coverage, sample_years, variables, force=force):
            log.info(
                "ipums_parse_entry_already_covered",
                collection=collection,
                extract_id=extract_id,
            )
            continue

        if request_kind == "variable_delta":
            touched_paths = merge_variables_into_bronze(
                data_path,
                ddi_path,
                collection,
                bronze_dir,
                new_variables=entry_columns,
                force=force,
            )
        else:
            touched_paths = parse_to_bronze(data_path, ddi_path, collection, bronze_dir)
        bronze_paths.extend(touched_paths)
        log.info(
            "ipums_parse_entry_complete",
            collection=collection,
            extract_id=extract_id,
            request_kind=request_kind,
            n_columns=len(entry_columns),
            flag_columns=[c for c in entry_columns if c not in set(requested)],
        )

        touched_years = [int(p.stem) for p in touched_paths]
        build_and_save_variable_dictionary(
            ddi_path,
            dictionaries_dir,
            touched_years,
            force=force,
            # A delta merge only wrote its own columns, so the dictionary
            # must not claim the rest of the codebook - bronze_coverage
            # reads these files as the record of what bronze holds.
            variables=entry_columns if request_kind == "variable_delta" else None,
        )
        for year in touched_years:
            coverage.setdefault(year, set()).update(variables)
    log.info("ipums_parse_pipeline_complete", n_bronze_paths=len(bronze_paths))
    return bronze_paths
