"""A module contains pipeline functions for parsing. It takes row data,
transforms and stores it into bronze-layer parquet files.

Also builds and saves each extract's DDI-derived JSON variable dictionary.
"""

from collections.abc import Collection
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
    bronze_columns_by_year,
    build_and_save_variable_dictionary,
    merge_variables_into_bronze,
    parse_to_bronze,
)
from src.schemas.bronze.ipums_long import modal_columns

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

    Returns:
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
                collection=collection,
            )
            continue
        if not _REQUIRED_ENTRY <= entry.keys():
            log.warning(
                "ipums_manifest_entry_skipped",
                reason="no_file_path",
                entry=str(entry)[:200],
                collection=collection,
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
            If true - bypass the coverage check entirely.

    Returns:
        bool:
            Whether to parse data or not (given coverage and parse request).
    """
    if force:
        return True
    if not sample_years:
        return True
    return not all(variables <= coverage.get(year, set()) for year in sample_years)


def _refusal_reason(
    entry_columns: set[str],
    existing_years: set[int],
    expected: frozenset[str],
    summary_known: bool,
    replace: bool,
) -> str | None:
    """Why a "new_samples" entry must not be written, or None to proceed.

    A "new_samples" entry is written whole, so against a year that already has
    bronze it replaces every column that year held. Two independent gates
    gate that:

      - `replace` covers overwriting at all, and is the operator's switch.
      - the column gate covers changing the shape, and is deliberately left
        armed under `replace`: restoring a year needs replace=True, and that
        same run still has to refuse the entry that damaged it.

    To widen the expected set on purpose, pass `expected_columns` explicitly.
    There is no flag for it: `force` travels in the manifest from the
    extractor, so reusing it would let one re-download disarm this for good.

    Args:
        entry_columns (set[str]):
            The columns this entry would write.
        existing_years (set[int]):
            The entry's years that already have a bronze file.
        expected (frozenset[str]):
            The column set every year is meant to hold.
        summary_known (bool):
            Whether the entry's columns came from its DDI codebook.
        replace (bool):
            Whether overwriting an existing year is permitted.

    Returns:
        str | None:
            A reason to log and skip, or None to write the entry.
    """
    if not existing_years:
        return None
    if not summary_known:
        # Columns fell back to the requested list, which omits the flag and
        # technical columns IPUMS adds. Unknown is not the same as safe.
        return "unknown_columns"
    if entry_columns - expected:
        return "unexpected_columns"
    if not replace:
        return "bronze_year_exists"
    return None


def parse_ipums_extracts(
    external_dir: Path,
    bronze_dir: Path,
    collection: str,
    dictionaries_dir: Path | None = None,
    replace: bool = False,
    expected_columns: Collection[str] | None = None,
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
            A path to dictionaries for data to parse. None uses
            data/reference/ipums/<collection>. Must match `collection` - a
            mismatch reads and writes another collection's dictionaries
            silently.
        replace (bool):
            Allow a "new_samples" entry to overwrite a year that already has
            a bronze file.
        expected_columns (Collection[str] | None):
            The column set every year is meant to hold, used to refuse an
            entry that would reshape an existing year. None derives it from
            the years already in bronze.

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
    # Read from the parquet files rather than the dictionaries: a dictionary
    # accumulates and never shrinks, so it keeps listing columns a year has
    # since lost and that year would never be reparsed.
    coverage = bronze_columns_by_year(bronze_dir, collection)
    # Frozen for the whole run, so an entry that shrinks a year partway
    # through cannot move the target the later entries are judged against.
    expected = (
        frozenset(expected_columns)
        if expected_columns is not None
        else modal_columns(coverage)
    )

    for entry in entries:
        metadata = entry["metadata"]
        data_path = Path(entry["file_path"])
        ddi_path = Path(metadata["ddi_path"])
        extract_id = metadata["extract_id"]
        request_kind = metadata.get("request_kind", "new_samples")
        requested = list(metadata["variables"])
        requested_set = set(requested)
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
            # A delta only ever adds columns, so it cannot reshape a year and
            # is not subject to the guard below.
            touched_paths = merge_variables_into_bronze(
                data_path,
                ddi_path,
                collection,
                bronze_dir,
                new_variables=entry_columns,
                force=force,
            )
        else:
            existing_years = sample_years & set(coverage)
            refusal = _refusal_reason(
                variables,
                existing_years,
                expected,
                summary is not None,
                replace,
            )
            if refusal is not None:
                log.warning(
                    "ipums_parse_entry_refused",
                    collection=collection,
                    extract_id=extract_id,
                    reason=refusal,
                    years=sorted(existing_years),
                    unexpected=sorted(variables - expected),
                )
                continue
            touched_paths = parse_to_bronze(
                data_path,
                ddi_path,
                collection,
                bronze_dir,
                replace=replace,
            )
        bronze_paths.extend(touched_paths)
        log.info(
            "ipums_parse_entry_complete",
            collection=collection,
            extract_id=extract_id,
            request_kind=request_kind,
            n_columns=len(entry_columns),
            flag_columns=[c for c in entry_columns if c not in requested_set],
        )

        touched_years = [int(p.stem) for p in touched_paths]
        build_and_save_variable_dictionary(
            ddi_path,
            dictionaries_dir,
            touched_years,
            force=force,
            # A delta merge only wrote its own columns, so the dictionary
            # must not claim the rest of the codebook.
            variables=entry_columns if request_kind == "variable_delta" else None,
        )
        for year in touched_years:
            if request_kind == "variable_delta":
                coverage.setdefault(year, set()).update(variables)
            else:
                # A wholesale rewrite replaces the year, so coverage has to
                # shrink with it. Union here would leave every delta that
                # once filled this year looking already-covered, and the
                # columns the rewrite dropped would never come back.
                coverage[year] = set(variables)

    log.info("ipums_parse_pipeline_complete", n_bronze_paths=len(bronze_paths))
    return bronze_paths
