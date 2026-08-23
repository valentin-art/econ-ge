"""Bronze stage: raw IPUMS extract (external) -> tidy bronze parquet (data/bronze/ipums/).

Companion to `pipelines.ipums_extract_pipeline` (IPUMS API -> external). Pure
function of already-downloaded external .dat.gz/.xml files - no network.
Also builds and saves each extract's DDI-derived JSON variable dictionary
(data/reference/ipums/{collection}/{year}.json), mirroring
`pipelines.bea_parse_pipeline`'s per-item parsing loop over `config.sources`.
"""

from pathlib import Path

import structlog

from src.config import sources
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


def _collection_manifest_entries(external_dir: Path, collection: str) -> list[dict]:
    """Manifest entries for `collection` whose data file and DDI codebook still
    exist on disk, with "new_samples" entries ordered before "variable_delta"
    ones (a delta merge needs an existing bronze file to merge into) while
    preserving relative extraction order within each group. A missing
    request_kind (manifest entries written before incremental extraction
    existed) is treated as "new_samples".
    """
    collection_dir = external_dir / collection
    entries = []
    for entry in read_manifest(collection_dir):
        metadata = entry["metadata"]
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
    """False only when every one of the entry's years is already in `coverage`
    with every one of `variables` present - i.e. bronze (per the
    reference-dir dictionaries) already fully reflects this entry, so it's
    safe to skip re-parsing/re-merging it. If no year could be parsed from
    any of the entry's sample ids, always process (fail safe rather than
    silently skip on an ambiguous entry). force=True always returns True -
    a deliberate forced refresh must never be skipped just because the
    dictionaries already show it as covered, since that's usually exactly
    why it was forced.
    """
    if force:
        return True
    if not sample_years:
        return True
    return not all(variables <= coverage.get(year, set()) for year in sample_years)


def parse_ipums_extracts(
    external_dir: Path,
    bronze_dir: Path,
    extracts: list[sources.IPUMSExtractRequest] | None = None,
) -> list[Path]:
    """Parse every already-downloaded IPUMS extract not yet reflected in bronze.

    For each distinct collection among `extracts`, walks every manifest entry
    still backed by files on disk. extractors.ipums_api.IPUMSExtractor writes
    one manifest entry per extract_incremental() submission, not one per
    request - a single request can span several manifest entries once prior
    coverage only needs a partial top-up. data/reference/ipums/{collection}/
    (one JSON dictionary per year) is the source of truth for what's actually
    already in bronze - the manifest can list more extracts than were ever
    parsed, and an entry whose years/variables are already fully covered
    there is skipped rather than reprocessed:
      - "new_samples" entries build+save their DDI-derived JSON variable
        dictionary and are parsed into bronze the normal way (one parquet
        file per YEAR).
      - "variable_delta" entries (extra variables pulled for samples/years
        already in bronze) are instead merged onto the existing per-year
        bronze files, via parsers.ipums.merge_variables_into_bronze.

    Parameters
    ----------
    external_dir : the `ipums` external root, e.g. settings.paths.external / "ipums"
    bronze_dir    : the `ipums` bronze root, e.g. settings.paths.bronze / "ipums"
    extracts      : requests naming which collections to parse; defaults to
                    sources.IPUMS_EXTRACTS. Only .collection is used - every
                    on-disk extract for that collection is considered, not
                    just the ones matching the request's own .samples/.variables.
    """
    extracts = extracts if extracts is not None else sources.IPUMS_EXTRACTS
    collections = dict.fromkeys(req.collection for req in extracts)

    log.info("ipums_parse_pipeline_start", n_collections=len(collections))
    bronze_paths: list[Path] = []
    for collection in collections:
        entries = _collection_manifest_entries(external_dir, collection)
        if not entries:
            raise RuntimeError(
                f"No downloaded IPUMS extract found for collection "
                f"{collection!r} in {external_dir / collection} - run "
                f"extract_ipums_extracts first"
            )
        dictionaries_dir = settings.paths.ipums_clean_dictionaries_dir(collection)
        coverage = bronze_coverage(dictionaries_dir)

        for entry in entries:
            metadata = entry["metadata"]
            data_path = Path(entry["file_path"])
            ddi_path = Path(metadata["ddi_path"])
            extract_id = metadata["extract_id"]
            request_kind = metadata.get("request_kind", "new_samples")
            requested = list(metadata["variables"])
            # The columns this entry actually contributes to bronze. A
            # "new_samples" pull is written whole by parse_to_bronze, so that
            # is every column in the file. A "variable_delta" merge keeps only
            # its own columns - which must include the flag columns IPUMS
            # attached to the requested variables, or they are silently
            # dropped even though they are sitting in the .dat.gz.
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
            if not _entry_needs_processing(
                coverage, sample_years, variables, force=force
            ):
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
                touched_paths = parse_to_bronze(
                    data_path, ddi_path, collection, bronze_dir
                )
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
