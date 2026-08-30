"""IPUMS API extractor: external -> raw .dat.gz + DDI .xml codebook on disk.

Submits a microdata extract request, waits for it to complete, and downloads
the data file and its DDI codebook as-is.
"""

from collections.abc import Collection, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import structlog
from ipumspy import IpumsApiClient, MicrodataExtract
from ipumspy.api.exceptions import BadIpumsApiRequest

from src.config.settings import settings
from src.extractors.base import (
    ExtractionRecord,
    Extractor,
    build_extraction_record,
)
from src.extractors.ipums_coverage import (
    RequestKind,
    build_coverage,
    plan_delta_requests,
    save_coverage,
)
from src.extractors.ipums_ddi import (
    FLAG_PARSER_VERSION,
    collection_flag_registry,
    try_summarize_ddi,
)
from src.extractors.manifest import (
    append_to_manifest,
    iter_valid_entries,
    read_manifest,
)

log = structlog.get_logger(__name__)

# Metadata keys find_matching_extract reads without a fallback. A frozenset at
# module scope rather than a tuple rebuilt on every manifest entry.
_REQUIRED_METADATA = frozenset({"samples", "variables", "ddi_path", "extract_id"})


class CachedExtract(NamedTuple):
    """Contains information about cached data extract.

    size_bytes and sha256 come from manifest, not recomputed. Both are None if
    the entry recorded neither usably - the file is still reused, and the
    checksum is recomputed from disk."""

    data_path: Path
    ddi_path: Path
    extract_id: int
    size_bytes: int | None
    sha256: str | None


def _default_data_structure() -> dict[str, dict[str, str]]:
    return {"rectangular": {"on": "P"}}


def _recorded_checksum(entry: dict[str, Any]) -> tuple[int, str] | tuple[None, None]:
    """Size and checksum an entry recorded, or (None, None) if neither is usable.

    (None, None) is not a reason to discard a match: re-reading one local file
    is cheaper than the extract quota a re-download would spend.
    """
    sha256 = entry.get("sha256")
    # str() would turn a null/absent checksum into the literal "None", so the
    # shape is checked rather than coerced. int() does raise, so it is not.
    if isinstance(sha256, str) and sha256:
        try:
            return int(entry["size_bytes"]), sha256
        except (KeyError, TypeError, ValueError):
            pass
    log.info(
        "ipums_manifest_checksum_recomputed",
        reason="unusable_size_or_checksum",
        entry=str(entry)[:200],
    )
    return (None, None)


def _validate_requested_variables(
    collection_dir: Path,
    variables: Sequence[str],
    manifest_entries: Collection[dict[str, Any]] | None = None,
    allow_flag_variables: bool = False,
) -> None:
    """Raise ValueError if any requested variable is a flag column.

    Since IPUMS flag columns cannot be requested by name - they are attached to
    a requested variable, the function aims to avoid sending API requests which
    would be rejected with HTTP 400: ("Invalid mnemonic: ..."). The function
    rejects a name if any codebook on the disk labels it a flag
    (extractors.ipums_ddi.collection_flag_registry).

    Args:
        collection_dir (Path):
            Path to collection's dir containing _MANIFEST.yaml.
        variables (Sequence[str]):
            IPUMS variable names, e.g. ["AGE", "SEX"].
        manifest_entries (Collection[dict[str, Any]] | None):
            Already-read _MANIFEST.yaml entries, to save a second read. None
            re-reads `collection_dir`.
        allow_flag_variables (bool):
            Submit anyway, logging instead of raising. The verdict comes from
            a label heuristic over every codebook in the directory, so a
            reworded label or a hand-copied .xml can make a legitimate
            variable permanently unrequestable - this is the way out that
            does not involve deleting files.

    Returns:
        None.

    Raises:
        ValueError, if any variable is identified as a flag and
        `allow_flag_variables` is False.
    """
    registry = collection_flag_registry(collection_dir, manifest_entries)
    if not registry:
        return

    problems = []
    for variable in variables:
        kind = registry.kind_of(variable)
        if kind is None:
            continue
        sources = ", ".join(registry.sources_of(variable))
        if kind == "quality":
            problems.append(
                f"{variable} is the IPUMS data quality flag for {sources}, not "
                f"a requestable variable. Request {sources} instead and pass "
                f"data_quality_flags=True - the flag column is then added "
                f"automatically. Dropping {variable} without requesting "
                f"{sources} yields an extract with no flag column at all."
            )
        else:
            problems.append(
                f"{variable} is the IPUMS topcode flag for {sources}, not a "
                f"requestable variable. Drop it from `variables`; it is "
                f"delivered automatically alongside {sources} regardless of "
                f"data_quality_flags."
            )
    if not problems:
        return
    log.warning(
        "ipums_flag_variable_requested",
        collection_dir=str(collection_dir),
        # the flagged names, not the whole request - the whole request is what
        # the caller already has, the verdict is what it does not
        flagged=sorted(v for v in variables if registry.kind_of(v) is not None),
        allowed=allow_flag_variables,
    )
    if allow_flag_variables:
        return
    raise ValueError(
        "\n".join(problems)
        + "\n\nIf this is wrong - a mislabeled or hand-copied codebook - "
        "re-run with allow_flag_variables=True to submit the request anyway."
    )


def find_matching_extract(
    collection_dir: Path,
    samples: Sequence[str],
    variables: Sequence[str],
    data_structure: dict[str, dict[str, str]],
    data_quality_flags: bool,
    manifest_entries: Collection[dict[str, Any]] | None = None,
) -> CachedExtract | None:
    """Compares the requested (samples, variables) against the manifest entries
    (as function arguments) in given collection.

    Searches the most recent manifest entry in `collection_dir` such that:
    - Samples exactly match `samples`
    - Variables are a superset of `variables`
    - Data file and DDI codebook both still exist on disk
    - The data_structure and data_quality_flags must also match

    Args:
        collection_dir (Path):
            Path to the collection's dir containing _MANIFEST.yaml.
        samples (Sequence[str]):
            IPUMS sample IDs, e.g. ["cps2006_09s"].
        variables (Sequence[str]):
            IPUMS variable names, e.g. ["AGE", "SEX"].
        data_structure (dict[str, dict[str, str]]):
            A form of data pulled: Hierarchical or rectangular data.
        data_quality_flags (bool):
            Whether to pull IPUMS Data quality flags.
        manifest_entries (Collection[dict[str, Any]] | None):
            Already-read _MANIFEST.yaml entries, to save a second read. None
            re-reads `collection_dir`.

    Returns:
        CachedExtract for the most recent matching entry; or None if no entry
        matches.
    """
    requested_samples = set(samples)
    requested_variables = set(variables)
    default_structure = _default_data_structure()
    match = None

    valid_entries = iter_valid_entries(
        collection_dir,
        required_entry_keys=("file_path",),
        required_metadata_keys=_REQUIRED_METADATA,
        entries=manifest_entries,
    )

    for entry, metadata in valid_entries:
        if set(metadata["samples"]) != requested_samples:
            continue
        if not requested_variables <= set(metadata["variables"]):
            continue
        # Entries predating these keys were all pulled with today's defaults
        # (rectangular-on-P, flags on) - verified against the cps manifest.
        if metadata.get("data_structure", default_structure) != data_structure:
            continue

        if metadata.get("data_quality_flags", True) != data_quality_flags:
            continue

        data_path = Path(entry["file_path"])
        ddi_path = Path(metadata["ddi_path"])

        if not data_path.exists() or not ddi_path.exists():
            continue

        try:
            extract_id = int(metadata["extract_id"])
        except (KeyError, TypeError, ValueError):
            log.warning(
                "ipums_manifest_entry_skipped",
                reason="unusable_extract_id",
                entry=str(entry)[:200],
            )
            continue

        size_bytes, sha256 = _recorded_checksum(entry)

        match = CachedExtract(
            data_path=data_path,
            ddi_path=ddi_path,
            extract_id=extract_id,
            size_bytes=size_bytes,
            sha256=sha256,
        )

    return match


class IPUMSExtractor(Extractor):
    """Downloads microdata extracts from the IPUMS API and persists them as-is."""

    def __init__(
        self,
        api_key: str | None = None,
        storage_dir: Path | None = None,
        client: IpumsApiClient | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.ipums_api_key

        if client is None and not self.api_key:
            raise ValueError(
                "IPUMS_API_KEY is not set; pass api_key= or set it in the environment."
            )

        self.storage_dir = (
            storage_dir
            if storage_dir is not None
            else settings.paths.external / "ipums"
        )
        self.client = (
            client if client is not None else IpumsApiClient(api_key=self.api_key)
        )

    def extract(
        self,
        collection: str,
        samples: Sequence[str],
        variables: Sequence[str],
        description: str = "",
        data_quality_flags: bool = True,
        data_structure: dict[str, dict[str, str]] | None = None,
        request_kind: RequestKind = "new_samples",
        force: bool = False,
        allow_flag_variables: bool = False,
    ) -> ExtractionRecord:
        """Pull a microdata extract for `collection` and save the raw .dat.gz +
        DDI .xml codebook as-is.

        Args:
            collection (str):
                IPUMS collection name, e.g. "cps" or "usa"
            samples (Sequence[str]):
                IPUMS sample IDs, e.g. ["cps2006_09s"]
            variables(Sequence[str]):
                IPUMS variable names, e.g. ["AGE", "SEX"]
            description (str):
                Free-text label sent to IPUMS and recorded on the extract
            data_quality_flags (bool):
                Whether to pull IPUMS Data quality flags
            data_structure (dict[...]):
                A form of data to pull: Hierarchical or rectangular data
            request_kind(RequestKind):
                Takes one of two values:
                "new_samples", if only new data samples are requested.
                "variable_delta", if new variables for existing samples are
                requested.
            force (bool):
                Submit and download a new extract even if a manifest entry
                already covers this exact (samples, variables) request, instead
                of reusing it
            allow_flag_variables (bool):
                Submit even if a requested name looks like a flag column. The
                check is a label heuristic over the codebooks on disk; this is
                the override for when it is wrong

        Returns:
            ExtractionRecord. Contains entry records both what was requested and
            what the codebook says was delivered (`delivered_variables`, plus the
            `quality_flags`/`topcode_flags` maps).

        Raises:
            ValueError
                A requested variable is a flag column already known from a
                codebook on disk and `allow_flag_variables` is False; nothing
                is submitted.
            BadIpumsApiRequest
                The API rejected the request. Re-raised with the verbatim server
                message plus guidance about flag columns.
            FileNotFoundError
                The download completed but the expected
                {collection}_{extract_id:05d}.dat.gz/.xml pair is not on disk.
        """
        log.info(
            "ipums_extract_start", collection=collection, samples=samples, force=force
        )
        collection_dir = self.storage_dir / collection
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Normalization of variables for convenence and to avoid typos in config files.
        variables = tuple(v.upper() for v in variables)
        # samples = tuple(v.lower() for v in samples)

        # One read, two consumers: the flag registry and the cache lookup.
        # build_coverage (extract_incremental only) still reads for itself.
        manifest_entries = read_manifest(collection_dir)
        _validate_requested_variables(
            collection_dir, variables, manifest_entries, allow_flag_variables
        )

        effective_data_structure = (
            data_structure if data_structure is not None else _default_data_structure()
        )

        cached = (
            None
            if force
            else find_matching_extract(
                collection_dir,
                samples,
                variables,
                effective_data_structure,
                data_quality_flags,
                manifest_entries,
            )
        )
        if cached is not None:
            data_path, ddi_path, extract_id = (
                cached.data_path,
                cached.ddi_path,
                cached.extract_id,
            )
            log.info(
                "ipums_extract_cached",
                collection=collection,
                extract_id=extract_id,
                file_path=data_path,
            )
        else:
            data_path, ddi_path, extract_id = self._submit_and_download(
                collection=collection,
                collection_dir=collection_dir,
                samples=samples,
                variables=variables,
                description=description,
                data_quality_flags=data_quality_flags,
                data_structure=effective_data_structure,
            )
        metadata = {
            "collection": collection,
            "samples": tuple(samples),
            "variables": tuple(variables),
            "data_structure": effective_data_structure,
            "data_quality_flags": data_quality_flags,
            "extract_id": extract_id,
            "ddi_path": str(ddi_path),
            "cached": cached is not None,
            "request_kind": request_kind,
            "force": force,
        }
        # Record what the codebook says was delivered: every column in the
        # data file plus the flag maps, so a later reader need not re-parse it.
        summary = try_summarize_ddi(ddi_path)
        if summary is not None:
            # Stamps the parser that produced the maps below, so a later fix to
            # parse_flag_label invalidates them instead of being shadowed.
            metadata["flag_parser_version"] = FLAG_PARSER_VERSION
            metadata["delivered_variables"] = tuple(summary.variables)
            metadata["quality_flags"] = {
                source: list(names) for source, names in summary.quality_flags.items()
            }
            metadata["topcode_flags"] = {
                source: list(names) for source, names in summary.topcode_flags.items()
            }
        else:
            log.warning(
                "ipums_delivered_variables_unrecorded",
                collection=collection,
                extract_id=extract_id,
                ddi_path=str(ddi_path),
            )
        extraction_id = f"{collection}_{extract_id:05d}"

        recorded_size = cached.size_bytes if cached is not None else None
        recorded_sha = cached.sha256 if cached is not None else None

        if recorded_size is not None and recorded_sha is not None:
            record = ExtractionRecord(
                source="ipums_api",
                extraction_id=extraction_id,
                extracted_at=datetime.now(timezone.utc),
                file_path=data_path,
                size_bytes=recorded_size,
                sha256=recorded_sha,
                metadata=metadata,
            )
        else:
            # A cache hit whose entry recorded no usable checksum lands here
            # too - it re-hashes the file it already has, but does not append.
            record = build_extraction_record(
                source="ipums_api",
                extraction_id=extraction_id,
                file_path=data_path,
                metadata=metadata,
            )
            if cached is None:
                append_to_manifest(collection_dir, record)

        log.info(
            "ipums_extract_complete",
            collection=collection,
            extract_id=extract_id,
            file_path=str(data_path),
            cached=metadata["cached"],
            request_kind=request_kind,
        )
        return record

    def _submit_and_download(
        self,
        collection_dir: Path,
        collection: str,
        samples: Sequence[str],
        variables: Sequence[str],
        data_structure: dict[str, dict[str, str]],
        description: str,
        data_quality_flags: bool,
    ) -> tuple[Path, Path, int]:
        """Submit a new extract, wait for it, and download it to collection_dir.

        Returns:
            (data_path, ddi_path, extract_id) for the downloaded files.

        Raises:
            BadIpumsApiRequest: re-raised with the requested samples/variables and,
                if the message mentions a variable/mnemonic, a hint about flag
                columns.
            FileNotFoundError: the download completed but the expected
                {collection}_{extract_id:05d}.dat.gz/.xml pair is not on disk.
        """

        microdata_extract = MicrodataExtract(
            collection=collection,
            samples=list(samples),
            variables=list(variables),
            description=description,
            data_structure=data_structure,
        )
        if data_quality_flags:
            # Per-variable rather than the extract-level `data_quality_flags=`
            # kwarg: ipumspy forwards unknown kwargs to the API unvalidated, so a
            # typo there would silently produce a flagless extract. Assumes IPUMS
            # ignores the request for variables that have no flag.
            microdata_extract.add_data_quality_flags(list(variables))
        try:
            self.client.submit_extract(microdata_extract)
        except BadIpumsApiRequest as exc:
            hint = ""
            if "mnemonic" in str(exc).lower() or "variable" in str(exc).lower():
                hint = (
                    "\nIf any of these are IPUMS flag columns, they cannot "
                    "be requested by name: drop them from `variables` and "
                    "pass data_quality_flags=True - the flag column is added "
                    "automatically for every requested variable that has one."
                )
            raise BadIpumsApiRequest(
                f"{exc}\n\nRequested samples: {sorted(samples)}\n"
                f"Requested variables: {sorted(variables)}{hint}"
            ) from exc

        self.client.wait_for_extract(microdata_extract)
        self.client.download_extract(microdata_extract, download_dir=collection_dir)
        extract_id = microdata_extract.extract_id
        data_path = collection_dir / f"{collection}_{extract_id:05d}.dat.gz"
        ddi_path = collection_dir / f"{collection}_{extract_id:05d}.xml"

        if not data_path.exists() or not ddi_path.exists():
            found = sorted(p.name for p in collection_dir.glob(f"*{extract_id:05d}*"))
            raise FileNotFoundError(
                f"Expected {data_path.name} and {ddi_path.name} after download, "
                f"found instead: {found}"
            )

        return data_path, ddi_path, extract_id

    def extract_incremental(
        self,
        collection: str,
        samples: Sequence[str],
        variables: Sequence[str],
        data_quality_flags: bool = True,
        data_structure: dict[str, dict[str, str]] | None = None,
        description: str = "",
        force: bool = False,
        allow_flag_variables: bool = False,
    ) -> list[ExtractionRecord]:
        """Extract just enough to cover `samples`/`variables`, given what this
        collection already has on disk per _MANIFEST.yaml.

        Compares the request against the collection's coverage
        (extractors.ipums_coverage.build_coverage) and submits 0, 1, or 2
        extracts depending on what's missing:
          - nothing missing -> returns [] without contacting the IPUMS API
          - only new samples -> one "new_samples" extract (full variable list)
          - only new variables for already-known samples -> one
            "variable_delta" extract (just the missing variables)
          - both -> one of each

        `force=True` skips the missing-variable/missing-sample diffing that
        drives the non-forced path above, but still splits on known-vs-new
        samples the same way: samples already in coverage are submitted as
        a "variable_delta" force-pull (so the parse stage merges them onto
        existing bronze columns instead of overwriting the whole file), and
        genuinely new samples as a "new_samples" force-pull - one or two
        extracts, mirroring extract(force=True) per group.
        With force=True, two extracts may be submitted.

        Args:
            collection (str):
                IPUMS collection name, e.g. "cps" or "usa".
            samples (Sequence[str]):
                IPUMS sample IDs, e.g. ["cps2006_09s"].
            variables (Sequence[str]):
                IPUMS variable names, e.g. ["AGE", "SEX"].
            data_quality_flags (bool):
                Whether to pull IPUMS Data quality flags.
            data_structure (dict[str, dict[str, str]] | None):
                A form of data to pull: Hierarchical or rectangular data.
            description (str):
                Description of the extract, stored in the manifest entry's
                metadata.
            force (bool):
                If True, submit and download a new extract even if
                a manifest entry already covers this exact request
                (samples, variables), instead of reusing it
            allow_flag_variables (bool):
                Forwarded to extract(): submit even if a requested name looks
                like a flag column.

        Returns:
            list[ExtractionRecord]:
                List of extraction records for the extracts that were
                submitted and downloaded. Empty if nothing was missing
                and no new extracts were needed.
        """
        variables = tuple(v.upper() for v in variables)
        # samples = tuple(v.lower() for v in samples)

        collection_dir = self.storage_dir / collection
        if force:
            coverage = build_coverage(collection_dir, collection)
            # Force means "pull it regardless of whether plan_delta_requests
            # would think it's needed"
            known_samples = [s for s in samples if s in coverage.samples]
            new_samples = [s for s in samples if s not in coverage.samples]
            groups: list[tuple[list[str], RequestKind]] = []
            if new_samples:
                groups.append((new_samples, "new_samples"))
            if known_samples:
                groups.append((known_samples, "variable_delta"))
            records = [
                self.extract(
                    collection=collection,
                    samples=group_samples,
                    variables=variables,
                    data_quality_flags=data_quality_flags,
                    data_structure=data_structure,
                    description=description,
                    request_kind=request_kind,
                    force=True,
                    allow_flag_variables=allow_flag_variables,
                )
                for group_samples, request_kind in groups
            ]
            if records:
                save_coverage(
                    build_coverage(collection_dir, collection), collection_dir
                )
            return records

        coverage = build_coverage(collection_dir, collection)
        planned = plan_delta_requests(coverage, samples, variables)
        if not planned:
            log.info(
                "ipums_extract_already_covered",
                collection=collection,
                samples=samples,
                variables=variables,
            )
            return []

        records = [
            self.extract(
                collection=collection,
                samples=plan.samples,
                variables=plan.variables,
                description=description,
                request_kind=plan.request_kind,
                data_quality_flags=data_quality_flags,
                data_structure=data_structure,
                allow_flag_variables=allow_flag_variables,
            )
            for plan in planned
        ]

        updated_coverage = build_coverage(collection_dir, collection)
        save_coverage(updated_coverage, collection_dir)

        return records
