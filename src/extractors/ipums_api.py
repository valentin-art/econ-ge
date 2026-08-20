"""IPUMS API extractor: external -> raw .dat.gz + DDI .xml codebook on disk.

Submits a microdata extract request, waits for it to complete, and downloads
the data file and its DDI codebook as-is.
"""

from collections.abc import Sequence
from pathlib import Path

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
from src.extractors.ipums_ddi import collection_flag_registry, try_summarize_ddi
from src.extractors.manifest import append_to_manifest, read_manifest

log = structlog.get_logger(__name__)


def _default_data_structure() -> dict[str, dict[str, str]]:
    return {"rectangular": {"on": "P"}}


def _validate_requested_variables(
    collection_dir: Path, variables: Sequence[str]
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

    Returns:
        None.

    Raises:
        ValueError, if any variable is identified as a flag.
    """
    registry = collection_flag_registry(collection_dir)
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
                f"{variable} is the IPUMS data quality flag for {sources}, not a "
                f"requestable variable. Drop it from `variables` and pass "
                f"data_quality_flags=True - the flag column is added "
                f"automatically for every requested variable that has one."
            )
        else:
            problems.append(
                f"{variable} is the IPUMS topcode flag for {sources}, not a "
                f"requestable variable. Drop it from `variables`; it is "
                f"delivered automatically alongside {sources} regardless of "
                f"data_quality_flags."
            )
    if problems:
        raise ValueError("\n".join(problems))


def find_matching_extract(
    collection_dir: Path,
    samples: Sequence[str],
    variables: Sequence[str],
    data_structure: dict[str, dict[str, str]],
    data_quality_flags: bool,
) -> tuple[Path, Path, int] | None:
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

    Returns:
        tuple[Path, Path, int] or None
        Contains following information
            data_path (Path): Path containing data for matched collection.
            ddi_path (Path): Path containing dictionaries for matched collection.
            extract_id: A number of raw data extract for matched collection.
        None if no such entry exists.
    """
    requested_samples = set(samples)
    requested_variables = set(variables)
    match = None
    for entry in read_manifest(collection_dir):
        metadata = entry["metadata"]
        if set(metadata["samples"]) != requested_samples:
            continue
        if not requested_variables <= set(metadata["variables"]):
            continue
        if metadata.get("data_structure", _default_data_structure()) != data_structure:
            continue
        if metadata.get("data_quality_flags", True) != data_quality_flags:
            continue
        data_path = Path(entry["file_path"])
        ddi_path = Path(metadata["ddi_path"])
        if data_path.exists() and ddi_path.exists():
            match = (data_path, ddi_path, metadata["extract_id"])
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

        Returns:
            ExtractionRecord. Contains entry records both what was requested and
            what the codebook says was delivered (`delivered_variables`, plus the
            `quality_flags`/`topcode_flags` maps).

        Raises:
            ValueError
                A requested variable is a flag column already known from a
                codebook on disk; nothing is submitted.
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
        # ipumspy upper-cases Variable.name on construction, and add_data_quality_flags
        # resolves by exact string match - so normalize before anything looks a name up.
        variables = tuple(v.upper() for v in variables)
        _validate_requested_variables(collection_dir, variables)

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
            )
        )
        if cached is not None:
            data_path, ddi_path, extract_id = cached
            log.info(
                "ipums_extract_cached",
                collection=collection,
                extract_id=extract_id,
                file_path=str(data_path),
            )
        else:
            microdata_extract = MicrodataExtract(
                collection=collection,
                samples=list(samples),
                variables=list(variables),
                description=description,
                data_structure=effective_data_structure,
            )
            if data_quality_flags:
                microdata_extract.add_data_quality_flags(list(variables))
            try:
                self.client.submit_extract(microdata_extract)
            except BadIpumsApiRequest as exc:
                raise BadIpumsApiRequest(
                    f"{exc}\n\n"
                    f"Requested variables: {sorted(variables)}\n"
                    f"If any of these are IPUMS flag columns, they cannot be "
                    f"requested by name: drop them from `variables` and pass "
                    f"data_quality_flags=True, and the flag column is added "
                    f"automatically for every requested variable that has one."
                ) from exc
            self.client.wait_for_extract(microdata_extract)
            self.client.download_extract(microdata_extract, download_dir=collection_dir)
            extract_id = microdata_extract.extract_id
            data_path = collection_dir / f"{collection}_{extract_id:05d}.dat.gz"
            ddi_path = collection_dir / f"{collection}_{extract_id:05d}.xml"

            if not data_path.exists() or not ddi_path.exists():
                found = sorted(
                    p.name for p in collection_dir.glob(f"*{extract_id:05d}*")
                )
                raise FileNotFoundError(
                    f"Expected {data_path.name} and {ddi_path.name} after download, "
                    f"found instead: {found}"
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
        # Variable types should be recorded to metadata
        summary = try_summarize_ddi(ddi_path)
        if summary is not None:
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
        record = build_extraction_record(
            source="ipums_api",
            extraction_id=extraction_id,
            file_path=data_path,
            metadata=metadata,
        )
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

    def extract_incremental(
        self,
        collection: str,
        samples: Sequence[str],
        variables: Sequence[str],
        data_quality_flags: bool = True,
        data_structure: dict[str, dict[str, str]] | None = None,
        description: str = "",
        force: bool = False,
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
                (samples, variables) , instead of reusing it

        Returns:
            list[ExtractionRecord]:
                List of extraction records for the extracts that were
                submitted and downloaded. Empty if nothing was missing
                and no new extracts were needed.
        """
        variables = tuple(v.upper() for v in variables)
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
            )
            for plan in planned
        ]

        updated_coverage = build_coverage(collection_dir, collection)
        save_coverage(updated_coverage, collection_dir)

        return records
