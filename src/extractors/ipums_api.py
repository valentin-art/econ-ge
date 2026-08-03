"""IPUMS API extractor: external -> raw .dat.gz + DDI .xml codebook on disk.

Submits a microdata extract request, waits for it to complete, and downloads
the data file and its DDI codebook as-is.
"""

from collections.abc import Sequence
from pathlib import Path

import structlog
from ipumspy import IpumsApiClient, MicrodataExtract

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
from src.extractors.manifest import append_to_manifest, read_manifest

log = structlog.get_logger(__name__)


def _default_data_structure() -> dict[str, dict[str, str]]:
    return {"rectangular": {"on": "P"}}


def find_matching_extract(
    collection_dir: Path,
    samples: Sequence[str],
    variables: Sequence[str],
    data_structure: dict[str, dict[str, str]],
    data_quality_flags: bool,
) -> tuple[Path, Path, int] | None:
    """Return (data_path, ddi_path, extract_id) for the most recent manifest
    entry in `collection_dir` whose samples exactly match `samples` and whose
    variables are a superset of `variables`, provided its data file and DDI
    codebook both still exist on disk. The data_structure and data_quality_flags must also match. Returns None if no such entry exists.
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
        if metadata.get("data_structure") != data_structure:
            continue
        if metadata.get("data_quality_flags") != data_quality_flags:
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

        Parameters
        ----------
        collection : IPUMS collection name, e.g. "cps" or "usa"
        samples    : IPUMS sample IDs, e.g. ["cps2006_09s"]
        variables  : IPUMS variable names, e.g. ["AGE", "SEX"]
        data_quality_flags: Whether to pull IPUMS Data quality flags
        data_structure: A form of data to pull: Hierarchical or rectangular data
        request_kind: recorded in the manifest entry's metadata so the parse
                     stage knows whether to parse this extract as a normal
                     new-samples pull or merge it into existing bronze data
                     for samples that were already partially covered
        force      : submit and download a new extract even if a manifest
                     entry already covers this exact (samples, variables)
                     request, instead of reusing it
        """
        log.info(
            "ipums_extract_start", collection=collection, samples=samples, force=force
        )
        collection_dir = self.storage_dir / collection
        collection_dir.mkdir(parents=True, exist_ok=True)

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
                data_quality_flags=data_quality_flags,
                data_structure=effective_data_structure,
            )
            self.client.submit_extract(microdata_extract)
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
        }
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

        `force=True` skips coverage-checking entirely and submits a single
        fresh extract for exactly (samples, variables), like extract(force=True).

        Args:
            collection (str): IPUMS collection name, e.g. "cps" or "usa"
            samples (Sequence[str]): IPUMS sample IDs, e.g. ["cps2006_09s"]
            variables (Sequence[str]): IPUMS variable names, e.g. ["AGE", "SEX"]
            data_quality_flags (bool): Whether to pull IPUMS Data quality flags
            data_structure (dict[str, dict[str, str]] | None): A form of data to pull: Hierarchical or rectangular data
            description (str): Description of the extract, stored in the manifest entry's metadata
            force (bool): If True, submit and download a new extract even if a manifest entry already covers this exact (samples, variables) request, instead of reusing it

        Returns:
            list[ExtractionRecord]: List of extraction records for the extracts that were submitted and downloaded. Empty if nothing was missing and no new extracts were needed.
        """
        collection_dir = self.storage_dir / collection
        if force:
            record = self.extract(
                collection=collection,
                samples=samples,
                variables=variables,
                data_quality_flags=data_quality_flags,
                data_structure=data_structure,
                description=description,
                force=True,
            )
            save_coverage(build_coverage(collection_dir, collection), collection_dir)
            return [record]

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
