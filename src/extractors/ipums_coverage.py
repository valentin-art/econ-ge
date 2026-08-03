"""Coverage tracking for IPUMS extracts: derives, from _MANIFEST.yaml (cross-
checked against the .dat.gz/.xml files it references), which samples and
variables a collection already has on disk, and plans the minimal set of new
extracts needed to satisfy a bigger request.

Pure and network-free - operates only on manifest entries and Path.exists()
checks, so it's testable without hitting the IPUMS API. Used by
extractors.ipums_api.IPUMSExtractor.extract_incremental to avoid re-submitting
extracts that would just duplicate samples/variables already pulled (every
submit_extract call counts against the user's IPUMS account quota).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from src.extractors.manifest import read_manifest

_SAMPLE_YEAR_RE = re.compile(r"(\d{4})")

_COVERAGE_FILENAME = "_COVERAGE.yaml"

RequestKind = Literal["new_samples", "variable_delta"]


def parse_sample_year(sample: str) -> int | None:
    """Best-effort year parsed out of an IPUMS sample id, e.g. 'cps2006_09s'
    -> 2006. For reporting/the coverage summary only - the match/diff logic
    below compares exact sample-id strings, never derived years.
    """
    match = _SAMPLE_YEAR_RE.search(sample)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class SampleCoverage:
    variables: frozenset[str]
    extraction_ids: tuple[str, ...]


@dataclass(frozen=True)
class CollectionCoverage:
    collection: str
    samples: dict[str, SampleCoverage]

    @property
    def variables(self) -> frozenset[str]:
        """Union of variables ever pulled for any sample in this collection."""
        result: set[str] = set()
        for coverage in self.samples.values():
            result |= coverage.variables
        return frozenset(result)

    @property
    def years(self) -> frozenset[int]:
        """Unique years parsed from every covered sample id (reporting only)."""
        years = (parse_sample_year(sample) for sample in self.samples)
        return frozenset(year for year in years if year is not None)


def build_coverage(collection_dir: Path, collection: str) -> CollectionCoverage:
    """Read collection_dir/_MANIFEST.yaml, drop any entry whose data file or
    DDI codebook no longer exists on disk, and union the surviving entries'
    variables into per-sample coverage.
    """
    variables_by_sample: dict[str, set[str]] = {}
    extraction_ids_by_sample: dict[str, list[str]] = {}

    for entry in read_manifest(collection_dir):
        metadata = entry["metadata"]
        data_path = Path(entry["file_path"])
        ddi_path = Path(metadata["ddi_path"])
        if not data_path.exists() or not ddi_path.exists():
            continue
        for sample in metadata["samples"]:
            variables_by_sample.setdefault(sample, set()).update(metadata["variables"])
            ids = extraction_ids_by_sample.setdefault(sample, [])
            # extract() appends a manifest entry on every call, including
            # cache hits that just re-point at an already-recorded
            # extraction_id - keep each id once, in first-seen order.
            if entry["extraction_id"] not in ids:
                ids.append(entry["extraction_id"])

    samples = {
        sample: SampleCoverage(
            variables=frozenset(variables_by_sample[sample]),
            extraction_ids=tuple(extraction_ids_by_sample[sample]),
        )
        for sample in variables_by_sample
    }
    return CollectionCoverage(collection=collection, samples=samples)


def save_coverage(coverage: CollectionCoverage, collection_dir: Path) -> Path:
    """Write collection_dir/_COVERAGE.yaml - fully rebuilt from `coverage`
    every time, never hand-edited, never itself a source of truth (that's
    still _MANIFEST.yaml). A read-optimized summary of all unique years and
    variables ever requested for the collection, and which extraction_id(s)
    cover each sample.
    """
    out_path = collection_dir / _COVERAGE_FILENAME
    payload = {
        "collection": coverage.collection,
        "years": sorted(coverage.years),
        "variables": sorted(coverage.variables),
        "samples": {
            sample: {
                "variables": sorted(sample_coverage.variables),
                "extraction_ids": list(sample_coverage.extraction_ids),
            }
            for sample, sample_coverage in sorted(coverage.samples.items())
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    tmp_path.replace(out_path)

    return out_path


@dataclass(frozen=True)
class PlannedExtract:
    samples: list[str]
    variables: list[str]
    request_kind: RequestKind


def plan_delta_requests(
    coverage: CollectionCoverage,
    samples: list[str],
    variables: list[str],
) -> list[PlannedExtract]:
    """Work out the minimal set of new extracts (0, 1, or 2) needed for
    `coverage` to fully cover `samples` (all of them) with `variables` (all
    of them).

    - Samples entirely missing from `coverage` -> a "new_samples" extract,
      pulled with the full requested variable list.
    - Samples already covered but missing at least one requested variable ->
      a "variable_delta" extract, pulled with just the union of missing
      variables across those samples (one MicrodataExtract applies a single
      variable list to every sample in it, so this may pull a few
      already-covered variables for some of those samples too - harmless).
    - Samples that already cover every requested variable contribute nothing
      to either extract.
    - If neither case applies, returns [] - the request is already fully
      covered, nothing new to submit.
    """
    requested_variables = set(variables)
    normalized_variables = sorted(requested_variables)
    new_samples: list[str] = []
    stale_samples: list[str] = []
    missing_variables: set[str] = set()

    for sample in samples:
        sample_coverage = coverage.samples.get(sample)
        if sample_coverage is None:
            new_samples.append(sample)
            continue
        missing = requested_variables - sample_coverage.variables
        if missing:
            stale_samples.append(sample)
            missing_variables |= missing

    planned: list[PlannedExtract] = []
    if new_samples:
        planned.append(PlannedExtract(new_samples, normalized_variables, "new_samples"))
    if stale_samples:
        planned.append(
            PlannedExtract(stale_samples, sorted(missing_variables), "variable_delta")
        )
    return planned
