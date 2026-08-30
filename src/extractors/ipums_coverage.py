"""What a collection already has on disk, and what a new request still needs.

Folds _MANIFEST.yaml into per-sample coverage - what was requested, and what the
files actually delivered - and diffs a request against it to plan the minimal
set of new extracts. Used by IPUMSExtractor.extract_incremental, because every
IPUMS submission counts against the account's extract quota.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import structlog
import yaml

from src.extractors.ipums_ddi import try_summarize_ddi
from src.extractors.manifest import as_name_list, read_manifest

log = structlog.get_logger(__name__)

# Keys build_coverage reads without a fallback. Kept as frozensets at module
# scope so the per-entry loop does not rebuild them.
_REQUIRED_METADATA = frozenset({"samples", "variables", "ddi_path"})
_REQUIRED_ENTRY = frozenset({"file_path", "extraction_id"})

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
    """Sample coverage definition.

    Attributes:
        requested_variables:
            Variables explicitly asked for in some extract covering the sample.
        delivered_variables:
            Columns actually present in those extracts' data files (e.g, iculding
            flags).
        extraction_ids:
            Actual extraction IDs that actually contain delivered variables.
    """

    requested_variables: frozenset[str]
    delivered_variables: frozenset[str]
    extraction_ids: tuple[str, ...]


@dataclass(frozen=True)
class CollectionCoverage:
    collection: str
    samples: Mapping[str, SampleCoverage]

    # Same reasoning as DDISummary/FlagRegistry: `samples` is unhashable, so
    # the frozen=True default __hash__ would only ever fail by accident.
    __hash__ = None  # type: ignore[assignment]

    @property
    def requested_variables(self) -> frozenset[str]:
        """Union of variables ever requested for any sample in this collection."""
        result: set[str] = set()
        for coverage in self.samples.values():
            result |= coverage.requested_variables
        return frozenset(result)

    @property
    def delivered_variables(self) -> frozenset[str]:
        """Union of columns ever delivered for any sample in this collection."""
        result: set[str] = set()
        for coverage in self.samples.values():
            result |= coverage.delivered_variables
        return frozenset(result)

    @property
    def years(self) -> frozenset[int]:
        """Unique years parsed from every covered sample id (reporting only)."""
        years = (parse_sample_year(sample) for sample in self.samples)
        return frozenset(year for year in years if year is not None)


def _delivered_variables(entry: dict[str, Any], ddi_path: Path) -> tuple[str, ...]:
    """Return the columns that an entry's extract really delivered.

    Prefers what the entry recorded at download time. Falls back to the
    local codebook as to the ground truth. Falls back further to the
    requested list, which understates the file by every flag and preselected
    column.
    """
    metadata = entry["metadata"]
    recorded = as_name_list(metadata.get("delivered_variables"))
    if recorded is not None:
        return tuple(recorded)

    summary = try_summarize_ddi(ddi_path)
    if summary is not None:
        return summary.variables

    log.warning(
        "ipums_coverage_delivered_fallback",
        extraction_id=entry.get("extraction_id"),
        ddi_path=str(ddi_path),
    )
    return tuple(metadata["variables"])


def build_coverage(collection_dir: Path, collection: str) -> CollectionCoverage:
    """Read collection_dir/_MANIFEST.yaml, drop any entry whose data file or
    DDI codebook no longer exists on disk, and union the surviving entries'
    requested and delivered variables into per-sample coverage.
    """
    requested_by_sample: dict[str, set[str]] = {}
    delivered_by_sample: dict[str, set[str]] = {}
    extraction_ids_by_sample: dict[str, list[str]] = {}

    for entry in read_manifest(collection_dir):
        metadata = entry.get("metadata") if isinstance(entry, dict) else None
        if not isinstance(metadata, dict) or not _REQUIRED_METADATA <= metadata.keys():
            log.warning(
                "ipums_manifest_entry_skipped",
                reason="missing_required_metadata_keys",
                entry=str(entry)[:200],
            )
            continue
        # Both are read below without a further guard, and neither can be
        # defaulted: Path("") is Path(".") - which exists - so a missing
        # file_path would sail past the existence check and be counted as
        # covered, hiding a sample that was never actually downloaded.
        if not _REQUIRED_ENTRY <= entry.keys():
            log.warning(
                "ipums_manifest_entry_skipped",
                reason="missing_required_entry_keys",
                entry=str(entry)[:200],
            )
            continue
        data_path = Path(entry["file_path"])
        ddi_path = Path(metadata["ddi_path"])

        if not data_path.exists() or not ddi_path.exists():
            continue

        sample_field = as_name_list(metadata["samples"])
        variables_field = as_name_list(metadata["variables"])

        if sample_field is None or variables_field is None:
            log.warning(
                "ipums_manifest_entry_skipped",
                reason="samples_or_variables_not_a_list_of_names",
                entry=str(entry)[:200],
            )
            continue

        delivered = _delivered_variables(entry, ddi_path)

        for sample in metadata["samples"]:
            requested_by_sample.setdefault(sample, set()).update(variables_field)
            delivered_by_sample.setdefault(sample, set()).update(delivered)
            ids = extraction_ids_by_sample.setdefault(sample, [])
            # extract() appends a manifest entry on every call, including
            # cache hits that just re-point at an already-recorded
            # extraction_id - keep each id once, in first-seen order.
            if entry["extraction_id"] not in ids:
                ids.append(entry["extraction_id"])

    samples = {
        sample: SampleCoverage(
            requested_variables=frozenset(requested_by_sample[sample]),
            delivered_variables=frozenset(delivered_by_sample[sample]),
            extraction_ids=tuple(extraction_ids_by_sample[sample]),
        )
        for sample in requested_by_sample
    }
    return CollectionCoverage(collection=collection, samples=MappingProxyType(samples))


def save_coverage(coverage: CollectionCoverage, collection_dir: Path) -> Path:
    """Write collection_dir/_COVERAGE.yaml - fully rebuilt from `coverage`
    every time, never hand-edited, never itself a source of truth (that's
    still _MANIFEST.yaml). A read-optimized summary of all unique years and
    variables ever pulled for the collection, and which extraction_id(s)
    cover each sample.

    `variables` is the delivered column set - what the files really contain,
    flag columns included. `requested_variables` is what was actually asked
    for, and is the set plan_delta_requests diffs against.
    """
    out_path = collection_dir / _COVERAGE_FILENAME
    payload = {
        "collection": coverage.collection,
        "years": sorted(coverage.years),
        "variables": sorted(coverage.delivered_variables),
        "requested_variables": sorted(coverage.requested_variables),
        "samples": {
            sample: {
                "variables": sorted(sample_coverage.delivered_variables),
                "requested_variables": sorted(sample_coverage.requested_variables),
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
    samples: tuple[str, ...]
    variables: tuple[str, ...]
    request_kind: RequestKind


def plan_delta_requests(
    coverage: CollectionCoverage,
    samples: Sequence[str],
    variables: Sequence[str],
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

    Diffs against each sample's *requested* variables, not its delivered ones.
    Delivered columns include flags and preselected technical columns that a
    caller cannot ask for by name, and - for samples covered only by
    variable-delta extracts - columns that were deliberately dropped before
    reaching bronze, so they do not mean "you already have this".
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
        missing = requested_variables - sample_coverage.requested_variables
        if missing:
            stale_samples.append(sample)
            missing_variables |= missing

    planned: list[PlannedExtract] = []
    if new_samples:
        planned.append(
            PlannedExtract(
                tuple(new_samples), tuple(normalized_variables), "new_samples"
            )
        )
    if stale_samples:
        planned.append(
            PlannedExtract(
                tuple(stale_samples), tuple(sorted(missing_variables)), "variable_delta"
            )
        )
    return planned
