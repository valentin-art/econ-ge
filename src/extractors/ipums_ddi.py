"""DDI codebook summaries: what an IPUMS extract actually delivered, and which
of those columns are flags attached to a requested variable.

IPUMS adds a data quality flag column for every requested variable that has
one whenever an extract is submitted with data_quality_flags=True, and adds
topcode flag columns regardless of that setting. Neither kind can be requested
by name - they are a property of a requested variable.

The DDI labels them unambiguously - "Data quality flag for INCWAGE", "Topcode
Flag for INCFARM". That label is the only signal used here.

Classes:
    DDISummary:
        Collects a content of a codebook: variables and quality/topcode flags.
    FlagRegistry:
        Every flag column known for a collection, mapped back to its source
        variable(s).

Functions:
    parse_flag_label(..):
        Takes a variable label string and returns its flag kind.
    _summarize_codebook(..):
        Returns information about variables and quality/topcode flags.
    summarize_ddi(..):
        Returns information about variables and quality/topcode flags.
        wraps _summarize_codebook().
    try_summarize_ddi(..):
        A version of summarize_ddi() that  prints warning instead of
        raising error in case of corrupted raw XML-codebook.
    summary_from_metadata(..):
        Rebuild a DDISummary from what a manifest entry recorded.
    registry_from_summaries(..)
    collection_flag_registry(..)
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import structlog
from ipumspy import Codebook, readers

from src.extractors.manifest import read_manifest

log = structlog.get_logger(__name__)

FlagKind = Literal["quality", "topcode"]

# Regex expressions aimed to recognize flags in local codebooks
_FLAG_LABEL_RES: dict[FlagKind, re.Pattern[str]] = {
    "quality": re.compile(
        r"^\s*data\s+quality\s+flags?\s+for\s+(?P<sources>.+)$", re.I
    ),
    "topcode": re.compile(r"^\s*topcode\s+flags?\s+for\s+(?P<sources>.+)$", re.I),
}

# Regex expression that aims to recognize a flag kind
_QUALIFIER_RE = re.compile(r"\s*\[[^\]]*\]\s*$")

# Regex expression that strips trailing sentence punctuation left over once
# qualifiers are removed (e.g. "INCFARM." -> "INCFARM")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:]+\s*$")

# Regex expression that aims to find candidates of a source variable
# for the flag (QINCWAGE -> INCWAGE)
_SOURCE_SPLIT_RE = re.compile(r"\s*,\s*and\s+|\s+and\s+|\s*,\s*", re.I)


# Regex expression that aims to determine if sepected candidate is
# a valid variable from a local codebook
_VALID_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def parse_flag_label(label: str) -> tuple[FlagKind, tuple[str, ...]] | None:
    """Takes a variable label and determines flag kind and source variable(s)

    Args:
        label (str): A string label (e.g., from a codebook).

    Returns:
        tuple[FlagKind, tuple[str, ...]]. See examples below.

    Examples:
        >>> parse_flag_label("Data quality flag for WKSWORK1 and WKSWORK2")
        ('quality', ('WKSWORK1', 'WKSWORK2'))
        >>> parse_flag_label("Data quality flag for SRCEARN [detailed version]")
        ('quality', ('SRCEARN',))
        >>> parse_flag_label("Topcode Flag for INCFARM")
        ('topcode', ('INCFARM',))
        >>> parse_flag_label("Topcode Flag for INCFARM.")
        ('topcode', ('INCFARM',))
        >>> parse_flag_label("Flag for ASEC") is None
        True
    """
    for kind, pattern in _FLAG_LABEL_RES.items():
        match = pattern.match(label)
        if match is None:
            continue
        tail = _QUALIFIER_RE.sub("", match.group("sources")).strip()
        tail = _TRAILING_PUNCT_RE.sub("", tail).strip()
        candidates = (token.strip() for token in _SOURCE_SPLIT_RE.split(tail))
        sources = tuple(
            candidate for candidate in candidates if _VALID_NAME_RE.match(candidate)
        )
        if sources:
            return kind, sources
    return None


@dataclass(frozen=True)
class DDISummary:
    """What one extract's codebook says the extract's data file contains.

    Atrributes:
        ddi_path (Path):
            Path that contains dictionaries.
        variables (tuple[...]):
            Every column in the data file, in codebook order - requested
            variables, their flag columns, and technical IPUMS columns.
        quality_flags (Mapping[...]):
            Quality flags and corresponding source variables. A read-only
            view (types.MappingProxyType): frozen=True only stops the field
            itself from being reassigned, not its contents from being
            mutated in place - and every cache hit in _SUMMARY_CACHE hands
            out the same shared instance, so an in-place mutation here would
            corrupt what every other caller sees for this codebook.
        topcode_flags (Mapping[...]):
            Topcode flags and corresponding source variables. Same
            read-only-view reasoning as quality_flags.

    Not hashable - see __hash__ below.

    Methods:
        kind_of(name: str):
            Returns a kind of a flag for a given flag name, or nothing.
    """

    ddi_path: Path
    variables: tuple[str, ...]
    quality_flags: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    topcode_flags: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    # quality_flags/topcode_flags hold unhashable MappingProxyType views, so
    # the frozen=True default __hash__ would only fail by accident. Disabled
    # explicitly rather than hashing a partial subset of fields nothing needs.
    __hash__ = None  # type: ignore[assignment]

    @property
    def flag_names(self) -> frozenset[str]:
        """Every flag column, of either kind."""
        return frozenset(
            name
            for mapping in (self.quality_flags, self.topcode_flags)
            for names in mapping.values()
            for name in names
        )

    def kind_of(self, name: str) -> FlagKind | None:
        """Which kind of flag `name` is, or None if it is not a flag."""
        if any(name in names for names in self.quality_flags.values()):
            return "quality"
        if any(name in names for names in self.topcode_flags.values()):
            return "topcode"
        return None


def _summarize_codebook(ddi_path: Path, codebook: Codebook) -> DDISummary:
    """Helper that takes a ddi's path and a codebook and generates a summary.

    The function uses a native Codebook instance from ipumspy (parsed by
    ipums native reader) rather than a local codebook file itself. That
    instance already contains a label string that is used to identify flag
    kind.

    Args:
        ddi_path (Path):
            Path to local codebooks. Not used - passed through.
        codebook (Codebook):
            A Codebook instance with valid variable labels.

    Returns:
        DDISummary.
    """
    variables: list[str] = []
    quality_flags: dict[str, list[str]] = {}
    topcode_flags: dict[str, list[str]] = {}

    for description in codebook.data_description:
        variables.append(description.name)
        parsed = parse_flag_label(description.label or "")
        if parsed is None:
            continue
        kind, sources = parsed
        mapping = quality_flags if kind == "quality" else topcode_flags
        for source in sources:
            names = mapping.setdefault(source, [])
            if description.name not in names:
                names.append(description.name)

    return DDISummary(
        ddi_path=ddi_path,
        variables=tuple(variables),
        quality_flags=MappingProxyType({k: tuple(v) for k, v in quality_flags.items()}),
        topcode_flags=MappingProxyType({k: tuple(v) for k, v in topcode_flags.items()}),
    )


# Trying to avoid reading the same raw dictionary (i.e., XML-file from raw
# extract) and re-evaluating DDISummary many times repeatedly: by extract(),
# build_coverage(), etc.
#
# Keyed on (resolved path, mtime_ns, size) rather than the path alone. It
# guards from file change if XML-file is re-downloaded to the same filename:
# Only the change in last-modified-time (mtime_ns) and file-size (size)
# triggers updating DDISummary.
_SUMMARY_CACHE: dict[tuple[str, int, int], DDISummary | None] = {}


def clear_ddi_summary_cache() -> None:
    """Drop every cached summary. For tests that rewrite a codebook in place."""
    _SUMMARY_CACHE.clear()


def _cache_key(ddi_path: Path) -> tuple[str, int, int]:
    stat = ddi_path.stat()
    return (str(ddi_path.resolve()), stat.st_mtime_ns, stat.st_size)


def summarize_ddi(ddi_path: Path) -> DDISummary:
    """Summarize one DDI codebook.

    Args:
        ddi_path (Path):
            Path to a codebook

    Returns:
        DDISummary
    """
    key = _cache_key(ddi_path)
    cached = _SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached
    summary = _summarize_codebook(ddi_path, readers.read_ipums_ddi(ddi_path))
    _SUMMARY_CACHE[key] = summary
    return summary


def try_summarize_ddi(ddi_path: Path) -> DDISummary | None:
    """A version of summarize_ddi, but returns a None (with a warning)
    instead of an exception when the codebook is unreachable.

    Used as a soft alternative to avoid early stop when it may corrupt metadata.
    Examples:
    - build_coverage() goes through all codebooks and extract variables.
        If a corrupted codebook raises error, coverage will not be constructed
        even over perfectly fine extracts.
    - extract() fails without any record in MANIFEST.yaml although the data file
        is already downloaded. Instead of hard stop, that function prints
        warning that helps to identify what went wrong. Further, XML-codebooks
        can be extracted manually (can be used for parsing job).

    Args:
        ddi_path (Path):
            Path to a codebook

    Returns:
        DDISummary
    """
    try:
        key = _cache_key(ddi_path)
    except OSError as exc:
        log.warning("ipums_ddi_unreadable", ddi_path=str(ddi_path), error=str(exc))
        return None
    if key in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[key]
    try:
        summary = _summarize_codebook(ddi_path, readers.read_ipums_ddi(ddi_path))
    except Exception as exc:
        log.warning(
            "ipums_ddi_unreadable",
            ddi_path=str(ddi_path),
            error=str(exc),
            exc_info=True,
        )
        _SUMMARY_CACHE[key] = None
        return None
    _SUMMARY_CACHE[key] = summary
    return summary


def summary_from_metadata(metadata: dict) -> DDISummary | None:
    """Rebuild a DDISummary from what a manifest entry recorded, without
    touching the codebook. None if the entry predates those keys.

    Lets a reader keep working when the .xml itself has become unreadable but
    the manifest entry written at download time is intact.

    Args:
        metadata (dict):
            Metadata parsed into dict.

    Returns:
        DDISummary
    """
    delivered = metadata.get("delivered_variables")
    if not delivered:
        return None
    return DDISummary(
        ddi_path=Path(metadata.get("ddi_path", "")),
        variables=tuple(delivered),
        quality_flags=MappingProxyType(
            {k: tuple(v) for k, v in (metadata.get("quality_flags") or {}).items()}
        ),
        topcode_flags=MappingProxyType(
            {k: tuple(v) for k, v in (metadata.get("topcode_flags") or {}).items()}
        ),
    )


@dataclass(frozen=True)
class FlagRegistry:
    """Flag column names known for a collection, mapped back to the variables
    that they flag.

    Built from codebooks already on disk, so a flag becomes known
    once any extract has delivered it.

    Attributes:
        quality (dict):
            Maps each quality flag to corresponding source variables.
        topcode (dict):
            Maps each topcode flag to corresponding source variables.

    Methods:
        kind_of(name):
            "quality" / "topcode" for a known flag column, else None.
        sources_of(name):
            The source variable(s) `name` flags, or () if it is not a flag.
        __bool__():
            True if any quality/topcode flags exist.
    """

    quality: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    topcode: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    # quality_flags/topcode_flags hold unhashable MappingProxyType views, so
    # the frozen=True default __hash__ would only fail by accident. Disabled
    # explicitly rather than hashing a partial subset of fields nothing needs.
    __hash__ = None  # type: ignore[assignment]

    def kind_of(self, name: str) -> FlagKind | None:
        if name in self.quality:
            return "quality"
        if name in self.topcode:
            return "topcode"
        return None

    def sources_of(self, name: str) -> tuple[str, ...]:
        """Source variable(s) `name` flags, or () if `name` is not a flag."""
        if name in self.quality:
            return self.quality[name]
        return self.topcode.get(name, ())

    def __bool__(self) -> bool:
        return bool(self.quality or self.topcode)


def registry_from_summaries(summaries: Iterable[DDISummary | None]) -> FlagRegistry:
    """Takes several DDI summaries and unions flags into one registry."""
    quality: dict[str, list[str]] = {}
    topcode: dict[str, list[str]] = {}
    for summary in summaries:
        if summary is None:
            continue
        for mapping, out in (
            (summary.quality_flags, quality),
            (summary.topcode_flags, topcode),
        ):
            for source, names in mapping.items():
                for name in names:
                    sources = out.setdefault(name, [])
                    if source not in sources:
                        sources.append(source)
    return FlagRegistry(
        quality=MappingProxyType({k: tuple(v) for k, v in quality.items()}),
        topcode=MappingProxyType({k: tuple(v) for k, v in topcode.items()}),
    )


def collection_flag_registry(
    collection_dir: Path,
    manifest_entries: Collection[dict] | None = None,
) -> FlagRegistry:
    """Tries to collect all flags known for `collection_dir`.

    Resolution order, cheapest first:
      1. flag maps recorded in the manifest at download time - no file I/O;
      2. the codebook itself, for entries written before those keys existed;
      3. every *.xml in the directory, if there is no manifest at all.

    `manifest_entries` lets a caller that has already read _MANIFEST.yaml pass
    it in rather than re-reading it.
    """

    entries = (
        list(manifest_entries)
        if manifest_entries is not None
        else read_manifest(collection_dir)
    )

    summaries: list[DDISummary | None] = []
    for entry in entries:
        # try to read from manifest
        metadata = entry.get("metadata") or {}
        recorded = summary_from_metadata(metadata)
        if recorded is not None:
            summaries.append(recorded)
            continue
        # try to read from raw XML-codebooks
        ddi_path = metadata.get("ddi_path")
        if ddi_path and Path(ddi_path).exists():
            summaries.append(try_summarize_ddi(Path(ddi_path)))

    # If no success, try to look into all XML files in the directory
    if not summaries and collection_dir.exists():
        summaries = [
            try_summarize_ddi(path) for path in sorted(collection_dir.glob("*.xml"))
        ]

    return registry_from_summaries(summaries)
