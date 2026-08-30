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
    summarize_ddi(..):
        Returns information about variables and quality/topcode flags.
    try_summarize_ddi(..):
        A version of summarize_ddi() that logs a warning instead of raising
        in case of a corrupted raw XML-codebook. Memoized per codebook.
    clear_ddi_summary_cache():
        Empties that memo - for tests, and for a codebook rewritten in place.
    summary_from_metadata(..):
        Rebuild a DDISummary from what a manifest entry recorded, if the entry
        was written by the current FLAG_PARSER_VERSION.
    flag_columns_for(..):
        Returns all flag columns belonging to variables.
    merge_column_names(..):
        Joins all variables together: source variables and quality/topcode
        flags.
    registry_from_summaries(..):
        Unions several DDI summaries into one FlagRegistry, inverting each
        summary's source-variable -> flag-names maps.
    collection_flag_registry(..):
        Every flag known for a collection, resolved from the manifest's
        recorded maps, then each entry's codebook, then any loose *.xml.
"""

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import structlog
from ipumspy import Codebook, readers

from src.extractors.manifest import as_name_list, read_manifest

log = structlog.get_logger(__name__)

FlagKind = Literal["quality", "topcode"]

# Bump whenever parse_flag_label or its regexes change. extract() stamps this
# into every manifest entry it writes; a recorded flag map carrying a different
# stamp is ignored and re-derived from the codebook, so a parser fix reaches
# entries written before it.
#
# The stamp gates only the flag maps read back by summary_from_metadata.
# ipums_coverage._delivered_variables trusts a recorded `delivered_variables`
# list regardless of version, so changing how _summarize_codebook enumerates
# columns needs its own decision - this constant does not cover it.
#
# NOTE: entries written before this constant existed carry no stamp and there is
# no update-in-place in extractors.manifest, so they re-parse their codebook on
# every call - by design, and cheap next to one API round trip.
FLAG_PARSER_VERSION = 1

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

    Attributes:
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
        flag_names:
            Property. Every flag column named here, of either kind.
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
        DDISummary, or None if the codebook cannot be read or parsed - which
        is the whole point of this wrapper over summarize_ddi().
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
        # A parse failure is a property of the file and stays true until the file
        # changes (i.e., mtime/size are in the key). A resource failure is not.
        # Save info about file reading failure (None value) only
        # if it is not an environment failure.
        if not isinstance(exc, (MemoryError, OSError)):
            _SUMMARY_CACHE[key] = None
        return None
    _SUMMARY_CACHE[key] = summary
    return summary


def summary_from_metadata(
    metadata: Mapping[str, Any], require_current_parser: bool = True
) -> DDISummary | None:
    """Rebuild a DDISummary from what a manifest entry recorded, without
    touching the codebook. None if the entry predates those keys, or if the
    flag maps were written by a different parser version.

    Lets a reader keep working when the .xml itself has become unreadable but
    the manifest entry written at download time is intact.

    Args:
        metadata (Mapping[str, Any]):
            One manifest entry's `metadata` block. Not mutated.
        require_current_parser (bool):
            Reject a flag map not stamped with the current
            FLAG_PARSER_VERSION. False accepts a stale map, for a caller that
            has already established there is no codebook to re-derive from -
            a possibly-outdated answer beats no answer for a heuristic whose
            only job is "is this name a flag?".

    Returns:
        DDISummary, or None if the entry carries nothing usable.

    Note:
        An entry with no recorded `ddi_path` yields `ddi_path=Path(".")` - a
        marker, not a location, and one that passes `.exists()`. Do not treat
        `summary.ddi_path` from this function as a codebook without checking
        `.suffix == ".xml"`.
    """
    delivered = as_name_list(metadata.get("delivered_variables"))
    if not delivered:
        return None

    # A map produced by an older parse_flag_label is not trusted: the codebook
    # next to it on disk is the ground truth, so let the caller re-derive.
    # `!=` rather than `<` is deliberate - it also rejects a map written by a
    # newer checkout, and it is right when YAML hands the stamp back as "1".
    if require_current_parser and (
        metadata.get("flag_parser_version") != FLAG_PARSER_VERSION
    ):
        return None

    quality = metadata.get("quality_flags")
    topcode = metadata.get("topcode_flags")
    quality_flags = quality if isinstance(quality, dict) else {}
    topcode_flags = topcode if isinstance(topcode, dict) else {}

    return DDISummary(
        ddi_path=Path(str(metadata["ddi_path"]))
        if metadata.get("ddi_path")
        else Path(),
        variables=tuple(delivered),
        quality_flags=MappingProxyType({k: tuple(v) for k, v in quality_flags.items()}),
        topcode_flags=MappingProxyType({k: tuple(v) for k, v in topcode_flags.items()}),
    )


def flag_columns_for(
    summary: DDISummary,
    variables: Iterable[str],
    include_topcode: bool = True,
) -> tuple[str, ...]:
    """Determine flag columns belonging to any of `variables`.

    A flag shared by several source variables (QWKSWORK covers WKSWORK1 and
    WKSWORK2) is included when any one of them is requested.
    """
    requested = set(variables)
    mappings = [summary.quality_flags]
    if include_topcode:
        mappings.append(summary.topcode_flags)
    wanted = {
        name
        for mapping in mappings
        for source, names in mapping.items()
        if source in requested
        for name in names
    }
    return tuple(name for name in summary.variables if name in wanted)


def merge_column_names(
    summary: DDISummary | None,
    requested: Sequence[str],
    include_topcode: bool = True,
) -> list[str]:
    """Joins all variables together: source variables and quality/topcode flags.

    With summary=None, falls back to `requested` - exactly the behaviour before
    flag columns were tracked.
    """
    if summary is None:
        return list(requested)
    flags = flag_columns_for(summary, requested, include_topcode=include_topcode)
    requested_set = set(requested)
    return [*requested, *(name for name in flags if name not in requested_set)]


@dataclass(frozen=True)
class FlagRegistry:
    """Flag column names known for a collection, mapped back to the variables
    that they flag.

    Built from codebooks already on disk, so a flag becomes known
    once any extract has delivered it.

    Attributes:
        quality (Mapping[str, tuple[str, ...]]):
            Read-only view mapping each quality flag column to the source
            variable(s) it flags. NOTE: this is the inverse of
            DDISummary.quality_flags, which maps source variable -> flag names.
        topcode (Mapping[str, tuple[str, ...]]):
            Same, for topcode flag columns.

    Not hashable - see __hash__ below.

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

    # quality/topcode hold unhashable MappingProxyType views, so the
    # frozen=True default __hash__ would only fail by accident. Disabled
    # explicitly rather than hashing a partial subset of fields nothing needs.
    __hash__ = None  # type: ignore[assignment]

    def kind_of(self, name: str) -> FlagKind | None:
        """Which kind of flag `name` is, or None if it is not a flag."""
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
        # A hand-edited/truncated manifest can carry `metadata:` empty or as a
        # scalar. This runs on every extract(), so one bad entry must not block
        # the whole collection.
        metadata = entry.get("metadata") if isinstance(entry, dict) else None
        if not isinstance(metadata, dict):
            # Warn for both shapes: a scalar `metadata:` and a stray non-mapping
            # entry in the YAML list. Silence here is what makes a corrupted
            # manifest look like an empty one.
            log.warning(
                "ipums_manifest_entry_malformed",
                collection_dir=str(collection_dir),
                entry=str(entry)[:200],
            )
            continue
        recorded = summary_from_metadata(metadata)
        if recorded is not None:
            summaries.append(recorded)
            continue
        # try to read from raw XML-codebooks
        ddi_path = metadata.get("ddi_path")
        if ddi_path and Path(ddi_path).exists():
            summaries.append(try_summarize_ddi(Path(ddi_path)))
            continue
        # Codebook gone and the recorded map stamped out by an older parser:
        # a stale flag map still beats "this name is not a flag", which costs a
        # rejected API round trip. Use it - but never silently.
        stale = summary_from_metadata(metadata, require_current_parser=False)
        if stale is not None:
            log.info(
                "ipums_flag_map_stale_but_used",
                collection_dir=str(collection_dir),
                ddi_path=str(ddi_path),
                recorded_version=metadata.get("flag_parser_version"),
            )
            summaries.append(stale)

    # If nothing usable came out of the manifest (no entries, or every recorded
    # codebook is unreadable), sweep the directory itself.
    if (
        not any(summary is not None for summary in summaries)
        and collection_dir.exists()
    ):
        summaries = [
            try_summarize_ddi(path) for path in sorted(collection_dir.glob("*.xml"))
        ]

    return registry_from_summaries(summaries)
