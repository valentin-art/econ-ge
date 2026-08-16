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

Functions:
    parse_flag_label(..):
        Takes a variable label string a returns its flag kind.
    _summarize_codebook(..):
        Returns information about variables and quality/topcode flags.
    summarize_ddi(..):
        Returns information about variables and quality/topcode flags.
        wraps _summarize_codebook().
    try_summarize_ddi(..):
        A version of summarize_ddi() that  prints warning instead of
        rising error in case of corrupted raw XML-codebook.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import structlog
from ipumspy import Codebook, readers

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

# Regex expression that aims to find candidates of a source variable
# for the flag (QINCWAGE -> INCWAGE)
_SOURCE_SPLIT_RE = re.compile(r"\s+and\s+|\s*,\s*", re.I)

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
        >>> parse_flag_label("Flag for ASEC") is None
        True
    """
    for kind, pattern in _FLAG_LABEL_RES.items():
        match = pattern.match(label)
        if match is None:
            continue
        tail = _QUALIFIER_RE.sub("", match.group("sources")).strip()
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
        quality_flags (dict[...]):
            Quality flags and corresponding source variables.
        topcode_flags (dict[...]):
            Topcode flags and corresponding source variables.

    Methods:
        kind_of(name: str):
            Returns a kind of a flag for a given flag name, or nothing.
    """

    ddi_path: Path
    variables: tuple[str, ...]
    quality_flags: dict[str, tuple[str, ...]] = field(default_factory=dict)
    topcode_flags: dict[str, tuple[str, ...]] = field(default_factory=dict)

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
        for kind, mapping in (
            ("quality", self.quality_flags),
            ("topcode", self.topcode_flags),
        ):
            if any(name in names for names in mapping.values()):
                return kind  # type: ignore[return-value]
        return None


def _summarize_codebook(ddi_path: Path, codebook: Codebook) -> DDISummary:  # type: ignore[no-untyped-def]
    """Helper that takes a ddi's path and a codebook and generates a summary.

    The function uses a native Codebook instance from ipumspy (parced by
    ipums native reader) rather than a localcodebook file itself. That
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
        quality_flags={k: tuple(v) for k, v in quality_flags.items()},
        topcode_flags={k: tuple(v) for k, v in topcode_flags.items()},
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
        log.warning("ipums_ddi_unreadable", ddi_path=str(ddi_path), error=str(exc))
        _SUMMARY_CACHE[key] = None
        return None
    _SUMMARY_CACHE[key] = summary
    return summary
