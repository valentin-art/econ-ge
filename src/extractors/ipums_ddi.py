"""DDI codebook label parsing: which columns in an IPUMS codebook are flags
attached to a requested variable.

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
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
