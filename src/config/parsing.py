"""Versioned parsing configuration: the column set a source's bronze layer is
expected to hold.

Read from YAML under config/parsing/<source>/<collection>.yaml, the parse-side
counterpart to config/cleaning/. Loading is deliberately path-in/values-out and
imports neither Settings nor CleaningContext: a caller resolves the path and
passes the loaded columns on explicitly, so the parse layer stays runnable from
a notebook or a test with a literal list.

Configuring a collection is optional. Without one the parse pipeline derives
the expected set from the years already in bronze, which is the right default
for a collection whose shape nobody has had to pin down yet.
"""

from collections.abc import Collection
from pathlib import Path

import structlog
import yaml

log = structlog.get_logger(__name__)


def parsing_config_path(config_root: Path, source: str, collection: str) -> Path:
    """Location of a collection's parsing config:
    {config_root}/{source}/{collection}.yaml."""
    return config_root / source / f"{collection}.yaml"


def load_expected_columns(path: Path) -> frozenset[str] | None:
    """Read the expected bronze column set from a parsing config file.

    Args:
        path (Path):
            The config file. A missing file is not an error - it means the
            collection has no declared contract.

    Returns:
        frozenset[str] | None:
            The declared columns, or None when the file does not exist or
            declares none.

    Raises:
        ValueError:
            The file exists but `expected_columns` is not a list of strings,
            or it names the same column twice. A malformed contract must not
            quietly become a narrower one - that would refuse valid extracts
            and mark good years as damaged.
    """
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Parsing config {path} must be a mapping")
    columns = payload.get("expected_columns")
    if columns is None:
        return None
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise ValueError(
            f"Parsing config {path} has an 'expected_columns' that is not a "
            f"list of strings"
        )
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"Parsing config {path} lists duplicate columns: {duplicates}")
    log.info(
        "ipums_parsing_config_loaded",
        path=str(path),
        collection=payload.get("collection"),
        n_columns=len(columns),
    )
    return frozenset(columns)


def load_collection_expected_columns(
    config_root: Path, source: str, collection: str
) -> frozenset[str] | None:
    """load_expected_columns for a collection's conventional config location."""
    return load_expected_columns(parsing_config_path(config_root, source, collection))


def describe_columns(columns: Collection[str] | None) -> str:
    """Short human-readable summary of a contract, for CLI output."""
    if columns is None:
        return "derived from bronze"
    return f"{len(columns)} declared column(s)"
