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

from collections import Counter
from pathlib import Path

import structlog
import yaml

log = structlog.get_logger(__name__)


def parsing_config_path(config_root: Path, source: str, collection: str) -> Path:
    """Location of a collection's parsing config:
    {config_root}/{source}/{collection}.yaml."""
    return config_root / source / f"{collection}.yaml"


def load_expected_columns(
    path: Path, expect_collection: str | None = None
) -> frozenset[str] | None:
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
            emptpy list, or it names the same column twice. A malformed contract
            must not quietly become a narrower one - that would refuse valid
            extracts and mark good years as damaged.
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

    if not columns:
        raise ValueError(
            f"Parsing config {path} declares an empty 'expected_columns' - "
            f"delete the file to derive the set from bronze instead or "
            f"add columns"
        )

    duplicates = sorted(c for c, n in Counter(columns).items() if n > 1)

    if duplicates:
        raise ValueError(f"Parsing config {path} lists duplicate columns: {duplicates}")

    if expect_collection is not None:
        declared = payload.get("collection")
        if declared is not None and declared != expect_collection:
            raise ValueError(
                f"Parsing config {path} declares collection {declared!r}, "
                f"loaded as {expect_collection!r}"
            )
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
    """load_expected_columns for a collection's conventional config location.

    Args:
        config_root (Path):
            A folder with parse config files
        source (str):
            A data source name (e.g., "ipums")
        collection (str):
            A collection name (e.g., "cps")

    Returns:
        frozenset[str] | None
            A set of columns from parse config file if any. Otherwise, None.
    """
    path = parsing_config_path(config_root, source, collection)
    columns = load_expected_columns(path, expect_collection=collection)

    return columns
