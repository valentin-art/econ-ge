"""Generic row-dropping primitives for IPUMS CPS bronze data.

Ported from the recurring row-filter patterns in `aa_clean/` (Katz-Murphy/
Wasserman-lineage March CPS cleaning, 1962-2009: `launcher.do`,
`clean6275km.do`, `clean7678km.do`, `clean7909km.do`). Unlike the original
Stata, these operate on the already-harmonized IPUMS bronze DataFrame, so
one parameterized function per *shape* of condition replaces what was
repeated per year-block in the source. Compose them at the call site to
build a specific sample, e.g. aa_clean's "universe filter" is:

    df = range_filter(df, "AGE", min_value=16)
    df = membership_filter(df, "POPSTAT", [1])
    df = range_filter(df, "WKSWORK1", 1, 52)

Companion module `cps_transformers.py` holds the column-deriving (non-row-
dropping) side; `cps_tables.py` holds the data constants referenced by
filter parameters (e.g. `CLASSWLY_WAGE_CODES`).
"""

from collections.abc import Collection, Sequence

import pandas as pd


def range_filter(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> pd.DataFrame:
    """Keep rows where `min_value <= df[column] <= max_value`.

    Either bound may be omitted for a one-sided range. Rows with NaN in
    `column` are dropped along with out-of-range rows, since comparisons
    against NaN are always False - matching Stata's `!=.` missing-value
    checks without a separate step.
    """
    mask = pd.Series(True, index=df.index)
    if min_value is not None:
        mask &= df[column] >= min_value
    if max_value is not None:
        mask &= df[column] <= max_value
    return df[mask]


def membership_filter(
    df: pd.DataFrame, column: str, allowed_values: Collection[object]
) -> pd.DataFrame:
    """Keep rows where `df[column]` is one of `allowed_values`."""
    return df[df[column].isin(allowed_values)]


def exclude_filter(
    df: pd.DataFrame, column: str, excluded_values: Collection[object]
) -> pd.DataFrame:
    """Keep rows where `df[column]` is NOT one of `excluded_values`.

    Complement of `membership_filter` - useful when a sample is more
    naturally described by what it excludes (e.g. self-employed/government
    codes) than by what it includes.
    """
    return df[~df[column].isin(excluded_values)]


def not_missing_filter(df: pd.DataFrame, columns: str | Sequence[str]) -> pd.DataFrame:
    """Keep rows with no NaN in any of `columns`."""
    cols = [columns] if isinstance(columns, str) else list(columns)
    return df[df[cols].notna().all(axis=1)]
