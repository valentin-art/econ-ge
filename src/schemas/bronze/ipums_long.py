"""Schema for the tidy DataFrame returned by IPUMS parsers, and for the column
structure of a whole bronze collection.

Two grains live here:

  - validate_ipums_long / check_no_duplicate_columns validate a *single*
    DataFrame. No fixed column set: the columns depend on which variables were
    requested in the extract, so they only check shape.
  - modal_columns / bronze_column_deviations validate the *collection's* shape
    across years - the "every year holds the same columns" invariant that a
    single DataFrame cannot express.

The second pair never names a column. The expected set is either declared by
the caller or derived from the data, so the column list is written down in one
place and it is not this module.
"""

from collections import Counter
from collections.abc import Collection, Mapping

import pandas as pd


def check_no_duplicate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if `df` has duplicate column names."""
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"IPUMS DataFrame has duplicate columns: {dupes}")


def validate_ipums_long(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a parsed IPUMS DataFrame; raises ValueError on failure."""
    check_no_duplicate_columns(df)
    if df.empty:
        raise ValueError("IPUMS DataFrame has no rows")
    return df


def modal_columns(observed: Mapping[int, Collection[str]]) -> frozenset[str]:
    """The most common column set across years - the expected set when the
    caller declares none.

    Modal rather than union or intersection so that one damaged year cannot
    move the target: a single year that lost its columns is outvoted by every
    year that kept them, whereas a union would absorb its stray columns and an
    intersection would shrink to its remnants.

    Ties break on (count, n_columns), so when exactly two shapes appear the
    same number of times the wider one wins - a year can only ever be judged
    against a target at least as complete as itself. A tie on both count and
    width resolves to the earliest year's shape.

    Args:
        observed (Mapping[int, Collection[str]]):
            Columns actually present per year.

    Returns:
        frozenset[str]:
            The modal column set, empty if `observed` is empty.
    """
    if not observed:
        return frozenset()
    counts = Counter(frozenset(columns) for columns in observed.values())
    return max(counts.items(), key=lambda item: (item[1], len(item[0])))[0]


def bronze_column_deviations(
    observed: Mapping[int, Collection[str]],
    expected: Collection[str],
) -> dict[int, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Years whose columns are not a superset of `expected`, as
    {year: (missing, extra)} with both tuples sorted.

    A strict superset is valid and absent from the result: a variable-delta
    extract pulled for only some years legitimately leaves those years wider
    than the rest, and that is not damage. Only a year missing expected
    columns is. `extra` is reported alongside `missing` for the years that do
    deviate, because a year that both lost and gained columns was written by
    an extract of the wrong grain, and the stray columns name it.

    Args:
        observed (Mapping[int, Collection[str]]):
            Columns actually present per year.
        expected (Collection[str]):
            The column set every year must hold.

    Returns:
        dict[int, tuple[tuple[str, ...], tuple[str, ...]]]:
            {year: (missing, extra)} for deviating years only.
    """
    expected_set = frozenset(expected)
    deviations = {}
    for year, columns in sorted(observed.items()):
        column_set = frozenset(columns)
        missing = expected_set - column_set
        if missing:
            deviations[year] = (
                tuple(sorted(missing)),
                tuple(sorted(column_set - expected_set)),
            )
    return deviations
