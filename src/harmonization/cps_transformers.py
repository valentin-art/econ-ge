"""Generic column-deriving primitives for IPUMS CPS bronze data.

Ported from the recurring column-derivation patterns in `aa_clean/`
(Katz-Murphy/Wasserman-lineage March CPS cleaning, 1962-2009). Each function
takes the specific input column(s) as `pd.Series` plus cleaning parameters -
never the whole DataFrame - and returns a `pd.Series` for the caller to
assign back, e.g.:

    df["EXP"] = experience_years(df["AGE"], years_schooling)

This is deliberately narrower than `cps_filters.py`'s `(df, column, ...)`
shape: filters need whole-DataFrame context to drop rows uniformly,
transformers only need the columns they actually read. Year-dependent facts
(topcode thresholds, deflators, the 1992+ schooling-years lookup) are passed
in from `cps_tables.py` rather than branched on internally, since IPUMS's
own harmonization removes most of the year-block structure the original
Stata needed.

Four corrected discrepancies vs. the original `aa_clean/*.do` (see plan doc
for detail, not replicated here):
  - `topcode_cap` for AGE is meant to apply to every year; the original's
    1992-2009 block omitted the `age=90 if age>=90` line.
  - `topcode_cap` for hours-last-week: the original's
    `replace hrsly=98 if hrslyr==99` targets a variable name (`hrsly`) that
    is never otherwise read - a typo that left the real `hrslyr`/`hours`
    variable untopcoded. Callers should apply `topcode_cap` to the actual
    hours column.
  - `cps_tables.KM_1982_DEFLATOR_REF` matches 11 of the original's 12
    occurrences of `109.031/53.615`; `clean7678km.do`'s lone `bcwkwg` used
    `109.031/3.615` (copy-paste typo), not replicated.
  - Hourly-wage windsorization (capping `hinc_ws` at the topcode-implied
    ceiling) was applied in only 3 of the original's 8 year-blocks; a
    windsorization transformer, when added, should be applied uniformly.
"""

from collections.abc import Hashable
from functools import reduce
from operator import mul
from typing import Literal

import numpy as np
import pandas as pd


def topcode_cap(series: pd.Series, ceiling: float) -> pd.Series:
    """Cap `series` at `ceiling` (values above it become `ceiling`)."""
    return series.clip(upper=ceiling)


def rescale(series: pd.Series, factor: float) -> pd.Series:
    """Multiply `series` by `factor`.

    Named (rather than left as inline `*`) because its aa_clean use case -
    undoing ASECWT's two implied decimal places (`wgt/100`) - is a unit
    fact worth documenting at the call site, not obvious from the code
    alone.
    """
    return series * factor


def weighted_product(*series: pd.Series) -> pd.Series:
    """Elementwise product of two or more Series.

    Covers aa_clean's derived weights: `wgt_wks = wgt*WKSWORK1`,
    `wgt_hrs = wgt*WKSWORK1*UHRSWORKLY`, `wgt_hrs_ft = wgt*UHRSWORKLY`.
    """
    return reduce(mul, series)


def code_to_value(
    series: pd.Series, table: dict[Hashable, float], default: float | None = None
) -> pd.Series:
    """Map `series` through a `{code: value}` lookup table.

    Backs every year-keyed or code-keyed lookup in `cps_tables.py`: the
    CPI/GDP deflator merge (`code_to_value(df["YEAR"] - 1, CPI_DEFLATOR)`),
    the MAXER/MAXWG per-year topcode threshold, and the WKSWORK2
    bracket-midpoint bridge. `default` fills codes absent from `table`
    (left as NaN if omitted).
    """
    mapped = series.map(table)
    if default is not None:
        mapped = mapped.fillna(default)
    return mapped


def multi_key_lookup(
    keys: pd.DataFrame, table: dict[tuple, float], default: float | None = None
) -> pd.Series:
    """Map each row of `keys` (columns in table-key order) through `table`.

    Backs `cps_tables.EDUCOMP_LOOKUP` (56-row race/sex/EDUC-code -> years-
    of-schooling table): `multi_key_lookup(df[["_RACE3", "FEMALE", "EDUC"]], EDUCOMP_LOOKUP)`
    replaces 56 explicit branches with one generic call.
    """
    tuple_keys = pd.Series(
        list(keys.itertuples(index=False, name=None)), index=keys.index
    )
    mapped = tuple_keys.map(table)
    if default is not None:
        mapped = mapped.fillna(default)
    return mapped


def topcode_multiplier(
    income: pd.Series,
    threshold: float,
    multiplier: float = 1.5,
    comparison: Literal["eq", "ge"] = "ge",
) -> pd.Series:
    """Scale `income` by `multiplier` where it hits a topcode `threshold`.

    `comparison="eq"` matches aa_clean's pre-1988 style (exact-equality
    against a year-banded threshold from `INCOME_TOPCODE_PRE1988`);
    `comparison="ge"` (default) matches its 1988+ style (`>=` against the
    per-year `MAXER`/`MAXWG` tables). Rows not meeting the condition pass
    through unchanged.
    """
    if comparison == "eq":
        hit = income == threshold
    else:
        hit = income >= threshold
    return income.where(~hit, income * multiplier)


def implied_rate_topcode_flag(
    income: pd.Series, denominator: pd.Series, rate_ceiling: float
) -> pd.Series:
    """Flag rows where `income / denominator > rate_ceiling`.

    Covers aa_clean's `tcwkwg` (`denominator=WKSWORK1`, weekly rate) and
    `tchrwg` (`denominator=WKSWORK1 * UHRSWORKLY`, hourly rate); the caller
    computes `rate_ceiling` from the topcode threshold, e.g.
    `threshold * 1.5 / 40` (weekly) or `threshold * 1.5 / 1400` (hourly).
    """
    return (income / denominator) > rate_ceiling


def real_value_flag(
    nominal: pd.Series, deflator: pd.Series, dollar_ceiling: float
) -> pd.Series:
    """Flag rows where `nominal * deflator < dollar_ceiling`.

    Covers aa_clean's low-wage flags `bcwkwg`/`bchrwg`/`bcwkwgkm`/
    `bchrwgkm`; the caller passes
    `dollar_ceiling = <40|1|67|1.675> * cps_tables.KM_1982_DEFLATOR_REF` for
    the four respective cutoffs.
    """
    return (nominal * deflator) < dollar_ceiling


def experience_years(
    age: pd.Series,
    years_schooling: pd.Series,
    schooling_start_age: int = 7,
    min_working_age: int = 17,
) -> pd.Series:
    """Potential labor-market experience, aa_clean's `exp` formula.

    `max(min(age - years_schooling - schooling_start_age, age - min_working_age), 0)`
    - assumes schooling starts at `schooling_start_age` and experience
    accumulation can't predate `min_working_age`. Same formula serves both
    the pre-1992 (HIGRADE-derived `educomp`) and post-1992
    (`cps_tables.EDUCOMP_LOOKUP`-derived) years-of-schooling inputs.
    """
    capped_by_schooling = age - years_schooling - schooling_start_age
    capped_by_min_age = age - min_working_age
    return np.maximum(np.minimum(capped_by_schooling, capped_by_min_age), 0)


def age_last_year(age: pd.Series, max_topcoded_age: int = 71) -> pd.Series:
    """Age as of last year, aa_clean's `agely` formula.

    `age - 1`, capped at `max_topcoded_age` for anyone who was already at
    or above the cap this year (equivalent to aa_clean's
    `agely=age-1 if 17<=age<=71; agely=71 if age>=72`: for `age=72`,
    `age-1=71=max_topcoded_age`; for a topcoded `age=90`, `age-1=89` clips
    down to `71`).
    """
    return np.minimum(age - 1, max_topcoded_age)
