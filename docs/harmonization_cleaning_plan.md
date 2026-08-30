# `src/harmonization/` cleaning functions from harmonization.md + cps_clean/

Status: **superseded in shape, not in substance.** This document plans a
split into *domain* modules (`demographics.py`, `labor_status.py`,
`wages.py`, ...). What was actually built instead is a set of *generic
primitives* plus transcribed constants, with tests in
`tests/unit/harmonization/`:

| Actual module | Contents |
|---|---|
| `cps_filters.py` | `range_filter`, `membership_filter`, `exclude_filter`, `not_missing_filter` |
| `cps_transformers.py` | `topcode_cap`, `rescale`, `code_to_value`, `multi_key_lookup`, `topcode_multiplier`, `experience_years`, `age_last_year`, ... |
| `cps_tables.py` | year-keyed and code-keyed constants transcribed from `aa_clean/` |

Neither `demographics.py` nor `labor_status.py` exists. Read the sections
below as the **methodological requirements** (which variables need what
treatment, and why) rather than as a description of the current file
layout — the requirements still hold; the packaging decision changed to
composable primitives that a `src/cleaning/` `Step` calls into.

## Context

`notebooks/harmonization.md` records a Q&A that classifies every ASEC/CPS
variable AKK (Autor-Katz-Kearney) touches into three blocks relative to
IPUMS's own harmonization: (1) AKK applies safely as-is, (2) AKK composes
with IPUMS in ways that require a *switchable* stage (risk of
double-treatment), (3) AKK doesn't apply at all and needs a modern
replacement. The legacy Stata scripts in `cps_clean/` (`cps_clean.do`,
`cps_clean_upd1.do`, `cps_clean_for_wages.do`,
`cps_clean_for_wages_upd.do`) are the actual historical implementation of
these ideas — `cps_clean_for_wages_upd.do` is the most complete/final
version and is the primary porting source for anything ported here.

The goal is a first draft of the Block 1 + Block 2 (+ two Block 3 pieces)
cleaning functions as plain, testable Python functions in a new
`src/harmonization/` package, following the variable-domain module split
already used in `src/features/bea/`. Any place needing information not
present in the repo (dictionaries, crosswalk tables) is called out rather
than guessed at (see "Explicit gaps" below), and dataset *content* is read
only through the saved JSON variable dictionaries in
`data/reference/ipums/cps/`, never by opening the raw `.dat.gz`/`.xml`.

User decisions already made (via clarifying questions):
- The wage-sample `CLASSWLY` filter matches the **legacy Stata code**, not
  the looser prose in harmonization.md: exclude both self-employed
  (`CLASSWLY` 10/13/14) and government employees (`CLASSWLY` 24-28).
- Block 3 covers only the two pieces that need **no external crosswalk
  data** — weights documentation and quality-flag passthrough.
  `occupation.py` (OCC→task-quadrant mapping) is **deferred entirely**;
  it is not scaffolded until the crosswalk gap below is resolved.

## Explicit gaps — external information needed (flag, don't guess)

1. **INCWAGE topcode/swap methodology per year** (blocks the switchable
   topcode stage, harmonization.md Block 2 / lines 23-24). The saved
   dictionary `data/reference/ipums/cps/cps_00032.json` has **zero value
   labels** for `INCWAGE` (it's continuous), so there is no per-year
   "raw vs. IPUMS-swapped" signal anywhere in what's on disk. This has to
   come from IPUMS's published income-topcode documentation
   (cps.ipums.org "Income Top Codes" tables), which isn't in this repo.
   The switch should default the AKK multiplier **off** everywhere until
   that per-year table is supplied, so it never silently double-treats.
2. **QINCWAGE availability is extract-dependent**: present in
   `cps_00030.json`, absent from the newer `cps_00032.json`. The
   allocation-exclusion function needs to detect this per-extract rather
   than assume the flag always exists — per harmonization.md: "flag
   present → filter; flag absent → `allocation_exclusion_unavailable`",
   not an error.
3. **CPI/deflator series** (`cpi_old` in the legacy `.do` files) is not in
   this repo. Real-wage construction (dividing by CPI) is **out of scope**
   for `src/harmonization/` — that's silver/feature-layer work analogous to
   `src/features/bea/deflators.py`, and will need a CPI-U or PCE series
   wired in separately later.
4. **WKSWORK1/WKSWORK2 bridge should eventually be shared with PSID**
   (harmonization.md's explicit point — same bridge function must be used
   on both sources so the weeks metric doesn't diverge). No PSID cleaning
   code exists in this repo yet. Building it CPS-only for now is correct,
   but it should be written source-agnostically (pure function on a
   `WKSWORK1`/`WKSWORK2`-shaped input) once PSID work starts.
5. **OCC→task-quadrant mapping** needs a crosswalk that isn't in the repo:
   the legacy Stata referenced an external `occ1990dd-recode.csv` (AA2011)
   at an unreachable absolute path, and the
   `adjust_occ19{60,70,80,90}_occ1990dd.do` bridge files in `cps_clean/`
   are **empty (0 bytes)** — the actual occ1960/70/80/90 → OCC1990DD
   adjustment logic was never committed. `occupation.py` stays deferred
   until this crosswalk (or a DOT/O*NET task-index table) is sourced.

## Design

`src/harmonization/`, one module per harmonization.md domain/block,
mirroring the plain-function style of `src/features/bea/deflators.py` (no
classes, docstrings explaining the *why*, functions over
`pd.DataFrame`/`pd.Series`). Inputs are the bronze IPUMS long DataFrame
(uppercase IPUMS column names, as produced by
`parsers.ipums.parser_ipums.parse_to_bronze`).
Wherever a function needs to know what a code means (is 999 "missing"? is 0
"NIU"?), it resolves that through the saved JSON dictionary via
`src.parsers.cps.dictionary_lookup.get_variable_info` (pointed at
`settings.paths.reference / "ipums"`) rather than a hardcoded magic number —
except for constants that are pure AKK/methodology convention (age band
16-64, topcode 1.5x multiplier, week-interval midpoints), which are not
IPUMS-dictionary facts and stay as function defaults/parameters, matching
the values already used in `cps_clean_for_wages_upd.do`.

### Block 1 — implemented

- **`src/harmonization/demographics.py`**
  - `working_age_filter(df, min_age=16, max_age=64)` — AKK working-age
    restriction on `AGE`.
  - `unrecognized_cell_codes(df, variable, collection="cps")` — boolean
    mask flagging rows whose code isn't documented in the saved IPUMS
    dictionary, via `get_variable_info`.
  - `validate_demographic_cells(df, variables=(SEX, RACE, HISPAN, MARST,
    RELATE), collection="cps")` — per-variable masks; AKK treats these as
    cells/reconstruction inputs, not values to clean, so this only audits
    that IPUMS's own coding is what's expected.
- **`src/harmonization/labor_status.py`**
  - `employed_in_labor_force_filter(df)` — `EMPSTAT==10` ('At work') and
    `LABFORCE==2` ('Yes, in the labor force').
  - `wage_salary_sample_filter(df)` — excludes `CLASSWLY` 10/13/14
    (self-employed) **and** 24-28 (federal/state/local government),
    matching `cps_clean_for_wages_upd.do:40`.
  - `passthrough_industry_control(df)` — documented no-op for `INDLY` as a
    control variable (moves to Block 3 if ever used as an industry axis).

Tests: `tests/unit/harmonization/test_demographics.py`,
`tests/unit/harmonization/test_labor_status.py` — synthetic DataFrames,
`get_variable_info` mocked via `monkeypatch` so dictionary-lookup tests
don't depend on what's currently on disk. Run with
`uv run pytest tests/unit/harmonization -q`.

### Block 2 — planned, not yet built

- **`src/harmonization/wages.py`**, ported from
  `cps_clean_for_wages_upd.do:242-360`:
  - `apply_topcode_multiplier(df, year_topcode_table)` — switchable AKK
    1.5x-above-topcode stage. `year_topcode_table` required
    `{year: TopcodeYearConfig}` (blocked on gap #1); years missing from the
    table stay untouched (multiplier off).
  - `allocation_exclusion(df, flag_col="QINCWAGE")` — filters on the
    quality flag when present; adds `allocation_exclusion_unavailable=True`
    when the column is missing (gap #2), never fabricates a filter.
  - `top_decile_wage_by_year(df)` — diagnostic table to verify the topcode
    switch behaves as intended.
- **`src/harmonization/hours_weeks.py`**, ported from
  `cps_clean_for_wages_upd.do:206-235`:
  - `allocation_exclusion` reused/generalized from `wages.py` for
    `QWKSWORK1`/`QWKSWORK2`/`QUHRSWORKLY`.
  - `bridge_weeks_worked(df)` — WKSWORK1/WKSWORK2 interval→midpoint bridge
    (7/20/33/43.5/48.5/51 for WKSWORK2 codes 1-6), written source-agnostic
    per gap #4.
  - `uhrsworkly_filter(df, per_year_ranges)` — positive-hours/sensible-range
    filter with caller-supplied per-year bounds and an `impossible_value`
    audit flag.
- **`src/harmonization/education.py`**, ported from
  `cps_clean_for_wages_upd.do:157-198`: `bridge_educ_to_4level(df)`,
  reusing the existing `school`/`edcat5`-style EDUC→level mapping, adapted
  to AKK's 4-level education tiers with an explicit pre/post-1992 bridge.

### Block 3 — two pieces planned (occupation.py deferred, gap #5)

- **`src/harmonization/weights.py`** — `document_weight_controls(df)`,
  passthrough for `ASECWT`/`ASECWTH` attaching a
  `weight_controls_revised=True` flag, per harmonization.md's
  "document, don't fix" guidance.
- **`src/harmonization/quality_flags.py`** — `carry_quality_indicators(df,
  cols=("QEDUC","QOCC","QOCCLY"))` keeps these as indicator columns rather
  than filtering; `sensitivity_drop(df, flag_col)` supports the
  include-vs-exclude sensitivity comparison harmonization.md calls for.

## Critical files referenced

- `notebooks/harmonization.md` — block classification driving the module
  split and each function's docstring rationale.
- `cps_clean/cps_clean_for_wages_upd.do` — porting source: topcode
  multiplier (lines 242-360), weeks bridge (206-235), education tiers
  (157-198), CLASSWLY sample filter (line 40).
- `data/reference/ipums/cps/cps_00032.json` — current variable dictionary;
  confirms which variables/value-labels are actually available (45 vars,
  covers essentially all of Blocks 1-2) and that `INCWAGE` has no value
  labels (gap #1).
- `src/parsers/cps/dictionary_lookup.py:get_variable_info` — the
  source-agnostic dictionary-lookup helper reused here (pointed at
  `data/reference/ipums/` instead of its default `data/reference/cps/`).
- `src/features/bea/deflators.py` — style precedent for this kind of
  module (plain functions, docstrings, no classes).
- `src/config/settings.py:DataPaths.ipums_clean_dictionaries_dir` /
  `ipums_bronze_dir` — path conventions for wiring this into a future
  bronze→silver job (not built yet, out of scope for now).

## Verification

- `tests/unit/harmonization/` mirrors the `tests/unit/features/` layout —
  one test module per source module, exercising each function against
  small synthetic DataFrames (not real bronze data).
- Run `uv run pytest tests/unit/harmonization -q`.
- No UI/browser verification needed — these are pure data-transform
  functions with no interactive surface.
