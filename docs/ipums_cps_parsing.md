# IPUMS CPS parsing: structure and logic

How the raw `.dat.gz`/`.xml` files that
[extraction](ipums_cps_extraction.md) downloads become tidy bronze parquet
and a reference variable dictionary. Covers `src/parsers/ipums/parser_ipums.py`,
`src/pipelines/ipums_parse_pipeline.py`, and
`src/schemas/bronze/ipums_long.py`.

## Where the code lives

`src/parsers/` is split one package per data source, because the three
sources share a layer but not a line of logic - each reads a different raw
format against a different metadata convention:

```
src/parsers/
├── ipums/parser_ipums.py        ← this document
├── bea/parser_bea.py            ← BEA JSON → tidy long
│   ├── asset_dim.py                 line numbers, depreciation rates, IT buckets
│   └── wide.py                      long → LineNumber × Year pivot
└── cps/parser_cps.py            ← NBER CPS fixed-width + .sps dictionary
    └── dictionary_lookup.py         lookup over the per-source CPS dictionaries
```

Two names collide across packages on purpose: `parse_to_bronze` and
`bronze_path` exist in both `parser_bea` and `parser_ipums`, doing the
analogous thing for their own source. The package prefix is what tells them
apart at the import site, so keep importing the module path rather than
re-exporting either name upward.

## No hand-rolled fixed-width layout

CPS's other source in this repo (NBER, via `parsers/cps/parser_cps.py`) has no machine-
readable dictionary, so that parser hand-rolls a fixed-width reader against
a regex-parsed `.sps` file. IPUMS ships a real DDI 2.5 XML codebook, and
`ipumspy` already knows how to read both the codebook
(`readers.read_ipums_ddi`) and the data file against it
(`readers.read_microdata_chunked`). Nothing here reimplements column-position
parsing - the DDI is parsed once and used for two independent things: (a)
building the tidy bronze DataFrame, (b) building a JSON variable dictionary,
in the same CPS-mirroring shape used elsewhere in this repo.

## On-disk layout

```
data/bronze/ipums/{collection}/
└── {year}.parquet              ← one file per calendar year, all requested
                                    variables for that year, columns accumulate
                                    over time as variable_delta extracts land

data/reference/ipums/{collection}/
└── {year}.json                 ← {variable: {start, end, numeric,
                                    Description, Values}}, same shape as
                                    parser_cps's dictionaries; keys accumulate
                                    on every write and never shrink on their
                                    own (prune_variable_dictionary is the only
                                    thing that removes one)

config/parsing/ipums/
└── {collection}.yaml           ← optional, versioned: the expected bronze
                                    column set for the collection
```

The two `data/` trees are keyed by **year**, not by extract_id - a single
extract can span many years (`cps_00030` alone produced all of
`1962.parquet` ... `2025.parquet`), and a single year accumulates
contributions from multiple extracts over time (a `cps_00030` full pull, then
`cps_00033`/`cps_00034` delta merges, both touching the same year files).

## Usage example

The normal entry point is the pipeline function, which walks every
manifest-listed extract for a collection and parses/merges whatever isn't
already in `data/bronze/`:

```python
from src.config.settings import settings
from src.pipelines.ipums_parse_pipeline import parse_ipums_extracts

bronze_paths = parse_ipums_extracts(
    external_dir=settings.paths.external / "ipums",
    bronze_dir=settings.paths.bronze / "ipums",
    collection="cps",
)
# bronze_paths: list[Path], one entry per {year}.parquet actually
# written/updated this run - empty for any collection that was already
# fully covered.
```

Three **keyword-only** arguments control what it is allowed to touch:

| argument | default | effect |
|---|---|---|
| `replace` | `False` | permit a `new_samples` entry to overwrite a year that already has bronze |
| `years` | `None` | restrict the run to these years (`None` = every year the entries cover) |
| `expected_columns` | `None` | the column contract (`None` = derive it from bronze) |

They sit behind a `*` in the signature, so they can only ever be passed by
name - no caller can drift into enabling `replace` by counting positions.

`years=[]` means *no* years, not every year - the code tests
`years is not None` rather than truthiness, so an empty filter can never
silently become a full rebuild.

`expected_columns=[]` is different: it raises `ValueError`. An empty
contract constrains nothing (the column gate below is written
`if expected and ...`, so it would go dead and every year would trivially
conform), and that is far more likely to be an accident than an intent.
Pass `None` to derive the set instead. `_resolve_expected` enforces this for
all three entry points that take the argument.

To parse one already-downloaded extract directly, without going through the
manifest-driven skip logic (e.g. debugging one `.dat.gz`/`.xml` pair by
hand):

```python
from pathlib import Path
from src.parsers.ipums.parser_ipums import parse_to_bronze

year_paths = parse_to_bronze(
    data_path=Path("data/external/ipums/cps/cps_00030.dat.gz"),
    ddi_path=Path("data/external/ipums/cps/cps_00030.xml"),
    collection="cps",
    bronze_dir=settings.paths.bronze / "ipums",
    years=[2006],          # or it writes every year in the extract
    replace=True,          # or it refuses a year that already has bronze
)
```

Note both keywords: against a populated bronze the call raises
`FileExistsError` without `replace=True`, and `cps_00030` spans 64 years, so
omitting `years` rewrites all of them.

## The central idea: the parquet files are the source of truth for bronze

`_MANIFEST.yaml` (extraction's ledger) can list more extracts than were ever
actually parsed - or extracts that were parsed once and are now fully
redundant with what's already in bronze. Blindly reparsing every manifest
entry on every pipeline run means re-streaming potentially hundreds of
megabytes through the chunked parser for no reason (this repo's `cps_00030`
alone is 238MB / 64 years - a real, measured cost, not hypothetical).

So parsing doesn't trust the manifest's mere *presence* of an entry as a
signal to act on. It reads **the bronze parquet footers themselves**, via
`bronze_columns_by_year(bronze_dir, collection)` - one footer read per year,
no row data, milliseconds for the whole collection.

```
                     ┌──────────────────────────────┐
                     │   _MANIFEST.yaml               │  "what was ever extracted"
                     │   (extraction's ledger)         │  - may overstate what's parsed
                     └───────────────┬────────────────┘
                                     │ read, but not trusted for "already done"
                                     ▼
                     ┌──────────────────────────────┐
                     │   bronze_columns_by_year()      │  reads the parquet FOOTERS in
                     │   {year: set(columns)}           │  data/bronze/ipums/{collection}/
                     └───────────────┬────────────────┘
                                     │ "is this manifest entry's work
                                     │  already reflected here?"
                                     ▼
                     ┌──────────────────────────────┐
                     │   _entry_needs_processing()     │
                     └───────────────┬────────────────┘
                          skip ◄─────┴─────► process
```

`read_parquet_columns` reads the footer only, and drops any pandas index
field it finds there, so what comes back is the table's columns *as written*
rather than as they would materialize into a DataFrame.

### What a missing year means

`bronze_columns_by_year` leaves a file out of its result, with a
`ipums_bronze_parquet_skipped` warning naming why, in three cases:

| reason | file |
|---|---|
| `leftover_tmp_file` | a `.tmp.parquet` from a run that was hard-killed |
| `non_year_filename` | anything else whose stem is not an integer |
| `unreadable_parquet` | a real `{year}.parquet` whose footer will not parse - truncated, not Parquet, or carrying bad pandas metadata |

The third is why `read_parquet_columns` raises a dedicated
`ParquetUnreadableError` rather than letting a bare `ValueError`/`OSError`
escape: a corrupt file has to be distinguishable from an absent one.

So a year absent from the result means **unknown**, not "has no columns" -
and the distinction matters at exactly one place. `bronze_column_deviations`
can only report on years it was given, so a year whose parquet is unreadable
is not "deviating", it is simply not there. The repair path compensates
explicitly (see below); nothing else needs to.

### Why not the reference dictionaries

This used to read `bronze_coverage(dictionaries_dir)` instead, on the
reasoning that a year's dictionary is only written at the moment that year's
bronze is written, so its keys *are* bronze's columns. That holds only while
bronze never loses a column - and `save_variable_dictionary` unions on every
write, so a dictionary can only ever grow.

When `cps_00037` (the `cps2006_09s` Basic Monthly sample) overwrote
`2006.parquet` down to 11 columns, `2006.json` went on listing all 65 it had
ever seen. Coverage read the dictionary, concluded 2006 was complete, and
skipped it - so the damage was both invisible *and* unrepairable by re-running
the pipeline. Every other year agreed with its dictionary; 2006 was the only
one that didn't, which is exactly the year that needed help.

`bronze_coverage` still exists and is still correct about what it actually
reports - the high-water mark of what has been *documented* for a year - it
just isn't the record of what bronze *holds*.

## The column contract

Bronze for a collection is meant to be uniform: every year holds the same
columns. That expectation is what makes damage detectable, and it comes from
one of two places:

- **Declared** in `config/parsing/ipums/{collection}.yaml` under
  `expected_columns`, loaded by `src.config.parsing.load_expected_columns`.
  Editing it is a methodological decision, so it is versioned alongside
  `config/cleaning/` rather than derived.
- **Derived**, when no config exists: `modal_columns()` takes the most common
  column set across years. Modal rather than union or intersection so that one
  damaged year cannot move the target - 2006's 11 columns could not outvote 62
  years at 63.

The loader refuses a contract it cannot trust rather than quietly narrowing
one: `expected_columns` that is not a list of strings, is an empty list, or
names the same column twice all raise. So does a file whose `collection:` key
disagrees with the collection it is being loaded for, when the caller passes
`expect_collection` - a `cps.yaml` holding an ASEC-only contract would
otherwise mark every Basic Monthly year as damaged. A *missing* file stays
the one benign case: it means "derive from bronze", not "expect nothing".

Deriving has a failure mode worth knowing before a large repair: modal is a
vote, so if *most* years are damaged the modal set collapses onto the damaged
shape and the healthy years become the deviations. That is the case to
declare `expected_columns` explicitly for, and it is why CPS ships a
`cps.yaml` rather than relying on derivation.

A year is valid when its columns are a **superset** of the expected set.
A strict superset is fine: `cps_00036` pulled `OINCWAGE` for 1989 alone, so
1989 carries 66 columns against everyone else's 63, and that is a
variable-delta doing its job, not damage. Only *missing* columns count, which
`bronze_column_deviations()` reports as `{year: (missing, extra)}`.

## Guarding a year that already exists

A `new_samples` entry goes through `parse_to_bronze`, which writes each year
**whole** - so against a year that already has bronze it replaces every column
that year held. `_refusal_reason` stands in front of that. It is really two
independent gates - one on *overwriting at all*, one on *changing the shape* -
which between them produce four reasons to skip an entry:

| reason | when |
|---|---|
| `unknown_years` | no year could be parsed from the entry's sample ids, and bronze is not empty |
| `unknown_columns` | the entry's DDI could not be read, so its columns are a guess |
| `unexpected_columns` | the entry carries columns outside the expected set |
| `bronze_year_exists` | the year already has bronze and `replace=False` |

They are tested in that order, and the order is the point: an entry is only
judged on its columns once its years are known, and only judged on `replace`
once its columns are known to conform. The first three all fire *whatever*
`replace` says.

`_refusal_reason` is handed `coverage_years` - every year that already has a
readable bronze parquet, collection-wide - and intersects it with the entry's
`sample_years` itself. Those years come from the coverage map the run already
built from the parquet footers, not from a fresh `bronze_path(...).exists()`
sweep, so an entry is judged against the same picture of bronze that every
other decision in the run uses, including the rewrites earlier entries made.

`unknown_years` mirrors the fail-safe in `_entry_needs_processing`, but
inverted, and the asymmetry is deliberate. There, an entry with no parseable
year is always *processed*, because the risk is skipping work that was needed.
Here it is always *refused*, because the risk is overwriting a year nobody can
name. Both resolve the same ambiguity toward not losing data. The one
exception: when bronze is empty there is nothing to overwrite, so the entry
proceeds.

`replace` is the operator's switch for overwriting at all. The column gate is
deliberately **not** disabled by `replace`, and that distinction is what makes
repair possible: restoring 2006 requires `replace=True` so the full ASEC pull
may rewrite it, while that same run must still refuse `cps_00037`, which is
also a `new_samples` entry targeting 2006. It carries `HWTFINL`/`WTFINL`,
which no ASEC year has, so it is refused either way.

`unknown_columns` matters because an unreadable DDI leaves `entry_columns`
falling back to the *requested* variable list, which omits the flag and
technical columns IPUMS adds - so an entry could look conforming while
carrying anything. Unknown is not the same as safe.

`variable_delta` entries are exempt from all of this: they only ever add
columns, so they cannot reshape a year.

There is no flag to widen the expected set on the fly - pass
`expected_columns` explicitly instead. In particular `force` is *not* reused
for it: `force` travels in the manifest from the extractor, so one
force-redownload would disarm the guard permanently.

## Checking and repairing

```bash
uv run python -m src.jobs.ipums_bronze check  --collection cps          # exit 1 if any year deviates
uv run python -m src.jobs.ipums_bronze repair --collection cps --year 2006 [--dry-run]
```

`check_bronze_columns` is the read-only half: it resolves the contract the
same way the pipeline does and returns `{year: (missing, extra)}`, writing
nothing. The CLI exits 1 when that is non-empty, which is what makes it usable
as a guardrail in a scheduled job.

`repair_bronze_years` re-parses the targeted years with `replace=True`,
restricted by the `years` filter so only those years are rewritten. Every
manifest entry covering a targeted year is replayed in manifest order, last
writer wins, which is what lets the deltas re-fill a year the full pull just
reset. It then prunes each repaired year's dictionary down to the columns its
parquet really has (`prune_variable_dictionary` - the counterweight to the
union-on-write above), and only then verifies.

It refuses to start rather than doing something useless:

| `ipums_bronze_repair_skipped` reason | when |
|---|---|
| `empty_years_argument` | `years=[]` was passed - an explicit request to repair nothing |
| `no_bronze` | the collection has no readable parquet at all, so there is nothing to repair *from* |
| `all_years_conform` | no year deviates |

`ipums_bronze_repair_start` records the resolved contract and, as
`expected_source`, whether it was `declared` or `modal`. On a repair that
behaves unexpectedly that field is usually the first thing to check - a modal
contract derived from a mostly-damaged collection is the trap described above.

**What the guard does and does not catch during a repair.** The column gate
stays armed, so an entry carrying columns *outside* the expected set - the
kind that damaged the year in the first place - is still refused. An entry
whose columns are a strict *subset* is not refused: it conforms, it is simply
narrow, and replaying it can leave the year thinner than the entry before it
did. Nothing prevents that mid-run; only the post-repair check catches it.

That check raises `RuntimeError` rather than logging - a repair that silently
half-worked is worse than one that fails loudly - and it counts two distinct
failures as "still deviating":

- the year is present but still missing expected columns, and
- the year is **absent from the re-read** entirely, via
  `targets - repaired.keys()`.

The second is the case that would otherwise slip through: an unreadable or
never-written parquet is not *in* `bronze_column_deviations`' output, so
checking deviations alone would certify it as repaired. `_deviation_detail`
renders it as `no bronze file`, against `missing [...], extra [...]` for the
first case, and the message names the expected set alongside the two
explanations that actually apply - no extract on disk covers the year, or a
later manifest entry rewrote it narrower.

The 2006 repair, as it actually ran:

```
cps_00030  new_samples     47 cols ⊆ expected, replace=True  → writes 2006 = 47
cps_00037  new_samples     HWTFINL/WTFINL ∉ expected         → REFUSED
cps_00033  variable_delta  2025 only                          → outside years filter
cps_00034  variable_delta  merges 10                          → 2006 = 57
cps_00035  variable_delta  merges 6                           → 2006 = 63  ✓
cps_00036  variable_delta  1989 only                          → outside years filter
                                                                 (1989 keeps OINCWAGE)
```

## The per-collection pipeline loop

`pipelines.ipums_parse_pipeline.parse_ipums_extracts`, per collection:

```
entries = manifest entries with .dat.gz + .xml still on disk,
          "new_samples" entries ordered before "variable_delta" ones
          (a delta merge needs an existing bronze file to merge into)

coverage = bronze_columns_by_year(bronze_dir, collection)   # {year: {columns...}}
expected = _resolve_expected(coverage, expected_columns)    # frozen for the whole run

for entry in entries:
    sample_years = {parse_sample_year(s) for s in entry.samples}   # best-effort
    variables    = set(entry.variables)
    force        = entry.metadata.get("force", False)

    if years_filter is not None and sample_years:
        sample_years &= years_filter
        if not sample_years:
            log "ipums_parse_entry_skipped" reason=outside_years_filter; continue

    if _entry_needs_processing(coverage, sample_years, variables, force) is False:
        log "ipums_parse_entry_already_covered"; skip
        continue

    if entry.request_kind == "variable_delta":
        touched_paths = merge_variables_into_bronze(..., force=force, years=years_filter)
    else:
        reason = _refusal_reason(entry_columns=variables,
                                 coverage_years=set(coverage),     # years bronze really has
                                 sample_years=sample_years,        # years the ids claim
                                 expected=expected, ...)
        if reason is not None:
            log "ipums_parse_entry_refused" reason=reason; continue
        try:
            touched_paths = parse_to_bronze(..., replace=replace, years=years_filter)
        except FileExistsError:                                    # ids under-reported the span
            log "ipums_parse_entry_refused" reason=bronze_year_unseen; continue

    if not touched_paths:        # filtered out every year it carried - nothing happened,
        continue                 # so don't write a dictionary or claim a completed entry

    touched_years = [int(p.stem) for p in touched_paths]           # ground truth
    build_and_save_variable_dictionary(ddi_path, dictionaries_dir, touched_years, force=force)
    for year in touched_years:                                     # seen by later entries this run
        if entry.request_kind == "variable_delta":
            coverage[year] |= variables      # a merge adds
        else:
            coverage[year]  = variables      # a wholesale rewrite REPLACES

report bronze_column_deviations(bronze_columns_by_year(...), expected)
```

`expected` is computed once, before the loop, so an entry that shrinks a year
partway through cannot move the target the later entries are judged against.

The asymmetric coverage update at the bottom is what makes the pipeline
**convergent**. Unioning there (as it did originally) meant that after a
wholesale rewrite shrank a year, every delta that had once filled it still
read as already-covered - so a repair would stop at 47 of 63 columns and
report success. Assigning on a rewrite lets those deltas see themselves as
uncovered again and re-merge.

Four more things worth calling out:

- **The skip check uses `parse_sample_year` (best-effort regex), but the
  dictionary write afterward uses the *actual* years returned by
  `parse_to_bronze`/`merge_variables_into_bronze`** (the real `YEAR` column
  values in the data, via each output parquet's filename stem). The regex is
  only ever used to decide *whether* to do work, never to decide *what* was
  done - that always comes from the data itself.
- **`_entry_needs_processing` fails safe.** If no year could be parsed from
  any of an entry's sample ids, it returns `True` (process it) rather than
  silently skipping something it can't reason about.
- **`force` on a manifest entry always wins over the *coverage* check.** A
  forced re-pull exists specifically to correct something already in bronze,
  so it must never be skipped just because coverage already shows that
  year/variable as present - that coverage is usually exactly *why* it was
  forced. It does **not** disarm the column guard; see above.
- **The years filter narrows `sample_years` before the coverage check**, so a
  filtered run judges an entry only on the years it is allowed to touch. An
  entry with no parseable year keeps an empty set and stays fail-safe.

```python
def _entry_needs_processing(coverage, sample_years, variables, force=False) -> bool:
    if force:
        return True
    if not sample_years:
        return True
    return not all(variables <= coverage.get(year, set()) for year in sample_years)
```

## Two ways to touch bronze

### `parse_to_bronze` - fresh years

```
.dat.gz + .xml
      │
      ▼
read_ipums_ddi(ddi_path) ──► Codebook
      │
      ▼
read_microdata_chunked(codebook, data_path, chunksize=100_000)
      │
      ▼  for each ~100k-row chunk:
      │    check_no_duplicate_columns(chunk)
      │    group the chunk by YEAR
      │    for each year present in this chunk:
      │      not in `years` filter?      → skip this year entirely
      │      first time this year seen?  → _open_year_writer(...):
      │                                       already has bronze and not replace?
      │                                         → FileExistsError
      │                                       else a _YearWriter holding the open
      │                                       pq.ParquetWriter + its tmp/out paths
      │      write_table(this year's rows)
      │
      ▼
_close_writers(..., completed=<did the stream finish?>):
    close every writer, collecting rather than swallowing any close failure
    not completed, OR any writer failed to close?
                            → unlink every .tmp.parquet staged this run
    any close failure?      → RuntimeError naming the collection and bronze dir
no rows at all?         → ValueError("IPUMS extract has no rows")
rows, but none for the requested years? → log + return []   (a no-op, not a failure)
rename every {year}.tmp.parquet → {year}.parquet, only after every writer closed cleanly
return sorted list of the final year parquet paths touched
```

The empty-extract and filtered-out-everything cases must stay distinct:
`total_rows` counts rows *before* the years filter, so a genuinely empty
`.dat.gz` still raises while `years=[1999]` against a real extract simply
returns `[]`.

`parse_to_bronze` **raises** `FileExistsError` here rather than
warn-and-skipping - it is a library function, and refusing loudly is the right
default for a caller that asked to write a year it did not know was there.

The pipeline is the caller that *does* know better, so it catches that
exception and turns it into a skip, logging `ipums_parse_entry_refused` with
`reason="bronze_year_unseen"`. The name says what happened: the pre-flight
guard had already cleared this entry, so hitting the writer's guard anyway
means the years the sample ids advertised disagreed with the extract's actual
`YEAR` values. One entry whose sample ids under-report its span must not abort
parsing for the whole collection, which is the same reasoning the malformed-
manifest-entry convention rests on.

So `bronze_year_unseen` and `bronze_year_exists` mean genuinely different
things, and it is worth reading the log carefully: the second is the guard
working from what the manifest claimed, the first is the manifest's claim
being wrong.

Memory-bounded by design: only one chunk is ever fully materialized, and
each year's `ParquetWriter` streams row groups rather than buffering the
whole year. A chunk can span several years, and a year's rows can arrive
across several non-adjacent chunks - both are handled by keeping the writer
map open across the whole loop, not per-chunk.

**Why the `.tmp.parquet` + rename dance:** `pq.ParquetWriter` on an existing
path truncates it immediately on open. Earlier, writing straight to
`{year}.parquet` meant a process death mid-run (this was hit for real, via
an OOM on a 238MB/64-year pull) left already-good bronze content for
whichever years' writers were open destroyed - replaced by a partial,
unreadable file - even though years not yet reached were untouched.
Writing to a `.tmp.parquet` sibling and only renaming it over the real file
once every writer in the run has closed successfully means a crash now
leaves the *previous* `{year}.parquet` exactly as it was.

The `.tmp.parquet` files staged by a run that does not reach the end are
unlinked on the way out, covering the refusal path and a mid-stream exception
with one mechanism. A writer that fails to *close* counts too: its row groups
may never have been flushed, so the file it leaves is unusable even though the
stream itself finished. Nothing downstream can tell a half-written
`.tmp.parquet` from a complete one, and only a rename promotes one, so leaving
them for the next run to trip over bought nothing.

A leftover that does survive (a hard kill, where no `finally` runs) is
reported by `bronze_columns_by_year` as `ipums_bronze_parquet_skipped` with
`reason="leftover_tmp_file"` rather than silently ignored.

### `merge_variables_into_bronze` - new (or corrected) variables on years that already exist

```
delta .dat.gz + .xml  (e.g. cps_00034: 63 samples, 6 variables)
      │
      ▼
parse_to_bronze(..., bronze_dir=<temp staging dir>,     ← reuses the exact
      │          replace=True, years=years)                 same chunked/
      ▼                                                     per-year logic
staged/{year}.parquet, one per year in the delta
      │            (the `years` filter is forwarded here, so repairing one
      │             year does not re-merge the delta into all 63 of them)
      │
      ▼  for each staged year:
      │    out_path = bronze/{collection}/{year}.parquet
      │    missing? → RuntimeError (coverage said this year exists; it should)
      │    no columns in common with merge_keys? → RuntimeError
      │
      │    staged_df   = read staged/{year}.parquet
      │    existing_df = read bronze/{collection}/{year}.parquet
      │
      │    add_columns = new_variables present in staged_df, filtered by:
      │                    not force → only columns bronze DOESN'T have yet
      │                    force     → all of them (may already exist)
      │    overlap_columns = (force only) the subset of add_columns that
      │                       DO already exist in bronze
      │    new_columns     = add_columns minus overlap_columns
      │
      │    merged = existing_df.merge(staged_df[merge_keys + new_columns],
      │                                on=merge_keys, how="left")
      │    row count changed? → RuntimeError (should never happen)
      │
      │    overlap_columns present? → indexed .loc update of just those
      │      columns' values for matching merge-key rows, dtype-cast to
      │      match, instead of a second left-join (a left-join would NaN
      │      out any row whose key isn't in this narrower staged extract -
      │      e.g. other months/samples sharing this year's bronze file)
      │
      │    restore original column order; write to {year}.tmp.parquet;
      │    rename over {year}.parquet (same atomicity as parse_to_bronze)
      │
      ▼
return list of updated year paths
```

`merge_keys` defaults to `("YEAR", "MONTH", "SERIAL", "PERNUM")` - IPUMS
CPS's standard person-level identifiers, always present because IPUMS
auto-includes a set of "preselected" technical variables on every CPS
extract regardless of what's requested (confirmed empirically: requesting
just `["AGE"]` still returns `YEAR, SERIAL, MONTH, HWTFINL, CPSID, PERNUM,
WTFINL, CPSIDP, CPSIDV, AGE, QAGE`-style columns). No code has to inject
merge keys into a delta request - they're just always there.

**Without `force`:** a variable already present in bronze is left alone -
`merge_variables_into_bronze` only ever *adds* columns bronze doesn't have.
**With `force=True`:** a variable already present gets its values replaced,
row-by-row on the merge keys, for exactly the rows the (typically narrower)
forced extract covers - every other existing column, and every row not
covered by the forced extract, is untouched. This is what makes "I pulled
`AGE` wrong the first time, re-pull just that" possible without a full
re-parse of everything else in that year.

## The variable dictionary: merge-on-write, with drift detection

`save_variable_dictionary(variable_dictionary, dictionaries_dir, year,
force=False)` unions onto whatever's already saved for that year rather than
replacing it - and, symmetrically with the bronze merge above, treats a
collision between the incoming definition and what's on disk as **drift to
be logged**, not silently ignored:

```python
for name, new_entry in variable_dictionary.items():
    old_entry = existing.get(name)
    if old_entry is not None and old_entry != new_entry:
        log.warning("ipums_variable_definition_drift", ..., overwritten=force)

merged = {**existing, **variable_dictionary} if force \
    else {**variable_dictionary, **existing}   # old wins by default
```

By default, an existing variable definition is never silently changed - a
variable's on-disk description should be stable. With `force=True`, the new
definition wins, mirroring `merge_variables_into_bronze(force=True)`: a
deliberate forced refresh should be able to correct a bad prior *definition*
(e.g. a wrong label picked up from a bad extract), not just bad prior bronze
*values*.

`build_and_save_variable_dictionary(ddi_path, dictionaries_dir, years,
force=False)` builds the dictionary once from one DDI codebook and saves
(merges) it into every year in `years`, forwarding `force` unchanged.

### The one thing that removes entries

Union-on-write means a dictionary can describe a variable its year no longer
has - which is how `2006.json` kept listing `HWTFINL`/`WTFINL` after the
parquet was rebuilt without them. `prune_variable_dictionary(dictionaries_dir,
year, columns)` drops the entries outside `columns` and logs
`ipums_variable_dictionary_pruned` with what it removed.

It is called **only from the repair path**, never during normal parsing: the
union is the right behaviour for a year accumulating variables across
extracts, and pruning on every write would fight it.

## Legacy files

Before this per-year scheme, dictionaries were saved as
`{collection}_{extract_id:05d}.json` (e.g. `cps_00030.json`). `bronze_coverage`
only recognizes purely-numeric `{year}.json` stems, so old files like that
are silently ignored - not read, not written to, just inert. Their content
is still valid JSON in the same shape, just unreachable by the new
year-keyed lookup. They haven't been deleted automatically since they're
real repo data, not build output.

## Validation

`schemas.bronze.ipums_long` validates at two grains.

**One DataFrame:** `validate_ipums_long` runs on every parsed chunk and on the
full single-shot `parse_ipums_extract` path - no duplicate column names, no
empty result. It checks no specific columns, because which columns a single
extract carries depends entirely on what was requested.

**The whole collection:** `modal_columns` and `bronze_column_deviations`
answer "does every year hold the same columns", which no single DataFrame can
express. They name no column themselves - the expected set is either declared
in `config/parsing/` or derived - so the contract lives in one place, and that
place is not Python. This is deliberately unlike `bea_long.py`, which hardcodes
its 4-column `BeaLongRow` contract; IPUMS bronze is 63 columns that change with
every extract request, so hardcoding would mean editing code to add a variable.

## Worked example: correcting a bad pull without losing anything else

This is the scenario the `force` machinery exists for, and it's covered
end-to-end by `test_force_refresh_replaces_variable_without_clobbering_other_columns`:

```
Starting state (bronze/reference already populated by a normal run):
  bronze/cps/2006.parquet:  YEAR, MONTH, AGE, SEX  (AGE pulled wrong: 25, 30)
  reference/cps/2006.json:  {"AGE": {"Description": "Age (bad pull)"}, "SEX": {...}}

A forced, AGE-only re-pull is submitted and lands in the manifest as:
  request_kind="variable_delta", variables=("AGE",), force=True

parse_ipums_extracts runs:
  _entry_needs_processing(...) → True unconditionally, because force=True
    (even though coverage already shows 2006 has AGE - that's expected;
    it's *why* this was forced)
  merge_variables_into_bronze(..., new_variables=["AGE"], force=True):
    AGE already in bronze → overlap_columns=["AGE"], new_columns=[]
    row count check passes (same rows, just updating one column's values)
    AGE values replaced in place for the matching rows; SEX untouched
  build_and_save_variable_dictionary(..., force=True):
    AGE's on-disk description differs from the new pull's →
      logs "ipums_variable_definition_drift", then overwrites it

Result:
  bronze/cps/2006.parquet:  YEAR, MONTH, AGE, SEX  (same 4 columns, AGE
    values corrected, SEX byte-for-byte untouched, same row count)
  reference/cps/2006.json:  AGE's Description now reads "Age (corrected)"
```

The regression this behavior guards against: a naive re-implementation could
easily route a forced single-variable pull through the same code path as a
fresh full pull, silently dropping every column the forced extract didn't
happen to also fetch.
