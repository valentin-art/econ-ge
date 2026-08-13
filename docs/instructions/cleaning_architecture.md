# Cleaning Pipeline Architecture — Implementation Instructions

## Purpose of this document

Instructions for Claude Code to plan and implement a **microdata cleaning pipeline layer** inside the `econ-ge-model` project. The layer must support CPS ASEC (primary target) with a path to PSID and other microdata sources without architectural change.

This document specifies the target architecture; it does **not** specify all cleaning steps themselves — those are added incrementally. The goal here is the **fundamental layer** on which steps will be built.

Before writing code, read the existing project structure (`src/`, `config/`, `tests/`) and align the plan with existing conventions: Polars, patito for schema validation, pydantic-settings for configuration, hexagonal architecture (business logic infrastructure-agnostic), `_prov_`-style provenance columns, per-year switchable stages driven by external config.

---

## 1. Architectural principles (non-negotiable)

These principles were established through prior design discussion. Do not violate them without raising the trade-off explicitly.

1. **Business logic is infrastructure-agnostic.** Cleaning code lives in `src/cleaning/`. It receives `DataFrame` and configuration objects; it knows nothing about file paths, Postgres, Dagster, or settings. The same functions run in notebooks, tests, CLI jobs, and Dagster assets unchanged.

2. **Two independent configuration objects, deliberately separated:**
   - **`Settings`** (existing, pydantic-settings): infrastructure — paths, connections, secrets, log levels. Lives on process. Read from env. Changes freely without invalidating scientific results.
   - **`CleaningContext`** (new, this layer): methodology — per-year tables, crosswalks, source profiles. Lives on run. Read from versioned YAML + crosswalk files. Every change is a methodological decision requiring code review and a note in `docs/methodology/`.

   These objects **must not know about each other**. Point of gluing is in `jobs/` or Dagster assets, not in either object.

3. **Provenance is a first-class output**, not a side-effect log. Every cleaning decision must leave a trace either in the DataFrame (as `_prov_`-prefixed columns for row-level decisions) or in a structured report (for aggregated decisions).

4. **Per-year and per-source switchable behavior is data, not code.** Behavior variations across years/sources are expressed as tables in configuration, not as `if` statements in code. This makes methodology a redactable artifact.

   *Narrow exception, established when building the pre-1976 weeks-worked bridge*: this governs behavior an operator might legitimately want to reconfigure without a code change (a topcode strategy, an allocation-flag policy). It does not require a step to source a *statistical fit* from precomputed config when the fit is cheap to compute from the DataFrame it already has and the boundary it switches on is a structural fact about the source, not a tunable policy - e.g. `bridge_weeks_pre_1976` (`src/cleaning/custom_functions.py`) does `pl.when(pl.col("YEAR") < 1976)` and fits its sex/race/bracket group means from the same DataFrame's own 1976-78 rows, rather than reading a frozen `WeeksBridgeConfig` table. `WKSWORK1` simply not existing before 1976 isn't something a YAML edit should be able to change. If a step ever needs to compute a fit from data *outside* the DataFrame it's given (a separate reference extract), that's the signal to go back to a config-table sub-context instead.

5. **Immutability of context.** `CleaningContext` is frozen after construction. Steps read it but cannot write to it. Inter-step communication happens through the DataFrame (flag columns) or step reports, never through the context.

6. **Validate configuration at construction, not at use.** Errors in per-year tables, crosswalks, or missing keys must fail when `CleaningContext.from_config()` is called — before running the pipeline on millions of rows.

---

## 2. Hybrid architecture — the design decision

The chosen architecture is **hybrid**:

- **Top level: methodological steps as first-class objects.** Each step is a class corresponding to one methodological decision (e.g., `TopcodeAdjuster`, `EducationBridge`, `AllocatedEarningsFilter`). Steps have names, known required/produced columns, per-year configuration, and produce provenance.
- **Internal level: Polars expressions as atoms.** Inside each step's `apply` method, use native Polars expressions (`df.filter(...)`, `df.with_columns(...)`, `df.join(...)`) directly. Do **not** build a custom atomic layer over Polars — Polars already provides optimized, typed, composable atomic operations.
- **Optional shared expression library.** If the same Polars expression logic reappears in two or more steps, extract it to `src/cleaning/expressions.py` as a plain function. Not an abstraction layer — just a shared utility module. Do not create this file preemptively; create it when the first duplication appears.

**What this architecture explicitly rejects:**
- A universal `Filter`/`Map`/`Drop` atomic step system with steps composed from atoms. Reason: it dilutes provenance (atoms don't know which methodological decision they belong to), pushes configuration into lambdas or a custom DSL, and duplicates functionality Polars already provides natively.
- sklearn-compatible interfaces (`fit`/`transform`/`fit_transform`, `get_params`/`set_params`, ColumnTransformer, FeatureUnion). These add ceremony without value for this use case.

**Registry/factory-from-YAML: built ahead of the original schedule.** This document originally deferred a step registry until "15+ steps and demonstrated need." An explicit, later decision pulled it forward at ~8 steps, after the trade-off was shown and chosen deliberately - the goal became letting a non-advanced user reconfigure a pipeline's step list/order/params by editing YAML alone, not just per-year tables inside an existing step. Implemented as:
- `src/cleaning/steps/registry.py`: `STEP_BUILDERS: dict[str, Callable[..., Step]]`, an **explicit whitelist dict** (every standard `Step` class is already callable as `StepClass(name=..., **kwargs)`), not `globals()`/dynamic class-name lookup - a fixed, greppable mapping, auditable in one file.
- `Pipeline.from_config(config_path, registry)` (`src/cleaning/base.py`): builds a `Pipeline` from a `steps:` list of `{name, type, ...kwargs}` blocks. `registry` is passed in, not imported, so `base.py` keeps zero dependency on `steps/*` (every step imports *from* `base.py`, never the reverse). Unknown `type`, or kwargs that don't match the target step's constructor, raise `ValueError` naming the file/step/type at call time (principle 6) - not mid-`apply()`.
- See §6.4 for the YAML shape and the `FunctionStep`/`function:` escape hatch this unlocked for custom, one-off logic.

This does **not** reopen the door on the other rejected patterns below - the registry is a fixed dict of real `Step` classes/builders, not a name→arbitrary-behavior DSL.

---

## 3. Directory layout

Create the following structure inside `src/`:

```
cleaning/
    __init__.py
    base.py              # Step protocol/base, Pipeline (+ from_config), StepReport, RunReport
    context.py           # CleaningContext + sub-contexts, from_config factory
    custom_functions.py  # plain (df, context) -> df functions wired via `type: FunctionStep` YAML blocks
    steps/
        __init__.py
        registry.py       # STEP_BUILDERS: dict[str, Callable[..., Step]], used by Pipeline.from_config()
        function_step.py  # FunctionStep: wraps a plain function as a Step, explicit name/required/produced
        # one file per step, added incrementally
    # expressions.py     # created later, when first duplication appears
```

Companion structure:

```
config/cleaning/
    cps/
        pipeline.yaml         # steps list assembled via Pipeline.from_config() + STEP_BUILDERS - see §6.4
        topcode.yaml
        allocation.yaml
        education_bridge.yaml
        source_profile.yaml
    # psid/ added later
    crosswalks/
        occ_to_quadrant.csv       # or .parquet
        education_bridge.csv
        # further crosswalks as needed

tests/unit/cleaning/
    test_cleaning_base.py    # Step contract, Pipeline composition
    test_context.py          # Context construction, validation, immutability
    test_pipeline_from_config.py  # Pipeline.from_config() against STEP_BUILDERS
    test_custom_functions.py      # plain-function step logic, synthetic data
    steps/
        # one test file per step
    fixtures/            # small synthetic DataFrames + fixture config/ dir for unit tests
```

Do not create empty directories or placeholder files preemptively. Each directory appears when the first real file in it is created.

---

## 4. Core contracts (v1 minimum)

### 4.1 `Step`

A single type, not two. The `Filter`/`Transformer` distinction is expressed in the step's report (`n_out < n_in` implies filtering behavior), not in the type system.

Required contract:

- **`name: str`** — unique within a pipeline. Used in reports, log messages, and (as prefix or suffix) in provenance column names. Do not rely on `type(step).__name__` — the same class may be instantiated multiple times with different configurations.
- **`required_columns: frozenset[str]`** — static class attribute. Columns that must exist in the input DataFrame. Enables static compatibility validation by the Pipeline before running.
- **`produced_columns: frozenset[str]`** — static class attribute. Columns that will be added or modified. Also enables static validation.
- **`is_idempotent: bool`** — declares whether `apply(apply(df)) == apply(df)` holds. Used later by Dagster for safe re-materialization. Most cleaning steps should be idempotent by design; document why if not.
- **`apply(df: pl.DataFrame, context: CleaningContext) -> tuple[pl.DataFrame, StepReport]`** — the operation. Pure function of inputs. No side effects, no state mutation.

If a step needs source-awareness (behavior differs between CPS and PSID), read the current source from `context.source_profile.kind` and branch internally. Alternatively — and preferably when differences are structural — build separate pipelines per source; do not make one step handle every source.

Use `typing.Protocol` or an abstract base class — implementer's choice, but be consistent. Protocol is lighter and matches Python idiom; ABC gives better error messages on incomplete subclasses. Implemented as an ABC (`src/cleaning/base.py`).

**`FunctionStep` — a lighter-weight `Step` for a plain function.** For an operation simple enough not to need its own class and file (e.g. `age_last_year`, `bridge_weeks_pre_1976` in `src/cleaning/custom_functions.py`), `FunctionStep(name, fn, required_columns, produced_columns)` wraps a plain `(df, context) -> df` function so only `apply()`'s body moves out of a subclass - `name`/`required_columns`/`produced_columns` are still required constructor arguments, so `Pipeline.validate_compatibility()` and provenance work exactly as for any other `Step`. This is deliberately *not* the rejected `Step.from_function(fn)` pattern in §9 (no anonymous steps): the difference is that every constructor argument stays explicit. Promote a `FunctionStep` to a dedicated `Step` subclass once it needs richer provenance than a generic row-count-delta `StepReport`, or once its logic stops being a first draft.

A function wired in this way can be, and often should be, **agnostic about which step produced the columns it reads** - it declares them via `required_columns` (checked by `Pipeline.validate_compatibility()`/`apply()`'s runtime guard) without needing to know or care whether an upstream `Step` produced them, or whether they arrived as already-known input columns. `bridge_weeks_pre_1976` is the example: it reads `FEMALE`/`RACE` without deriving sex/race groups itself, on the assumption some other, possibly not-yet-built step supplies them.

### 4.2 `StepReport`

A structured record of what the step did. Minimum fields:

- `step_name: str`
- `n_in: int`
- `n_out: int`
- `dropped_reason_counts: dict[str, int]` — e.g., `{"self_employed": 1234, "allocated_earnings": 567}`. Empty if no rows dropped.
- `branches_taken: dict[str, int]` — for source-aware or per-year steps, count of rows per branch. E.g., `{"akk_multiplier": 40000, "ipums_native": 60000, "skip": 10000}`. Empty if step has no branching.
- `warnings: list[str]` — non-fatal issues, e.g., "QINCWAGE unavailable for year 1987, allocation filter skipped for 45 rows".
- `duration_seconds: float` — optional but useful for profiling.

Use a `@dataclass(frozen=True)` or pydantic model.

### 4.3 `RunReport`

Simply a list of `StepReport` plus run-level metadata:

- `steps: list[StepReport]`
- `context_hash: str` — hash of serialized CleaningContext used (for reproducibility)
- `pipeline_name: str`
- `started_at`, `finished_at`

**`git_commit` was cut from `RunReport`, not just deferred to the caller as originally planned.** Decided explicitly: code/config versioning is MLflow's job once that integration lands, not `RunReport`'s - keep it simple now rather than add a field only `Settings`-aware callers can fill in. Today `context_hash` alone identifies a run's `CleaningContext`; it does **not** cover `pipeline.yaml` (step list/order/params, §6.4) now that those live outside `CleaningContext` - a known, accepted reproducibility gap, left for the same future MLflow work rather than a second hash field. `pipeline.yaml` is source-controlled like code and is version-identified by git history the same way.

### 4.4 `Pipeline`

Minimal composition. ~30 lines of real code.

- **`__init__(steps: list[Step], name: str, validate_between_steps: bool = False, known_input_columns: frozenset[str] = frozenset())`**
- **`apply(df: pl.DataFrame, context: CleaningContext) -> tuple[pl.DataFrame, RunReport]`** — linear loop: call each step, accumulate reports, optionally run patito validation between steps.
- **`validate_compatibility() -> list[str]`** — static check before running: for each consecutive pair, verify that columns produced (or preserved) by earlier steps satisfy `required_columns` of later steps. Return a list of issues; empty list means valid. Call this once at pipeline construction time in production code.
- **`from_config(config_path: Path, registry: Mapping[str, Callable[..., Step]]) -> Pipeline`** (classmethod) — builds a `Pipeline` from YAML instead of manual construction. See §2's registry section and §6.4 for the YAML shape.

No `fit`, no `fit_transform`, no branching, no ColumnTransformer analog. If branching is needed (per-source pipelines), express it at the caller level (`pipelines_by_source[src].apply(df_src, ctx)`).

### 4.5 `CleaningContext`

Immutable pydantic model (`model_config = ConfigDict(frozen=True)`), composed of sub-contexts by domain. Not a flat dict.

Minimum sub-contexts to define in v1 (empty/skeleton is fine for those without steps yet; add fields as steps are added):

- **`TopcodeConfig`** — `per_year: dict[int, Literal["akk_multiplier", "ipums_native", "skip"]]`, plus multiplier values, etc.
- **`AllocationConfig`** — `per_year_flag_available: dict[int, dict[str, bool]]` (indexed by flag name like `QINCWAGE`), strategy for years without flag (`"skip"` | `"flag_only"`).
- **`WeeksBridgeConfig`** — kept as a field-less skeleton, deliberately: the pre-1976 `WKSWORK1`/`WKSWORK2` bridge turned out not to need a config sub-context at all. It's a self-contained `FunctionStep` (`bridge_weeks_pre_1976`, §4.1/§6.4) that fits its sex/race/bracket group means from the DataFrame's own 1976-78 rows at runtime, conditional on `YEAR` — see principle 4's exception in §1. Not every step needs a sub-context; add fields here only if a future step actually needs a config-driven (not data-driven) bridging rule.
- **`EducationBridgeConfig`** — target scheme (4 levels), pre-1992 mapping table reference, post-1992 mapping table reference.
- **`OccMappingConfig`** — path key to the occ→quadrant crosswalk (loaded as Polars DataFrame at context construction), border-of-quadrant thresholds.
- **`SourceProfile`** — `kind: Literal["ipums_cps_asec", "nber_mw", "raw_asec_march", "psid"]`, `available_flags: frozenset[str]`, `default_topcode_branch`, `notes: str`.
- **`RunMetadata`** — `run_id: str`, `config_version: str`, `random_seed: int | None`.

Additional fields to hold loaded crosswalks:
- `occ_to_quadrant: pl.DataFrame` — loaded from CSV/Parquet at construction, immutable by convention.
- Other crosswalks similarly.

Factory:

```python
CleaningContext.from_config(
    config_dir: Path,
    source: SourceKind,
    crosswalks_dir: Path,
    overrides: dict | None = None,
) -> CleaningContext
```

`config_dir` and `crosswalks_dir` are passed **explicitly** by the caller (which reads them from `Settings`). `CleaningContext` never imports or references `Settings`. This is the point where the two-object separation is enforced.

Validate on construction — pydantic does this natively for field types, but add custom validators for cross-field constraints (e.g., all years covered by topcode table match years available in source profile).

Serialize (for `context_hash`) by dumping the pydantic model to a stable JSON representation and hashing. DataFrame fields (crosswalks) should be represented by content hash, not full serialization, to avoid huge hashes.

---

## 5. Provenance conventions (fix these now)

This is the one contract that is expensive to change later, because Silver Parquet files accumulate with these columns embedded. Fix now.

### 5.1 Column naming

- **All provenance columns** carry a common prefix. Use **`_prov_`**.
- Format: `_prov_{step_name}_{aspect}`. Examples: `_prov_topcode_branch`, `_prov_allocation_dropped_reason`, `_prov_weeks_bridge_rule`, `_prov_educ_bridge_source_year`.
- Prefix enables trivial filtering on export: drop all provenance columns for model input via a regex or `pl.selectors.starts_with("_prov_")`.

### 5.2 Types

Prefer categorical/string types for branch labels (readable in logs and Parquet inspection tools). Prefer boolean or small-int types for flag columns (dropped/not-dropped, imputed/not-imputed).

### 5.3 Silver schema

The patito Silver schema **must explicitly acknowledge** provenance columns as valid. Do not treat them as unexpected columns to be dropped by validation.

### 5.4 Documentation

Every step class documents in its docstring:
- Which `_prov_` columns it produces
- The domain of values for each
- The semantics (what does `_prov_topcode_branch = "skip"` mean?)

This is the canonical spec. Anyone reading a Silver Parquet later can trace column semantics from column name back to step docstring.

---

## 6. Configuration format

### 6.1 YAML for per-year and per-source tables

Human-editable, version-controlled, reviewable in PRs.

Example (`config/cleaning/cps/topcode.yaml`):

```yaml
default_branch: akk_multiplier
per_year:
  1962: akk_multiplier
  1963: akk_multiplier
  # ...
  1996: ipums_native
  # ...
akk_multiplier_value: 1.5
```

Load with a YAML library (PyYAML is fine; `ruamel.yaml` if round-tripping is ever needed — probably not).

### 6.2 CSV/Parquet for crosswalks

Crosswalks are **data**, not parameters. Store as CSV or Parquet under `config/cleaning/crosswalks/`, load into Polars DataFrame at context construction, hold as immutable field in `CleaningContext`.

Do not put crosswalks in YAML — they lose types, are hard to edit at scale, and version poorly.

### 6.3 Overrides

`CleaningContext.from_config(..., overrides=...)` accepts a dict that deep-merges over the YAML-loaded config. Use for sensitivity analysis and experimentation — not for production runs. Log any override applied into `RunReport`.

### 6.4 `pipeline.yaml` — step assembly from YAML

`config/cleaning/cps/pipeline.yaml`, loaded by `Pipeline.from_config()` (§2, §4.4):

```yaml
name: cps_universe_and_topcode
known_input_columns: [AGE, CLASSWLY, YEAR, WKSWORK1, WKSWORK2, FEMALE, RACE]
validate_between_steps: true
steps:
  - name: age_band_filter
    type: BandFilter
    column: AGE
    min_value: 16
  - name: weeks_worked_bridge
    type: FunctionStep
    function: src.cleaning.custom_functions.bridge_weeks_pre_1976
    required_columns: [YEAR, WKSWORK1, WKSWORK2, FEMALE, RACE]
    produced_columns: [WEEKS_WORKED]
```

Conventions:
- `name:` before `type:` within a step block (GitHub Actions' `- name: ... / uses: ...` steps-list precedent) - purely readability, key order has no effect on `Pipeline.from_config()` (it reads the parsed dict via `.pop()`/`**kwargs`).
- `known_input_columns` lists every column some step needs that no earlier step produces - the same role `AGE`/`CLASSWLY` already played manually, now declared once per pipeline instead of per test.
- **`type: FunctionStep` + a `function:` dotted import path is the custom-step escape hatch, reachable from YAML.** `function:` is resolved via `importlib.import_module()` + `getattr()` (`src/cleaning/steps/registry.py::_build_function_step`) - it names the function's real, greppable location, exactly what a Python `import` statement would resolve, not a hand-maintained string→behavior alias table. That distinction is why this isn't the "DSL for step definitions in YAML" §9 rejects: **logic stays in a real Python function in `custom_functions.py`; YAML only wires it in** (`name`, `required_columns`, `produced_columns`, optional static `params:` bound via `functools.partial`). Only ever point `function:` at trusted, version-controlled modules in this repo.

---

## 7. Testing strategy

Three levels, each addressed by the architecture:

### 7.1 Step unit tests (fast, isolated)

For each step, construct a **mock CleaningContext manually** with only the sub-context that step needs. Do not read YAML or crosswalks from disk. Feed the step a small synthetic Polars DataFrame (5–20 rows covering edge cases). Assert on output DataFrame and StepReport.

The fact that the sub-context is small and independently constructible is precisely the tested property of the two-object separation. If you find yourself needing full Settings or full CleaningContext for a step's unit test, the step is doing too much or leaking a dependency.

### 7.2 Pipeline integration tests (synthetic data)

Construct `CleaningContext` from real YAML files (fixture configs under `tests/unit/cleaning/fixtures/config/`). Assemble a small pipeline of real steps. Run on a synthetic DataFrame with known properties (each row hits one edge case). Assert on `RunReport` invariants (counts, branches taken, no unexpected warnings).

### 7.3 Real-data invariant tests (slow, opt-in)

Run pipeline on a real single-year ASEC file. Assert **properties**, not values: `n_out > 0`, no negative FTFY-equivalents, all `_prov_` columns present with values in expected domains, `StepReport.warnings` empty or matching known expected warnings.

Mark these tests to run on-demand (`pytest -m realdata`), not in every CI run.

---

## 8. Implementation plan (order of work)

Execute in this order. Each stage is independently valuable and stops at a working state.

### Stage A: base + context skeleton
1. Create `src/cleaning/base.py` with `Step` protocol/ABC, `StepReport`, `RunReport`, `Pipeline`.
2. Create `src/cleaning/context.py` with `CleaningContext` and sub-context skeletons (empty or minimal fields).
3. Create `CleaningContext.from_config()` — reads YAML from a config dir, loads crosswalks, validates, freezes.
4. Write tests for base and context: Pipeline composition, StepReport shape, CleaningContext construction, immutability enforcement, validation failure paths.
5. Do **not** yet write any step. Verify the fundament with a single trivial `NoOpStep` used in Pipeline tests.

### Stage B: first real step end-to-end
Choose the simplest CPS step: `AgeBandFilter`. Implement fully:
1. Step class with `required_columns`, `produced_columns`, `apply`.
2. Sub-context in `CleaningContext` if it needs one (age band bounds).
3. YAML config file.
4. Unit test with mock context.
5. Integration test in a minimal pipeline.

This validates the whole architecture on one concrete case. If anything feels awkward, fix here — before proliferating steps.

### Stage C: two more steps to reveal patterns
Add `SelfEmployedFilter` (uses `CLASSWLY`) and `EducationBridge` (uses per-year mapping). These exercise:
- Source-aware behavior (`CLASSWLY` semantics differ; document but likely no behavioral branching needed).
- Per-year configuration (education pre/post 1992).
- Crosswalk loading in context.

After Stage C, evaluate: does anything repeat between steps? If yes, and only if yes, extract to `expressions.py`.

### Stage D: methodologically critical steps
Now implement steps where switchable behavior matters most: `TopcodeAdjuster`, `AllocatedEarningsFilter`. These are the ones with per-year `branches_taken` populating StepReport meaningfully. Verify the branch-counting works end-to-end and that operator can inspect `RunReport.steps[i].branches_taken` to see which branches fired for which years.

### Stage E: derived/computed steps
Weeks-worked bridging (combines `WKSWORK1`/`WKSWORK2`; done - see §1 principle 4 exception, §4.1, §4.5 - as `bridge_weeks_pre_1976`, a `FunctionStep` in `custom_functions.py`, not a dedicated `Step` subclass), `FTFYNormalizer` (produces `_prov_ftfy_years`). These are Polars-heavy; they exercise whether the internal-atoms-as-Polars-expressions decision holds up. If a step becomes >200 lines, reconsider whether it's actually two steps - or, per the weeks-bridge case, whether it can be a plain `FunctionStep` instead of a class at all.

### Stage F: occupation mapping
`OccToQuadrantMapper`. This joins the crosswalk from context. Ensures the crosswalk-in-context pattern works cleanly. This is also the step that will be shared (with adaptation) for PSID.

### Later, when PSID enters
Add `SourceProfile.kind = "psid"`, PSID-specific steps under `steps/` (naming: `psid_*` if PSID-only, unprefixed if shared and source-aware). Do not preemptively refactor CPS steps for PSID until PSID actually requires it — apply the rule of two: generalize on the second real case, not the imagined one.

---

## 9. What NOT to build in v1

To keep v1 tractable, explicitly defer:

- ~~Step registry / factory-from-name (`build_step("topcode", ...)`) — defer until step count justifies it (15+).~~ **Superseded, see §2**: built at ~8 steps after an explicit decision to prioritize YAML-only reconfiguration over the original threshold. Still an explicit whitelist dict (`STEP_BUILDERS`), not `build_step(name, ...)`-style dynamic dispatch.
- Lambda-based `Step` (`Step.from_function(fn)`) — provokes anonymous steps without required_columns, produced_columns, or provenance. Every step is an explicit class. **`FunctionStep` (§4.1) is not a reopening of this** - it still requires `name`/`required_columns`/`produced_columns` explicitly, only `apply()`'s body is a plain function instead of a subclass method.
- Parallel step execution — Polars parallelizes internally. Step-level parallelism buys nothing.
- Custom atomic operations layer over Polars — use Polars expressions directly inside steps.
- DSL for step definitions in YAML — expressions in YAML are a well-known source of pain. **Still holds** - `type: FunctionStep`'s `function:` field (§6.4) is a dotted *import path* resolved by `importlib`, not an expression language; nothing beyond wiring (name, required/produced columns, a real Python callable) lives in YAML.
- Automatic before/after DataFrame diffing — expensive on large data; `n_in`/`n_out`/`dropped_reason_counts` covers the aggregate need.
- Full `fit`/`transform` split — steps are deterministic given context. Add `fit` only for the rare step that legitimately estimates parameters (e.g., a later `PSIDReweighter`).
- A generic multi-source cleaning framework — universality emerges from the second real source (PSID), not from speculative design.

---

## 10. Coordination with the rest of the project

- **Dagster assets** (existing under orchestration environment): the asset function is where `Settings` and `CleaningContext` meet. Asset reads paths from `Settings`, constructs `CleaningContext.from_config(...)`, calls `Pipeline.apply(df, context)`, returns the resulting DataFrame. Provenance columns propagate into the Parquet asset naturally. `src/pipelines/ipums_cleaning_example.py::run_cps_cleaning_example()` is the first real (non-test) caller of `CleaningContext`/`Pipeline`, following this same shape ahead of any Dagster asset existing yet - it's what proved the YAML-only edit loop (`config/cleaning/cps/pipeline.yaml`) actually works end-to-end against real bronze data.
- **CLI jobs** under `src/jobs/`: same pattern. Read settings, build context, run pipeline. Job is a thin click wrapper.
- **patito Silver schema**: extend to include `_prov_` columns. Consider a helper that generates the provenance-column portion of the schema from the Pipeline's step list (each step declares produced_columns and their types) — but this is an optimization; hand-writing them in v1 is fine.
- **MLflow / experiment tracking** (existing): log `context_hash` and key elements of `RunReport` (branches taken counts, drop reason counts) as run parameters/metrics. Code/config versioning (what `git_commit` would have covered, §4.3) is this integration's job, not `RunReport`'s.
- **`docs/methodology/`**: every non-trivial step and every notable per-year table gets a short markdown note here explaining the methodological rationale. Not code documentation — methodology documentation. Written at the time of decision, not retroactively.

---

## 11. Deliverables of the plan (what Claude Code should produce)

Given this document as input, produce:

1. A concrete file-level implementation plan for Stage A (base + context skeleton), listing:
   - Files to create with their intended contents (module-level summary, key classes/functions).
   - New tests to add.
   - Any existing files that need modification (imports, `__init__.py`, patito schemas) — but no new steps yet.
2. Explicit call-outs to the existing codebase where alignment is needed: naming conventions used in `src/`, existing pydantic-settings patterns, existing patito schema layout, existing Polars usage patterns, existing test structure.
3. A checklist for verifying each Stage-A deliverable before proceeding to Stage B.

Do not begin implementation until the plan for Stage A is reviewed. Do not go beyond Stage A in the initial plan — Stages B–F are placeholders for future planning cycles once Stage A is in place.

---

## 12. Reference summary of architectural decisions

For quick reference during implementation:

- Language of composition: **Polars expressions inside methodological steps**, not a custom atomic layer.
- Unit of methodology: **step class with name, required/produced columns, apply method**.
- Unit of provenance: **`_prov_`-prefixed columns in the DataFrame** (row-level) plus **StepReport** (aggregate).
- Unit of configuration: **CleaningContext**, separate from Settings, immutable, constructed from versioned YAML + crosswalks per run.
- Unit of reproducibility: **`context_hash`**, captured in `RunReport` (`git_commit` was cut, not deferred - see §4.3; `pipeline.yaml` isn't covered by any hash yet, a known gap).
- Complexity growth policy: **rule of two** — abstract only after the second real case demonstrates the pattern.
- Step assembly: **`Pipeline.from_config()` + `STEP_BUILDERS`** (§2, §6.4) — pulled forward from the original 15-step deferral by explicit decision; still an explicit whitelist, not dynamic dispatch.
- Custom/one-off step logic: **`FunctionStep` + `custom_functions.py`**, wired from YAML via a dotted `function:` import path (`importlib`), not a DSL (§4.1, §6.4, §9).
- A step's inputs don't have to come from a config table: **a step may compute its own fit at runtime from the DataFrame it's given** when the switch is a structural fact, not a tunable policy (§1 principle 4 exception, `bridge_weeks_pre_1976`).
