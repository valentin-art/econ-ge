# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A macroeconomic general-equilibrium model of occupational choice under technological
progress, calibrated from microdata (CPS ASEC, IPUMS, US Census) and macro series (BEA,
CPI). Data flows one direction: external -> bronze (parsed, schema-validated) -> silver
(cleaned/harmonized) -> Postgres -> dbt marts -> model input, with model outputs stored
back for analysis. `src/` subpackages map onto those stages.

## Setup

**IMPORTANT: nothing runs without decrypting secrets first.** `.env` is committed empty
(0 bytes), so `load_dotenv()` in `src/config/settings.py` supplies nothing — every
credential comes from `.envrc`, which runs:

```bash
export SOPS_AGE_KEY_FILE="$HOME/.config/sops/age/keys.txt"
eval "$(sops -d --output-type dotenv secrets.enc.env | sed 's/^/export /')"
```

So you need `direnv`, `sops`, and an age private key at that path whose public key
matches `.sops.yaml`. `secrets.enc.env` (tracked, encrypted) holds `POSTGRES_*`,
`BEA_API_KEY`, `IPUMS_API_KEY`. Skip this and the failures are indirect, not obvious:
`docker compose up -d` interpolates empty `${POSTGRES_USER}`/`${POSTGRES_PASSWORD}`,
dbt fails on `env_var(...)`, and the BEA/IPUMS extractors get empty-string API keys
because `Settings` defaults them to `""` rather than raising.

The `.envrc` files (root, `dbt/`, `orchestration/` — the latter two use `source_up`)
are all gitignored, so this is the only durable record of the setup.

## Environments

Three independent Python environments — not interchangeable:

| | Python | Purpose |
|---|---|---|
| root `.venv` | 3.14 | all `src/` code and tests; package name is `econ-model`, import root is `src` |
| `orchestration/` | 3.14 | Dagster; imports root code via an editable path dep on `..` |
| `dbt/` | 3.10 | dbt only — pinned back because dbt-core's mashumaro pin won't import on 3.14 |

Root `pyproject.toml` says `requires-python = ">=3.12"` but the venv is actually 3.14;
don't assume 3.12 semantics. Run root commands with `uv run ...` from the repo root;
run the other two from their own directories via their `.venv/bin/`.

## Commands

```bash
# Tests (root env)
uv run pytest                                   # full suite
uv run pytest tests/unit/<module> -q            # one module
uv run pytest path/to/test.py::test_name        # single test

# Lint / format / type-check — via pre-commit, which supplies its own isolated
# ruff/mypy/sqlfluff. `uv run ruff` and `uv run mypy` FAIL: the `dev` extra that
# declares them is not installed in .venv (only pytest and pre-commit are).
uv run pre-commit run --all-files
uv run pre-commit run ruff --all-files      # a single hook
uv sync --extra dev                          # only if you want ruff/mypy on PATH directly

# dbt / Dagster (their own venvs)
cd dbt && .venv/bin/dbt run
cd orchestration && .venv/bin/dagster dev

# Infra — postgres only, despite what README.md claims
docker compose up -d
```

pytest, ruff, and mypy all run on **pure defaults** — there is no `[tool.pytest.ini_options]`,
`[tool.ruff]`, or `[tool.mypy]` anywhere in the repo, and no `pytest.ini`/`ruff.toml`/
`mypy.ini`. `.pre-commit-config.yaml` is the only place scope is narrowed (mypy to
`^src/`, sqlfluff to `^dbt/models/`) — so it is effectively the lint configuration, and
pre-commit is the only way those tools run at all. No pytest markers are registered or
used, so `-m <marker>` selects nothing.

## Configuration — keep these separate

- **`Settings`** (`src/config/settings.py`, pydantic-settings, env-driven): infrastructure
  — paths, API keys, DB connection. Changes freely, never affects scientific results.
- **`CleaningContext`** (`src/cleaning/context.py`, frozen once built): methodology —
  per-year tables, crosswalks, source profile, from versioned YAML under
  `config/cleaning/<source>/`. Every change is a methodological decision.
- **`src/config/sources.py`**: plain constants (table names, sample window, reference
  year) — fixed source parameters, neither env-driven nor per-run methodology.

`Settings` and `CleaningContext` must never import or reference each other. They meet
only in a `pipelines/`/`jobs/` caller or a Dagster asset, which reads paths from
`Settings` and passes them explicitly into `CleaningContext.from_config(...)`. Keep
`cleaning/` and `harmonization/` free of paths, Postgres, Dagster, and `Settings`
entirely, so the same code runs unchanged in notebooks, tests, CLI jobs, and assets.

## Conventions

- **Polars for new cleaning-layer code; the existing BEA/parser/schema code is pandas**
  and stays that way. Do not rewrite working pandas to Polars as a drive-by — it's the
  larger half of the codebase.
- `structlog` for logging (`structlog.get_logger(__name__)`), not stdlib `logging`
  (which appears only in `src/utils/logging.py`, the configuration entry point).
- Resolve all filesystem paths via `Settings.paths`, never hardcoded. `data/` is
  gitignored at every depth — never commit anything under it, and note that a doc placed
  under a nested `data/` directory would be silently ignored too.
- `src/input_output/` is the IO boundary (csv/parquet/Postgres); `src/jobs/` are thin
  `click` CLI wrappers (`uv run python -m src.jobs.<name>`) over `src/pipelines/`.

## Naming — deliberately inconsistent, do not "fix"

The installed distribution is **`econ-model`** (`pyproject.toml`), the import root is
**`src`**, the directory is **`econ-ge`**, and the dbt project *and* profile are both
**`econ_ge`**. These are load-bearing: `dbt_project.yml`'s `profile:` key must keep
matching `profiles.yml`, so renaming toward consistency breaks dbt silently. Use
`econ-model` for any `uv`/packaging operation.

## Further reading

- `.claude/review_code.prompt.md` + `.claude/agents/code-reviewer.md` — the on-demand
  review rubric; reviews are saved to `reviews/`. The `claude-pr-review` job in
  `.github/workflows/claude.yml` runs a *separate* automated review with its own inline
  prompt — check that file directly rather than assuming the two match.
