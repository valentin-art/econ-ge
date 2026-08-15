"""Step/Pipeline/report contracts for the CPS (and future PSID/other-source)
data-cleaning layer.

Classes:
    Pipeline:
        implements cleaning pipeline as a sequence of steps.
    Step:
        Individual steps (wrapper for a cleaning function).
    NoOpStep:
        Empty step example.
    StepReport:
        logs info about cleaning on an individual step.
    RunReport:
        Wraps StepReports over full cleaning pipeline.

"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import polars as pl
import structlog
import yaml

from src.cleaning.context import CleaningContext

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StepReport:
    """What one Step did to one DataFrame. Needs for debugging."""

    step_name: str
    n_in: int
    n_out: int
    dropped_reason_counts: dict[str, int] = field(default_factory=dict)
    branches_taken: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float | None = None


@dataclass(frozen=True)
class RunReport:
    """Contains a summary report from all Steps of a Pipeline."""

    pipeline_name: str
    steps: list[StepReport]
    context_hash: str
    started_at: datetime
    finished_at: datetime
    pipeline_hash: str | None = None


class Step(ABC):
    """Atomic cleaning step, e.g. an age-band restriction
    or a topcode adjustment.

    `required_columns` and `produced_columns` are used to validate
    step inputs and outputs. `is_idempotent` declares whether re-applying
    this step to its own output would change the result.

    Attributes:
        required_columns (frozenset[str]):
            Helps to validate step input.
        produced_columns (frozenset[str]):
            Helps to validate step output
        is_idempotent (bool) :
            Ensures the step applies once.

    Methods:
        apply(df, context):
            Use context (cleaning definition) to apply the cleaning step
            to the dataset.
        validate_context(context):
            Static check: does `context` actually have what this step will
            need at `apply()` time (e.g. a named sub-context entry)?
    """

    required_columns: frozenset[str] = frozenset()
    produced_columns: frozenset[str] = frozenset()
    is_idempotent: ClassVar[bool] = True

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        """Pure function of (df, context): no side effects, no state mutation."""
        ...

    def validate_context(self, context: CleaningContext) -> list[str]:
        """Returns issues if this step's dependencies are missing
        from `context`. Empty means nothing to report.
        """
        return []


class NoOpStep(Step):
    """Trivial pass-through step. Exists only for test purpose."""

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        return df, StepReport(step_name=self.name, n_in=len(df), n_out=len(df))


class Pipeline:
    """Linear composition of Steps.

    Attributes:
        steps (list[Step]) :
            A list of Steps (applied in the defined order)
        name (str):
            Name of the pipeline (e.g., may identify dataset/project)
        validate_between_steps (bool):
            A flag whether validate required/produced columns of each step,
            as well as steps consistency
        known_input_columns (frozenset[str]):
            Used to check whether sets have all columns required to complete
            the Pipeline.

    Methods:
        from_config(...):
            Opens the Pipeline (i.e., all Steps with definitions) from
            a config (yaml) file. Checks that all Steps are defined and can
            be initialized.
        validate_compatibility(...):
            Checks all required and produced columns are compatible on each
            step.
        validate_against_context(...):
            Checks each step's declared context dependencies (e.g. a
            named sub-context key) actually exist on a given context.
        apply(...):
            Applies steps one-by-one, returns cleaned dataset and cleaning
            report. Accepts a `pl.LazyFrame` (e.g. from `pl.scan_parquet`)
            as well as a `pl.DataFrame`, so a caller can compose lazy
            row-level operations (a `.filter()`, a `scan_parquet` glob over
            several files) before handing off - `apply()` collects once, in
            full, with no column projection: `known_input_columns` is a
            *minimum*-columns compatibility check (`validate_compatibility`),
            not the pipeline's promised output schema, so it must not be
            used to decide what to drop - the source's other columns
            (survey weights, record IDs, ...) are exactly what a caller
            typically still wants in the result. See the note on `apply()`
            for why a full Step-level LazyFrame protocol wouldn't buy more
            than accepting a LazyFrame at the boundary already does.
    """

    def __init__(
        self,
        steps: list[Step],
        name: str,
        validate_between_steps: bool = False,
        known_input_columns: frozenset[str] = frozenset(),
        fail_on_warning: bool = False,
        source_hash: str | None = None,
    ) -> None:
        self.steps = steps
        self.name = name
        self.validate_between_steps = validate_between_steps
        self.known_input_columns = known_input_columns
        self.fail_on_warning = fail_on_warning
        self.source_hash = source_hash

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        registry: Mapping[str, Callable[..., Step]],
        fail_on_warning: bool | None = None,
    ) -> Pipeline:
        """Build a Pipeline from a YAML file: `name`, `known_input_columns`,
        `validate_between_steps`, and a `steps` list of blocks, each
        `{type: <registry key>, name: <step name>, ...constructor kwargs}`.

        Args:
            config_path (Path):
                A path to yaml-file that contains full information about pipeline
                steps.
            registry (Mapping[...]):
                A mapping between Step types in config file and known steps that
                can be initialized.

        Returns:
            Pipeline

        Raise:
            ValueError if config file has unknown or invalid steps.
        """
        text = config_path.read_text()
        # Load config file
        raw = yaml.safe_load(text) or {}

        known_keys = {
            "name",
            "steps",
            "validate_between_steps",
            "known_input_columns",
            "fail_on_warning",
        }
        unknown = set(raw) - known_keys
        if unknown:
            raise ValueError(
                f"Pipeline.from_config: {config_path} has unknown top-level keys "
                f"{sorted(unknown)}; expected {sorted(known_keys)}"
            )

        steps: list[Step] = []
        seen_names: set[str] = set()
        for position, block in enumerate(raw.get("steps") or []):
            if not isinstance(block, Mapping):
                raise ValueError(  # noqa: TRY004
                    f"Pipeline.from_config: {config_path} step #{position} is "
                    f"{type(block).__name__}, expected a mapping of "
                    "{type: ..., name: ..., **kwargs}"
                )
            block = dict(block)
            type_name = block.pop("type", None)
            step_name = block.get("name", f"<unnamed #{position}>")
            if step_name in seen_names:
                raise ValueError(
                    f"Pipeline.from_config: {config_path} has duplicate step name "
                    f"{step_name!r}; names must be unique within a pipeline"
                )
            seen_names.add(step_name)

            builder = registry.get(type_name)
            if builder is None:
                raise ValueError(
                    f"Pipeline.from_config: {config_path} step {step_name!r} "
                    f"has unknown type {type_name!r}; available: "
                    f"{sorted(registry)}"
                )
            try:
                steps.append(builder(**block))
            except TypeError as exc:
                raise ValueError(
                    f"Pipeline.from_config: {config_path} step {step_name!r} "
                    f"(type={type_name!r}) failed to construct: {exc}"
                ) from exc
        return cls(
            steps=steps,
            name=raw.get("name", config_path.stem),
            validate_between_steps=raw.get("validate_between_steps", False),
            known_input_columns=frozenset(raw.get("known_input_columns", [])),
            # An explicit constructor argument wins; otherwise the YAML decides.
            fail_on_warning=(
                raw.get("fail_on_warning", False)
                if fail_on_warning is None
                else fail_on_warning
            ),
            source_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def validate_compatibility(self) -> list[str]:
        """Is each step's required_columns satisfied by `known_input_columns`
        plus the columns produced by every earlier step?

        Returns a list of human-readable issues; empty means no static
        incompatibility was found.
        """
        issues: list[str] = []
        available: set[str] = set(self.known_input_columns)

        # Missing are required minus available on each step
        for index, step in enumerate(self.steps):
            missing = step.required_columns - available
            if missing:
                issues.append(
                    f"step {index} ({step.name!r}) requires {sorted(missing)}, "
                    f"not in known_input_columns or produced by any earlier "
                    f"step in pipeline {self.name!r}"
                )
            available |= step.produced_columns
        return issues

    def validate_against_context(self, context: CleaningContext) -> list[str]:
        """Static check: does `context` have everything that every step
        declares it needs?

        Returns a list of issues. If any step has missing sub-context.
        """
        issues: list[str] = []
        for index, step in enumerate(self.steps):
            for issue in step.validate_context(context):
                issues.append(f"step {index} ({step.name!r}): {issue}")
        return issues

    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, RunReport]:
        """Runs `df` through every step in order.

        `df` may be a `pl.LazyFrame` (e.g. `pl.scan_parquet(path)` instead
        of `pl.read_parquet(path)`, or a lazily-filtered/-projected frame a
        caller built themselves) - `apply()` collects it once, in full,
        before the first step. Individual `Step`s still take and return
        `pl.DataFrame` rather than `pl.LazyFrame`: every step's
        `StepReport` needs concrete counts (rows dropped, branches taken),
        which forces a materialization at each step boundary regardless of
        the type used there, so threading `pl.LazyFrame` through the whole
        chain wouldn't fuse operations across steps - only defer the same
        unavoidable per-step collect to a slightly different line. Any
        column pushdown belongs in what the *caller* passes in (e.g.
        `pl.scan_parquet(path).select(wanted_columns)`), not here:
        `known_input_columns` is a minimum-columns check, not the
        pipeline's output schema, so `apply()` must not use it to decide
        what to drop.
        """
        context_issues = self.validate_against_context(context)
        if context_issues:
            raise ValueError(
                f"pipeline {self.name!r} is not compatible with the given "
                f"context: {context_issues}"
            )
        started_at = datetime.now(tz=UTC)
        reports: list[StepReport] = []
        current = df.collect() if isinstance(df, pl.LazyFrame) else df
        for step in self.steps:
            missing = step.required_columns - set(current.columns)
            if missing:
                raise ValueError(
                    f"step {step.name!r} in pipeline {self.name!r} requires columns "
                    f"{sorted(missing)}, not present in the input frame "
                    f"(columns: {sorted(current.columns)})"
                )
            step_started = datetime.now(tz=UTC)
            current, report = step.apply(current, context)
            if report.duration_seconds is None:
                report = replace(
                    report,
                    duration_seconds=(
                        datetime.now(tz=UTC) - step_started
                    ).total_seconds(),
                )
            if self.validate_between_steps:
                produced_missing = step.produced_columns - set(current.columns)
                if produced_missing:
                    raise ValueError(
                        f"step {step.name!r} declared produced_columns "
                        f"{sorted(step.produced_columns)} but {sorted(produced_missing)} "
                        "are missing from its output"
                    )
                if len(current) != report.n_out:
                    raise ValueError(
                        f"step {step.name!r} reported n_out={report.n_out} but "
                        f"returned a frame with {len(current)} rows"
                    )
            reports.append(report)

            # Log and raise warnings
            logger.info(
                "cleaning_step_applied",
                pipeline=self.name,
                step=step.name,
                n_in=report.n_in,
                n_out=report.n_out,
                dropped=report.dropped_reason_counts,
                branches=report.branches_taken,
            )
            for warning in report.warnings:
                logger.warning(
                    "cleaning_step_warning",
                    pipeline=self.name,
                    step=step.name,
                    warning=warning,
                )
                if self.fail_on_warning:  # new __init__ flag, default False
                    raise ValueError(
                        f"step {step.name!r} in pipeline {self.name!r} warned: {warning}"
                    )
        finished_at = datetime.now(tz=UTC)
        run_report = RunReport(
            pipeline_name=self.name,
            steps=reports,
            context_hash=context.compute_hash(),
            started_at=started_at,
            finished_at=finished_at,
            pipeline_hash=self.source_hash,
        )
        return current, run_report
