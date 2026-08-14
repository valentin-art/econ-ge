import polars as pl
import pytest

from src.cleaning.base import NoOpStep, Pipeline, Step, StepReport
from src.cleaning.context import (
    CleaningContext,
    SourceProfile,
    TopcodeConfig,
    YearBandThreshold,
)


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


class _NeedsFoo(Step):
    required_columns = frozenset({"FOO"})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        return df, StepReport(step_name=self.name, n_in=len(df), n_out=len(df))


class _ProducesFoo(Step):
    produced_columns = frozenset({"FOO"})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        return df, StepReport(step_name=self.name, n_in=len(df), n_out=len(df))


class _LiesAboutOutput(Step):
    produced_columns = frozenset({"NEVER_ACTUALLY_ADDED"})

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        return df, StepReport(step_name=self.name, n_in=len(df), n_out=len(df))


class _NeedsTopcodeKey(Step):
    def __init__(self, name: str, topcode_key: str) -> None:
        super().__init__(name)
        self.topcode_key = topcode_key

    def validate_context(self, context: CleaningContext) -> list[str]:
        if self.topcode_key in context.topcode:
            return []
        return [f"no topcode config named {self.topcode_key!r}"]

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        return df, StepReport(step_name=self.name, n_in=len(df), n_out=len(df))


def test_pipeline_apply_runs_steps_in_order_and_builds_run_report() -> None:
    df = pl.DataFrame({"AGE": [16, 40, 90]})
    pipeline = Pipeline(
        steps=[NoOpStep("first"), NoOpStep("second")], name="test_pipeline"
    )

    result_df, run_report = pipeline.apply(df, _context())

    assert result_df.equals(df)
    assert run_report.pipeline_name == "test_pipeline"
    assert [r.step_name for r in run_report.steps] == ["first", "second"]
    assert all(r.n_in == 3 and r.n_out == 3 for r in run_report.steps)
    assert all(r.duration_seconds is not None for r in run_report.steps)


def test_pipeline_apply_raises_when_required_column_missing() -> None:
    pipeline = Pipeline(steps=[_NeedsFoo("needs_foo")], name="test_pipeline")
    df = pl.DataFrame({"BAR": [1, 2]})

    with pytest.raises(ValueError, match="FOO"):
        pipeline.apply(df, _context())


def test_validate_compatibility_passes_for_compatible_chain() -> None:
    pipeline = Pipeline(
        steps=[_ProducesFoo("produces_foo"), _NeedsFoo("needs_foo")],
        name="test_pipeline",
    )

    assert pipeline.validate_compatibility() == []


def test_validate_compatibility_flags_a_genuine_mismatch() -> None:
    pipeline = Pipeline(
        steps=[NoOpStep("noop"), _NeedsFoo("needs_foo")], name="test_pipeline"
    )

    issues = pipeline.validate_compatibility()

    assert len(issues) == 1
    assert "FOO" in issues[0]


def test_validate_between_steps_catches_a_step_lying_about_produced_columns() -> None:
    pipeline = Pipeline(
        steps=[_LiesAboutOutput("liar")],
        name="test_pipeline",
        validate_between_steps=True,
    )
    df = pl.DataFrame({"AGE": [16]})

    with pytest.raises(ValueError, match="NEVER_ACTUALLY_ADDED"):
        pipeline.apply(df, _context())


def test_validate_context_defaults_to_no_issues() -> None:
    assert NoOpStep("noop").validate_context(_context()) == []


def test_validate_against_context_passes_when_step_dependency_exists() -> None:
    pipeline = Pipeline(
        steps=[_NeedsTopcodeKey("wage_topcode", topcode_key="wage")],
        name="test_pipeline",
    )
    context = CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        topcode={
            "wage": TopcodeConfig(
                uncovered_years="skip",
                thresholds=[
                    YearBandThreshold(
                        start_year=2000,
                        end_year=2000,
                        threshold=1.0,
                        match_mode="gte",
                    )
                ],
            )
        },
    )

    assert pipeline.validate_against_context(context) == []


def test_validate_against_context_flags_missing_topcode_key() -> None:
    pipeline = Pipeline(
        steps=[_NeedsTopcodeKey("income_topcode", topcode_key="income")],
        name="test_pipeline",
    )

    issues = pipeline.validate_against_context(_context())

    assert len(issues) == 1
    assert "step 0 ('income_topcode')" in issues[0]
    assert "income" in issues[0]
