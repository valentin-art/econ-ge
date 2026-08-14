import polars as pl
import pytest

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.band_filter import BandFilter
from src.cleaning.steps.function_step import FunctionStep, _resolve_function_step
from src.cleaning.steps.registry import STEP_BUILDERS, _build_function_step
from src.cleaning.steps.topcode_adjuster import TopcodeAdjuster

CUSTOM_FN_PATH = (
    "tests.unit.cleaning.fixtures.custom_functions.fill_missing_weeks_with_default"
)
FIXTURES_PREFIX = "tests.unit.cleaning.fixtures."


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


def test_registry_contains_expected_standard_types() -> None:
    assert STEP_BUILDERS["BandFilter"] is BandFilter
    assert STEP_BUILDERS["TopcodeAdjuster"] is TopcodeAdjuster
    assert STEP_BUILDERS["FunctionStep"] is _build_function_step


def test_build_function_step_resolves_dotted_path_and_runs() -> None:
    # _build_function_step (the STEP_BUILDERS-registered, YAML-facing
    # builder) is fixed to src.cleaning.custom_functions.; a fixtures-only
    # dotted path needs the shared resolver with an explicit, code-level
    # (not YAML-level) allowlist widening.
    step = _resolve_function_step(
        name="fill_missing_weeks",
        function=CUSTOM_FN_PATH,
        required_columns=["WKSWORK1"],
        produced_columns=["WKSWORK1"],
        allowed_prefixes=(FIXTURES_PREFIX,),
    )

    assert isinstance(step, FunctionStep)
    df = pl.DataFrame({"WKSWORK1": [10, None, 40]})
    result, report = step.apply(df, _context())

    assert result["WKSWORK1"].to_list() == [10, 26.0, 40]
    assert report.step_name == "fill_missing_weeks"


def test_build_function_step_applies_params_via_partial() -> None:
    step = _resolve_function_step(
        name="fill_missing_weeks",
        function=CUSTOM_FN_PATH,
        required_columns=["WKSWORK1"],
        produced_columns=["WKSWORK1"],
        params={"default": 30.0},
        allowed_prefixes=(FIXTURES_PREFIX,),
    )

    df = pl.DataFrame({"WKSWORK1": [None]})
    result, _ = step.apply(df, _context())

    assert result["WKSWORK1"].to_list() == [30.0]


def test_build_function_step_raises_on_unresolvable_path() -> None:
    with pytest.raises(ValueError, match="not_a_real_module"):
        _build_function_step(
            name="broken",
            function="not_a_real_module.not_a_real_function",
            required_columns=[],
            produced_columns=[],
        )


def test_build_function_step_raises_on_non_dotted_path() -> None:
    with pytest.raises(ValueError, match="not a valid dotted path"):
        _build_function_step(
            name="broken",
            function="not_dotted",
            required_columns=[],
            produced_columns=[],
        )
