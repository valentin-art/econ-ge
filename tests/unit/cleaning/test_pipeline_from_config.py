import functools
from pathlib import Path

import polars as pl
import pytest
import yaml

from src.cleaning.base import Pipeline
from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.band_filter import BandFilter
from src.cleaning.steps.function_step import FunctionStep, _resolve_function_step
from src.cleaning.steps.registry import STEP_BUILDERS

# The production STEP_BUILDERS["FunctionStep"] fixes its allowlist to
# src.cleaning.custom_functions. and cannot be widened from a YAML block
# (see MAJOR-4 in the review). To exercise `type: FunctionStep` resolution
# against the tests-only fixtures module, these tests use a registry that
# substitutes a wider allowlist bound at the Python level, not via config.
TEST_STEP_BUILDERS = {
    **STEP_BUILDERS,
    "FunctionStep": functools.partial(
        _resolve_function_step,
        allowed_prefixes=("tests.unit.cleaning.fixtures.",),
    ),
}


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


def test_builds_steps_in_order_with_declared_params(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "name": "my_pipeline",
            "known_input_columns": ["AGE"],
            "validate_between_steps": True,
            "steps": [
                {
                    "type": "BandFilter",
                    "name": "age_band",
                    "column": "AGE",
                    "min_value": 16,
                },
            ],
        },
    )

    pipeline = Pipeline.from_config(config_path, STEP_BUILDERS)

    assert pipeline.name == "my_pipeline"
    assert pipeline.known_input_columns == frozenset({"AGE"})
    assert pipeline.validate_between_steps is True
    assert len(pipeline.steps) == 1
    step = pipeline.steps[0]
    assert isinstance(step, BandFilter)
    assert step.name == "age_band"
    assert step.column == "AGE"
    assert step.min_value == 16


def test_defaults_name_to_file_stem_and_flags_to_false(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"steps": []})

    pipeline = Pipeline.from_config(config_path, STEP_BUILDERS)

    assert pipeline.name == "pipeline"
    assert pipeline.known_input_columns == frozenset()
    assert pipeline.validate_between_steps is False
    assert pipeline.steps == []


def test_runs_end_to_end_against_fixture_config() -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "config" / "cps" / "pipeline.yaml"
    )

    pipeline = Pipeline.from_config(fixture_path, STEP_BUILDERS)

    assert pipeline.validate_compatibility() == []
    df = pl.DataFrame(
        {
            "AGE": [15, 20, 95, 40],
            "CLASSWLY": [22, 29, 10, 99],
            "YEAR": [1970, 1970, 1995, 1970],
            "WKSWORK1": [None, None, 40, None],
            "WKSWORK2": [4, 4, 0, 4],
            "FEMALE": [0, 0, 0, 0],
            "RACE": [1, 1, 1, 1],
        }
    )
    result, run_report = pipeline.apply(df, _context())

    assert result["AGE"].to_list() == [90]
    assert result["CLASSWLY"].to_list() == [10]
    assert result["AGELY"].to_list() == [71]
    assert result["WEEKS_WORKED"].to_list() == [40.0]
    assert [step.step_name for step in run_report.steps] == [
        "age_band_filter",
        "wage_or_self_employed_filter",
        "age_last_year",
        "age_topcode",
        "weeks_worked_bridge",
    ]


def test_function_step_block_resolves_and_runs(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "steps": [
                {
                    "type": "FunctionStep",
                    "name": "fill_missing_weeks",
                    "function": "tests.unit.cleaning.fixtures.custom_functions.fill_missing_weeks_with_default",
                    "required_columns": ["WKSWORK1"],
                    "produced_columns": ["WKSWORK1"],
                },
            ],
        },
    )

    pipeline = Pipeline.from_config(config_path, TEST_STEP_BUILDERS)

    assert isinstance(pipeline.steps[0], FunctionStep)
    df = pl.DataFrame({"WKSWORK1": [10, None]})
    result, _ = pipeline.apply(df, _context())
    assert result["WKSWORK1"].to_list() == [10, 26.0]


def test_function_step_block_applies_params(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "steps": [
                {
                    "type": "FunctionStep",
                    "name": "fill_missing_weeks",
                    "function": "tests.unit.cleaning.fixtures.custom_functions.fill_missing_weeks_with_default",
                    "required_columns": ["WKSWORK1"],
                    "produced_columns": ["WKSWORK1"],
                    "params": {"default": 30.0},
                },
            ],
        },
    )

    pipeline = Pipeline.from_config(config_path, TEST_STEP_BUILDERS)

    df = pl.DataFrame({"WKSWORK1": [None]})
    result, _ = pipeline.apply(df, _context())
    assert result["WKSWORK1"].to_list() == [30.0]


def test_unknown_type_raises_with_available_types_listed(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"steps": [{"type": "NotARealStep", "name": "broken"}]},
    )

    with pytest.raises(ValueError, match="NotARealStep") as exc_info:
        Pipeline.from_config(config_path, STEP_BUILDERS)
    assert "BandFilter" in str(exc_info.value)


def test_bad_kwarg_raises_naming_file_step_and_type(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "steps": [
                {
                    "type": "BandFilter",
                    "name": "age_band",
                    "column": "AGE",
                    "not_a_real_kwarg": 1,
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="age_band") as exc_info:
        Pipeline.from_config(config_path, STEP_BUILDERS)
    assert "BandFilter" in str(exc_info.value)
    assert str(config_path) in str(exc_info.value)
