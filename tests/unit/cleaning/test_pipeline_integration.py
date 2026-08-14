from pathlib import Path

import polars as pl

from src.cleaning.base import Pipeline
from src.cleaning.context import CleaningContext
from src.cleaning.steps.registry import STEP_BUILDERS

FIXTURES = Path(__file__).parent / "fixtures" / "config"


def test_five_step_pipeline_runs_end_to_end_against_real_config() -> None:
    context = CleaningContext.from_config(
        config_dir=FIXTURES / "cps",
        source="ipums_cps_asec",
        crosswalks_dir=FIXTURES / "crosswalks",
    )
    pipeline = Pipeline.from_config(FIXTURES / "cps" / "pipeline.yaml", STEP_BUILDERS)
    # row0: AGE=15 dropped by BandFilter (below min_value)
    # row1: AGE=20, CLASSWLY=29 (unpaid family worker) dropped by wage_or_self_employed_filter
    # row2: AGE=95, CLASSWLY=10 (self-employed) survives both filters, AGELY derived
    #       from the pre-topcode AGE, then AGE itself topcoded to 90; YEAR=1995
    #       so weeks_worked_bridge just passes WKSWORK1=40 through directly
    # row3: AGE=40, CLASSWLY=99 (missing) dropped by wage_or_self_employed_filter
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

    assert pipeline.validate_compatibility() == []

    result, run_report = pipeline.apply(df, context)

    assert result["AGE"].to_list() == [90]
    assert result["CLASSWLY"].to_list() == [10]
    assert result["AGELY"].to_list() == [71]
    assert result["WEEKS_WORKED"].to_list() == [40.0]

    assert run_report.pipeline_name == "cps_universe_and_topcode"
    assert [step.step_name for step in run_report.steps] == [
        "age_band_filter",
        "wage_or_self_employed_filter",
        "age_last_year",
        "age_topcode",
        "weeks_worked_bridge",
    ]
    assert (run_report.steps[0].n_in, run_report.steps[0].n_out) == (4, 3)
    assert (run_report.steps[1].n_in, run_report.steps[1].n_out) == (3, 1)
    assert (run_report.steps[2].n_in, run_report.steps[2].n_out) == (1, 1)
    assert (run_report.steps[3].n_in, run_report.steps[3].n_out) == (1, 1)
    assert (run_report.steps[4].n_in, run_report.steps[4].n_out) == (1, 1)
    assert run_report.context_hash == context.compute_hash()
