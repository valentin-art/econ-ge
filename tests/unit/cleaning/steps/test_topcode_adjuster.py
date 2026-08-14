import polars as pl
import pytest

from src.cleaning.context import (
    CleaningContext,
    SourceProfile,
    TopcodeConfig,
    YearBandThreshold,
)
from src.cleaning.steps.topcode_adjuster import TopcodeAdjuster


def _context() -> CleaningContext:
    return CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        topcode={
            "wage": TopcodeConfig(
                multiplier=1.5,
                uncovered_years="skip",
                thresholds=[
                    YearBandThreshold(
                        start_year=1968,
                        end_year=1981,
                        threshold=50000,
                        match_mode="exact",
                    ),
                    YearBandThreshold(
                        start_year=1996,
                        end_year=1996,
                        threshold=150000,
                        match_mode="gte",
                    ),
                ],
            )
        },
    )


def test_applies_multiplier_across_pre_and_post_1988_eras() -> None:
    df = pl.DataFrame(
        {
            "YEAR": [1970, 1970, 1996, 1996, 1996, 1960],
            "INCWAGE": [50000.0, 40000.0, 150000.0, 200000.0, 100000.0, 99999.0],
        }
    )

    result, report = TopcodeAdjuster("topcode_adjuster").apply(df, _context())

    assert result["INCWAGE"].to_list() == [
        75000.0,
        40000.0,
        225000.0,
        300000.0,
        100000.0,
        99999.0,
    ]
    assert report.n_in == 6
    assert report.n_out == 6
    # row0 (1970, 50000): in band1 (exact, threshold 50000) -> hit
    # row1 (1970, 40000): in band1 -> not hit (not == 50000)
    # row2 (1996, 150000): in band2 (gte, threshold 150000) -> hit
    # row3 (1996, 200000): in band2 -> hit
    # row4 (1996, 100000): in band2 -> not hit (not >= 150000)
    # row5 (1960, 99999): no band covers 1960 -> no_threshold_for_year
    assert report.branches_taken == {
        "exact_match": 1,
        "gte_match": 2,
        "in_band_not_hit": 2,
        "null_income": 0,
        "no_threshold_for_year": 1,
    }


def test_pre_1988_uses_exact_equality_not_ge() -> None:
    # aa_clean's pre-1988 style is `== threshold`, not `>= threshold` -
    # a value strictly above the threshold should NOT be multiplied.
    df = pl.DataFrame({"YEAR": [1970], "INCWAGE": [60000.0]})

    result, _ = TopcodeAdjuster("topcode_adjuster").apply(df, _context())

    assert result["INCWAGE"].to_list() == [60000.0]


def test_1988_plus_uses_ge_not_exact_equality() -> None:
    df = pl.DataFrame({"YEAR": [1996], "INCWAGE": [999999.0]})

    result, _ = TopcodeAdjuster("topcode_adjuster").apply(df, _context())

    assert result["INCWAGE"].to_list() == [1499998.5]


def test_only_incwage_and_year_columns_remain_besides_originals() -> None:
    df = pl.DataFrame({"YEAR": [1970], "INCWAGE": [50000.0], "OTHER": [1]})

    result, _ = TopcodeAdjuster("topcode_adjuster").apply(df, _context())

    assert set(result.columns) == {"YEAR", "INCWAGE", "OTHER"}


def test_column_defaults_to_incwage_but_is_reusable_for_another_income_column() -> None:
    step = TopcodeAdjuster("topcode_adjuster")
    assert step.required_columns == frozenset({"INCWAGE", "YEAR"})
    assert step.produced_columns == frozenset({"INCWAGE"})

    df = pl.DataFrame({"YEAR": [1970], "INCBUS": [50000.0]})
    other_step = TopcodeAdjuster("business_income_topcode", column="INCBUS")

    result, _ = other_step.apply(df, _context())

    assert other_step.required_columns == frozenset({"INCBUS", "YEAR"})
    assert result["INCBUS"].to_list() == [75000.0]


def test_topcode_key_selects_among_multiple_named_instances() -> None:
    context = CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        topcode={
            "wage": TopcodeConfig(
                multiplier=1.5,
                uncovered_years="skip",
                thresholds=[
                    YearBandThreshold(
                        start_year=2000,
                        end_year=2000,
                        threshold=1.0,
                        match_mode="gte",
                    )
                ],
            ),
            "income": TopcodeConfig(
                multiplier=2.0,
                uncovered_years="skip",
                thresholds=[
                    YearBandThreshold(
                        start_year=1996,
                        end_year=1996,
                        threshold=100000,
                        match_mode="gte",
                    )
                ],
            ),
        },
    )
    df = pl.DataFrame({"YEAR": [1996], "INCTOT": [200000.0]})
    step = TopcodeAdjuster("income_topcode", column="INCTOT", topcode_key="income")

    result, _ = step.apply(df, context)

    assert result["INCTOT"].to_list() == [400000.0]


def test_missing_topcode_key_raises_clear_error() -> None:
    step = TopcodeAdjuster("topcode_adjuster", topcode_key="income")
    df = pl.DataFrame({"YEAR": [1996], "INCWAGE": [200000.0]})

    with pytest.raises(ValueError, match="income.*available.*wage"):
        step.apply(df, _context())


def test_validate_context_passes_when_topcode_key_present() -> None:
    step = TopcodeAdjuster("topcode_adjuster", topcode_key="wage")

    assert step.validate_context(_context()) == []


def test_validate_context_flags_missing_topcode_key_before_apply() -> None:
    step = TopcodeAdjuster("topcode_adjuster", topcode_key="income")

    issues = step.validate_context(_context())

    assert len(issues) == 1
    assert "income" in issues[0]
    assert "wage" in issues[0]


def test_null_income_in_a_covered_band_is_reported_separately_from_a_miss() -> None:
    # A null income value in an otherwise-covered band should not be
    # indistinguishable from "in band, below threshold" - see null_income.
    df = pl.DataFrame({"YEAR": [1970], "INCWAGE": [None]})

    result, report = TopcodeAdjuster("topcode_adjuster").apply(df, _context())

    assert result["INCWAGE"].to_list() == [None]
    assert report.branches_taken["null_income"] == 1
    assert report.branches_taken["in_band_not_hit"] == 0


def test_raises_when_input_already_has_a_reserved_scratch_column() -> None:
    df = pl.DataFrame({"YEAR": [1970], "INCWAGE": [50000.0], "_topcode_hit": [True]})

    with pytest.raises(ValueError, match="_topcode_hit"):
        TopcodeAdjuster("topcode_adjuster").apply(df, _context())


def test_integer_column_is_promoted_to_float_without_a_warning() -> None:
    # The promotion is unconditional and documented, so it is a property of
    # the step, not a data condition. Warning about it would fire on every
    # single run and make Pipeline(fail_on_warning=True) permanently unusable.
    df = pl.DataFrame({"YEAR": [1970], "INCWAGE": [50000]})

    result, report = TopcodeAdjuster("topcode_adjuster").apply(df, _context())

    assert result.schema["INCWAGE"] == pl.Float64
    assert report.warnings == []


def test_output_dtype_does_not_depend_on_whether_a_row_was_topcoded() -> None:
    hit = pl.DataFrame({"YEAR": [1970], "INCWAGE": [50000]})
    miss = pl.DataFrame({"YEAR": [1970], "INCWAGE": [10]})
    step = TopcodeAdjuster("topcode_adjuster")

    hit_result, _ = step.apply(hit, _context())
    miss_result, _ = step.apply(miss, _context())

    assert hit_result.schema["INCWAGE"] == miss_result.schema["INCWAGE"] == pl.Float64


def test_uncovered_year_warns_when_configured_to_skip() -> None:
    # Under `skip` an uncovered year must still leave a trace in the report -
    # a silent pass-through is exactly the failure mode the policy exists to
    # avoid. DeflatorMergeStep already behaves this way.
    df = pl.DataFrame({"YEAR": [1970, 2015], "INCWAGE": [50000.0, 200000.0]})

    _, report = TopcodeAdjuster("topcode_adjuster").apply(df, _context())

    assert len(report.warnings) == 1
    assert "2015" in report.warnings[0]
    assert report.branches_taken["no_threshold_for_year"] == 1


def test_null_year_is_reported_separately_from_an_uncovered_year() -> None:
    # Regression: a null YEAR matches no band, so it lands among the uncovered
    # rows - and `sorted()` over a year list containing None raised TypeError
    # instead of the intended, actionable ValueError.
    df = pl.DataFrame(
        {"YEAR": [1970, None, 2015], "INCWAGE": [50000.0, 60000.0, 200000.0]}
    )

    _, report = TopcodeAdjuster("topcode_adjuster").apply(df, _context())

    assert len(report.warnings) == 2
    assert any("null YEAR" in warning for warning in report.warnings)
    assert any("2015" in warning for warning in report.warnings)
    assert report.branches_taken["no_threshold_for_year"] == 2


def test_null_year_raises_a_value_error_not_a_type_error_under_error_policy() -> None:
    context = CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        topcode={
            "wage": TopcodeConfig(
                multiplier=1.5,
                uncovered_years="error",
                thresholds=[
                    YearBandThreshold(
                        start_year=1996,
                        end_year=1996,
                        threshold=150000,
                        match_mode="gte",
                    )
                ],
            )
        },
    )
    df = pl.DataFrame({"YEAR": [None, 2015], "INCWAGE": [200000.0, 200000.0]})

    with pytest.raises(ValueError, match=r"no threshold band.*\[2015\]"):
        TopcodeAdjuster("topcode_adjuster").apply(df, context)


def test_uncovered_year_raises_when_configured_to_error() -> None:
    context = CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        topcode={
            "wage": TopcodeConfig(
                multiplier=1.5,
                uncovered_years="error",
                thresholds=[
                    YearBandThreshold(
                        start_year=1996,
                        end_year=1996,
                        threshold=150000,
                        match_mode="gte",
                    )
                ],
            )
        },
    )
    df = pl.DataFrame({"YEAR": [2015], "INCWAGE": [200000.0]})

    with pytest.raises(ValueError, match=r"no threshold band.*\[2015\]"):
        TopcodeAdjuster("topcode_adjuster").apply(df, context)
