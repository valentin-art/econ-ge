import polars as pl
import pytest

from src.cleaning.context import CleaningContext, SourceProfile
from src.cleaning.steps.function_step import FunctionStep


def _context() -> CleaningContext:
    return CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))


# Toy example only - not real aa_clean methodology. Shows the shape a
# contributor's own function would take: (df, context) -> df, no ABC
# ceremony required. A real "backward" imputation (e.g. pre-1976
# weeks-worked from 1976-78 group means) would look like this in outline,
# but with real regression/group-mean logic instead of a flat fill.
def _fill_missing_weeks_with_flat_default(
    df: pl.DataFrame, context: CleaningContext
) -> pl.DataFrame:
    return df.with_columns(pl.col("WKSWORK1").fill_null(26).alias("WKSWORK1"))


def test_wraps_a_plain_function_and_computes_n_in_n_out() -> None:
    df = pl.DataFrame({"WKSWORK1": [10, None, 40]})

    step = FunctionStep(
        "fill_missing_weeks",
        fn=_fill_missing_weeks_with_flat_default,
        required_columns=frozenset({"WKSWORK1"}),
        produced_columns=frozenset({"WKSWORK1"}),
    )
    result, report = step.apply(df, _context())

    assert result["WKSWORK1"].to_list() == [10, 26, 40]
    assert report.step_name == "fill_missing_weeks"
    assert report.n_in == 3
    assert report.n_out == 3
    assert report.dropped_reason_counts == {}


def test_reports_generic_row_count_delta_when_function_drops_rows() -> None:
    def _drop_negative(df: pl.DataFrame, context: CleaningContext) -> pl.DataFrame:
        return df.filter(pl.col("X") >= 0)

    df = pl.DataFrame({"X": [-1, 0, 1]})
    step = FunctionStep(
        "drop_negative",
        fn=_drop_negative,
        required_columns=frozenset({"X"}),
        produced_columns=frozenset(),
    )

    _, report = step.apply(df, _context())

    assert report.n_in == 3
    assert report.n_out == 2
    assert report.dropped_reason_counts == {"function_step": 1}


def test_wrapped_function_can_return_warnings_alongside_the_dataframe() -> None:
    def _fn(
        df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, list[str]]:
        return df, ["fit population was empty"]

    step = FunctionStep(
        "warns",
        fn=_fn,
        required_columns=frozenset(),
        produced_columns=frozenset(),
    )
    _, report = step.apply(pl.DataFrame({"X": [1]}), _context())

    assert report.warnings == ["fit population was empty"]


def test_warns_when_wrapped_function_adds_rows() -> None:
    def _fanout(df: pl.DataFrame, context: CleaningContext) -> pl.DataFrame:
        return pl.concat([df, df])

    step = FunctionStep(
        "fanout",
        fn=_fanout,
        required_columns=frozenset(),
        produced_columns=frozenset(),
    )
    _, report = step.apply(pl.DataFrame({"X": [1, 2]}), _context())

    assert report.n_in == 2
    assert report.n_out == 4
    assert len(report.warnings) == 1
    assert "added 2 rows" in report.warnings[0]


def test_raises_when_wrapped_function_returns_a_non_dataframe() -> None:
    def _bad(df: pl.DataFrame, context: CleaningContext) -> None:
        return None

    step = FunctionStep(
        "bad",
        fn=_bad,
        required_columns=frozenset(),
        produced_columns=frozenset(),
    )

    with pytest.raises(TypeError, match="NoneType"):
        step.apply(pl.DataFrame({"X": [1]}), _context())


def test_required_and_produced_columns_are_explicit_constructor_args() -> None:
    step = FunctionStep(
        "noop",
        fn=lambda df, context: df,
        required_columns=frozenset({"A", "B"}),
        produced_columns=frozenset({"C"}),
    )

    assert step.required_columns == frozenset({"A", "B"})
    assert step.produced_columns == frozenset({"C"})
