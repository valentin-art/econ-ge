"""A collection of custom functions for Functional steps."""

import polars as pl

from src.cleaning.context import CleaningContext


def age_last_year(
    df: pl.DataFrame, context: CleaningContext, max_topcoded_age: int = 71
) -> pl.DataFrame:
    """Derives AGELY (age as of last year) from AGE: `age - 1`, capped at
    `max_topcoded_age` for anyone already at or above the cap this year -
    aa_clean's `agely=age-1 if 17<=age<=71; agely=71 if age>=72`. Ports
    `cps_transformers.age_last_year(age_series, max_topcoded_age=71)`.
    """
    return df.with_columns(
        (pl.col("AGE") - 1).clip(upper_bound=max_topcoded_age).alias("AGELY")
    )


def bridge_weeks_pre_1976(
    df: pl.DataFrame, context: CleaningContext
) -> tuple[pl.DataFrame, list[str]]:
    """Derives WEEKS_WORKED: WKSWORK1 where present; for YEAR<1976 (where
    WKSWORK1 is always null and only the bracketed WKSWORK2 exists), the
    mean WKSWORK1 within the record's own FEMALE/RACE/WKSWORK2 cell, fit
    from this same DataFrame's own 1976-78 rows - ports aa_clean's
    clean7678km.do:132-140 (fit) + clean6275km.do:142-151 (apply).

    FEMALE/RACE are assumed already produced upstream; this function only
    reads them, it never derives sex/race groups itself.

    Requires the input DataFrame to actually contain 1976-78 rows to fit
    from - called on a purely pre-1976 slice (e.g. a single year's bronze
    file on its own), every bridged value comes out null, since the fit
    population is empty; that case is reported back as a warning rather
    than raised, since processing a single year's bronze file on its own
    is a legitimate, if incomplete, way to call this. For YEAR>=1976 this
    always just returns WKSWORK1, independent of the fit.
    """

    TEMP = "_bracket_mean"
    if TEMP in df.columns:
        raise ValueError(
            f"bridge_weeks_pre_1976: input already has reserved column {TEMP!r}"
        )
    group_means = (
        df.filter(pl.col("YEAR").is_between(1976, 1978))
        .group_by(["FEMALE", "RACE", "WKSWORK2"])
        .agg(pl.col("WKSWORK1").mean().cast(pl.Float64).alias(TEMP))
    )

    n_pre_1976 = df.filter(pl.col("YEAR") < 1976).height
    warnings: list[str] = []
    if n_pre_1976 and group_means.is_empty():
        warnings.append(
            "bridge_weeks_pre_1976: no 1976-1978 rows in the input frame to fit "
            f"bracket means from; all {n_pre_1976} pre-1976 row(s) get a null "
            "WEEKS_WORKED"
        )

    result = (
        df.join(
            group_means,
            on=["FEMALE", "RACE", "WKSWORK2"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.when(pl.col("YEAR") < 1976)
            .then(pl.coalesce([pl.col("WKSWORK1").cast(pl.Float64), pl.col(TEMP)]))
            .otherwise(pl.col("WKSWORK1").cast(pl.Float64))
            .alias("WEEKS_WORKED")
        )
        .drop(TEMP)
    )
    return result, warnings
