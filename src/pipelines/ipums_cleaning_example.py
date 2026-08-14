"""Example: end-to-end example that constructs a pipeline and performs cleaning with
basic cleaners described in pipeline.yaml.
"""

import polars as pl
import structlog

from src.cleaning.base import Pipeline
from src.cleaning.context import CleaningContext
from src.cleaning.steps.registry import STEP_BUILDERS
from src.config.settings import settings

log = structlog.get_logger(__name__)


def run_cps_cleaning_example(year: int = 2019) -> pl.DataFrame:
    """Load bronze IPUMS CPS ASEC data for `year` and run it through the
    pipeline defined in config/cleaning/cps/pipeline.yaml.
    """
    config_dir = settings.cleaning_config_root / "cps"

    context = CleaningContext.from_config(
        config_dir=config_dir,
        source="ipums_cps_asec",
    )
    pipeline = Pipeline.from_config(config_dir / "pipeline.yaml", STEP_BUILDERS)

    issues = pipeline.validate_compatibility() + pipeline.validate_against_context(
        context
    )
    if issues:
        raise ValueError(f"pipeline.yaml is not internally consistent: {issues}")

    bronze_file = settings.paths.ipums_bronze_dir("cps") / f"{year}.parquet"
    # Lazy scan: Pipeline.apply() accepts a LazyFrame and collects it once,
    # in full (no column pushdown - the result is meant to carry every
    # bronze column through, not just what the cleaning steps touch).
    lf = pl.scan_parquet(bronze_file)

    result, run_report = pipeline.apply(lf, context)
    log.info(
        "cps_cleaning_example_complete",
        year=year,
        n_in=run_report.steps[0].n_in if run_report.steps else result.height,
        n_out=result.height,
        steps=[step.step_name for step in run_report.steps],
    )
    return result


if __name__ == "__main__":
    run_cps_cleaning_example()
