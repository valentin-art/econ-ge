"""CLI entry point: transform BEA bronze parquet into the silver panel.

Pure bronze -> silver transform (no BEA API call, no network) — reads the
FixedAssets/NIPA bronze parquet already produced by jobs/extract_bea.py +
jobs/parse_bea_bronze.py (or the fused jobs/build_ces_capital_inputs.py) and
writes:
    data/silver/bea/bea_silver.parquet      -- one row per year, short column names
    data/silver/bea/bea_silver_columns.json -- {short_name: one-line description}

Standalone, no orchestrator dependency:
    uv run python -m src.jobs.transform_bea_silver
"""

import json

import click

from src.config.settings import settings
from src.input_output.parquet import write_parquet
from src.pipelines.bea_silver_pipeline import run_bea_silver_pipeline
from src.schemas.silver.bea_silver import column_descriptions
from src.utils.logging import configure_logging


@click.command()
def main() -> None:
    """Run the bronze->silver BEA transform and write parquet + column-description outputs."""
    configure_logging()

    bronze_dir = settings.paths.bronze / "bea"
    silver = run_bea_silver_pipeline(bronze_dir)

    out_dir = settings.paths.silver / "bea"
    write_parquet(silver.reset_index(), out_dir / "bea_silver.parquet")

    columns_path = out_dir / "bea_silver_columns.json"
    columns_path.parent.mkdir(parents=True, exist_ok=True)
    columns_path.write_text(json.dumps(column_descriptions(), indent=2) + "\n")

    click.echo(f"Wrote silver outputs to {out_dir}")


if __name__ == "__main__":
    main()
