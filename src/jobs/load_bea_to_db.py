"""CLI entry point: load the BEA silver panel into Postgres.

Reads data/silver/bea/bea_silver.parquet and loads it into `silver.bea_nipa`
via a full TRUNCATE + COPY refresh.

Standalone, no orchestrator dependency:
    uv run python -m src.jobs.load_bea_to_db
"""

import click
import psycopg

from src.config.settings import settings
from src.pipelines.bea_load_to_db import run_bea_load_to_db_pipeline
from src.utils.logging import configure_logging


@click.command()
def main() -> None:
    """Load data/silver/bea/bea_silver.parquet into silver.bea_nipa."""
    configure_logging()

    parquet_path = settings.paths.silver / "bea" / "bea_silver.parquet"
    with psycopg.connect(**settings.postgres_connection_params) as connection:
        run_bea_load_to_db_pipeline(parquet_path, connection)

    click.echo(f"Loaded {parquet_path} into silver.bea_nipa")


if __name__ == "__main__":
    main()
