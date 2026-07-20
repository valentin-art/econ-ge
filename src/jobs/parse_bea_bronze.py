"""CLI entry point: parse extracted BEA JSON (external) into bronze parquet.

No network — operates on JSON already downloaded by jobs/extract_bea.py.
Standalone, no orchestrator dependency:
    uv run python -m src.jobs.parse_bea_bronze
"""

import click

from src.config.settings import settings
from src.pipelines.bea_parse_pipeline import parse_bea_tables
from src.utils.logging import configure_logging


@click.command()
def main() -> None:
    """Parse every BEA table in sources.BEA_TABLES from external JSON to bronze parquet."""
    configure_logging()

    bronze_dir = settings.paths.bronze / "bea"
    paths = parse_bea_tables(settings.paths.external / "bea", bronze_dir)

    click.echo(f"Parsed {len(paths)} BEA tables to {bronze_dir}")


if __name__ == "__main__":
    main()
