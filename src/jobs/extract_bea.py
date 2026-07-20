"""CLI entry point: download BEA FixedAssets/NIPA tables (external, raw JSON).

Standalone, no orchestrator dependency:
    uv run python -m src.jobs.extract_bea
"""

import click

from src.config.settings import settings
from src.pipelines.bea_extract_pipeline import extract_bea_tables
from src.utils.logging import configure_logging


@click.command()
def main() -> None:
    """Extract every BEA table in sources.BEA_TABLES to data/external/bea/."""
    configure_logging()

    records = extract_bea_tables(settings.bea_api_key)

    click.echo(
        f"Extracted {len(records)} BEA tables to {settings.paths.external / 'bea'}"
    )


if __name__ == "__main__":
    main()
