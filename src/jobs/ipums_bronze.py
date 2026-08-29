"""CLI entry point: report and repair the column structure of IPUMS bronze.

Operates on parquet already written by the parse pipeline - no network, no
re-download:

    uv run python -m src.jobs.ipums_bronze check
    uv run python -m src.jobs.ipums_bronze repair --year 2006
"""

from collections.abc import Collection

import click

from src.config.parsing import load_collection_expected_columns
from src.config.settings import settings
from src.parsers.ipums import bronze_columns_by_year
from src.pipelines.ipums_parse_pipeline import (
    check_bronze_columns,
    repair_bronze_years,
)
from src.utils.logging import configure_logging

_MAX_LISTED_COLUMNS = 12


def _columns_line(columns: tuple[str, ...]) -> str:
    """Comma-joined column names, truncated so one badly-scoped contract does
    not bury the report under every column of every year."""
    if len(columns) <= _MAX_LISTED_COLUMNS:
        return ", ".join(columns)
    shown = ", ".join(columns[:_MAX_LISTED_COLUMNS])
    return f"{shown}, ... (+{len(columns) - _MAX_LISTED_COLUMNS} more)"


def _describe_columns(columns: Collection[str] | None) -> str:
    """Short human-readable summary of a contract, for CLI output."""
    if columns is None:
        return "derived from bronze"
    return f"{len(columns)} declared column(s)"


def _expected_columns(collection: str) -> frozenset[str] | None:
    """The declared contract for `collection`, or None to derive it."""
    return load_collection_expected_columns(
        settings.parsing_config_root, "ipums", collection
    )


@click.group()
def main() -> None:
    """Inspect and repair the bronze layer of an IPUMS collection."""


@main.command()
@click.option("--collection", default="cps", show_default=True)
def check(collection: str) -> None:
    """Report bronze years missing expected columns; exit 1 if any do."""
    configure_logging()

    bronze_dir = settings.paths.bronze / "ipums"
    expected = _expected_columns(collection)
    click.echo(f"{collection}: contract is {_describe_columns(expected)}.")

    observed = bronze_columns_by_year(bronze_dir, collection)
    if not observed:
        raise click.ClickException(
            f"{collection}: no bronze year found under {bronze_dir / collection}"
        )

    deviations = check_bronze_columns(bronze_dir, collection, expected_columns=expected)
    if not deviations:
        click.echo(
            f"{collection}: all {len(observed)} bronze year(s) hold the "
            f"expected columns."
        )
        return

    click.echo(f"{collection}: {len(deviations)} year(s) deviate.")
    for year, (missing, extra) in deviations.items():
        click.echo(f"  {year}: missing {len(missing)} column(s)")
        click.echo(f"    missing: {_columns_line(missing)}")
        if extra:
            click.echo(f"    extra:   {_columns_line(extra)}")
    raise SystemExit(1)


@main.command()
@click.option("--collection", default="cps", show_default=True)
@click.option(
    "--year",
    "years",
    multiple=True,
    type=int,
    help="Repair only these years. Default: every deviating year.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report the years that would be rebuilt, without writing.",
)
def repair(collection: str, years: tuple[int, ...], dry_run: bool) -> None:
    """Rebuild bronze years missing expected columns, from extracts on disk."""
    configure_logging()

    bronze_dir = settings.paths.bronze / "ipums"
    external_dir = settings.paths.external / "ipums"
    expected = _expected_columns(collection)

    targets = (
        sorted(years)
        if years
        else sorted(
            check_bronze_columns(bronze_dir, collection, expected_columns=expected)
        )
    )
    observed = bronze_columns_by_year(bronze_dir, collection)

    if not targets:
        # repair_bronze_years draws this distinction itself, but the early
        # return below means its log never fires through the CLI.
        state = (
            f"no bronze year found under {bronze_dir / collection}"
            if not observed
            else f"all {len(observed)} bronze year(s) conform"
        )
        click.echo(f"{collection}: {state}; nothing to repair.")
        return

    if dry_run:
        click.echo(
            f"{collection}: would rebuild {targets} from {external_dir} "
            f"against {_describe_columns(expected)}."
        )
        return
    try:
        bronze_paths = repair_bronze_years(
            external_dir,
            bronze_dir,
            collection,
            years=targets,
            expected_columns=expected,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"{collection}: rebuilt {len(bronze_paths)} bronze file(s) for {targets}."
    )


if __name__ == "__main__":
    main()
