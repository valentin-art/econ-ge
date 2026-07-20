"""Dagster assets for the BEA pipeline: thin wrappers over src/pipelines/bea_*.

Each asset mirrors the corresponding CLI in src/jobs/ 1:1 -- those CLIs remain
a valid standalone way to run each stage; these assets exist so the stages
show up as one materializable, cacheable graph in the Dagster UI.
"""

import json

import psycopg
from dagster import AssetExecutionContext, MetadataValue, Output, asset

from src.config.settings import settings
from src.input_output.parquet import write_parquet
from src.pipelines.bea_extract_pipeline import extract_bea_tables
from src.pipelines.bea_load_to_db import run_bea_load_to_db_pipeline
from src.pipelines.bea_parse_pipeline import parse_bea_tables
from src.pipelines.bea_silver_pipeline import run_bea_silver_pipeline
from src.schemas.silver.bea_silver import column_descriptions


@asset(group_name="external", compute_kind="beaapi")
def bea_external(context: AssetExecutionContext) -> Output[None]:
    """Downloads every BEA table in sources.BEA_TABLES to data/external/bea/."""
    records = extract_bea_tables(settings.bea_api_key)
    target_dir = settings.paths.external / "bea"
    return Output(
        None,
        metadata={
            "n_tables": len(records),
            "target_dir": MetadataValue.path(str(target_dir)),
        },
    )


@asset(group_name="bronze_bea", compute_kind="pandas", deps=[bea_external])
def bea_bronze(context: AssetExecutionContext) -> Output[None]:
    """Parses extracted BEA JSON into bronze parquet, one file per table."""
    bronze_dir = settings.paths.bronze / "bea"
    paths = parse_bea_tables(settings.paths.external / "bea", bronze_dir)
    return Output(
        None,
        metadata={
            "n_tables": len(paths),
            "bronze_dir": MetadataValue.path(str(bronze_dir)),
        },
    )


@asset(group_name="silver_bea", compute_kind="pandas", deps=[bea_bronze])
def bea_silver(context: AssetExecutionContext) -> Output[None]:
    """Runs the bronze->silver BEA transform and writes the silver panel."""
    bronze_dir = settings.paths.bronze / "bea"
    silver = run_bea_silver_pipeline(bronze_dir)

    out_dir = settings.paths.silver / "bea"
    parquet_path = write_parquet(silver.reset_index(), out_dir / "bea_silver.parquet")

    columns_path = out_dir / "bea_silver_columns.json"
    columns_path.parent.mkdir(parents=True, exist_ok=True)
    columns_path.write_text(json.dumps(column_descriptions(), indent=2) + "\n")

    return Output(
        None,
        metadata={
            "n_years": len(silver),
            "path": MetadataValue.path(str(parquet_path)),
        },
    )


@asset(
    key=["silver", "bea_nipa"],
    group_name="postgres_load",
    compute_kind="postgres",
    deps=[bea_silver],
)
def bea_nipa_in_postgres(context: AssetExecutionContext) -> Output[None]:
    """Loads data/silver/bea/bea_silver.parquet into silver.bea_nipa (TRUNCATE + COPY).

    The asset key is set to ["silver", "bea_nipa"] to match the key dagster-dbt
    generates for the `silver.bea_nipa` dbt source (dbt/models/sources.yml), so
    this asset and the dbt models in the separate dbt/ code location connect
    into one lineage graph (see dbt/dagster_defs.py and workspace.yaml).
    """
    parquet_path = settings.paths.silver / "bea" / "bea_silver.parquet"
    with psycopg.connect(**settings.postgres_connection_params) as connection:
        run_bea_load_to_db_pipeline(parquet_path, connection)

    return Output(None, metadata={"table": "silver.bea_nipa"})
