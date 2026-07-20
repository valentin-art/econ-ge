"""Load the BEA silver panel into Postgres.

Reads data/silver/bea/bea_silver.parquet (produced by
pipelines.bea_silver_pipeline / jobs.transform_bea_silver), validates it
against the same schema used to build it, and bulk-loads it into the
existing `silver.bea_nipa` table via a full TRUNCATE + COPY refresh. The
table has no unique constraint on `year` and the parquet is always the full
regenerated panel, so a full refresh is the correct idempotent behavior for
re-runs.
"""

from pathlib import Path

import structlog

from src.input_output.parquet import read_parquet
from src.input_output.sqldb import load_to_db
from src.schemas.silver.bea_silver import BeaSilverRow, validate_bea_silver

log = structlog.get_logger(__name__)

TABLE_NAME = "silver.bea_nipa"


def run_bea_load_to_db_pipeline(parquet_path: Path, connection) -> None:
    """Load the BEA silver parquet panel into `silver.bea_nipa`.

    Parameters
    ----------
    parquet_path : path to the silver parquet, e.g.
        settings.paths.silver / "bea" / "bea_silver.parquet"
    connection : an open psycopg connection (caller owns open/close)
    """
    log.info("bea_load_to_db_start", parquet_path=str(parquet_path))

    validate_bea_silver(read_parquet(parquet_path).set_index("year"))

    columns = list(BeaSilverRow.model_fields)
    load_to_db(
        str(parquet_path),
        TABLE_NAME,
        columns,
        connection,
        file_type="parquet",
        truncate=True,
    )

    log.info("bea_load_to_db_complete", table=TABLE_NAME)
