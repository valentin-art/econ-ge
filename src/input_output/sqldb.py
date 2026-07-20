import gzip
import io
from typing import List, Literal, Optional

import psycopg
import pyarrow.parquet as pq
import structlog
from psycopg.sql import SQL, Identifier

log = structlog.get_logger(__name__)


def query_db(query: str, connection_params: dict) -> Optional[list]:
    """
    Fetch data from a SQL database using the provided query and connection parameters.

    Args:
        query (str): The SQL query to execute.
        connection_params (dict): A dictionary containing the connection parameters
            (e.g., host, database, user, password).
    """
    try:
        with psycopg.connect(**connection_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query.encode())
                return cursor.fetchall()
    except Exception:
        log.exception("An error occurred while querying the database")
        return None


def load_to_db(
    file_path: str,
    table_name: str,
    columns: List[str],
    connection,
    file_type: Literal["csv", "parquet"],
    compressed: bool = False,
    chunk_size: int = 1000,
    truncate: bool = False,
) -> None:
    """Bulk load data from a file into a SQL database table.

    `table_name` may be schema-qualified (e.g. "silver.bea_nipa"). If `truncate`
    is set, the table is truncated before the load, in the same transaction.
    """
    identifier = Identifier(*table_name.split("."))
    copy_sql = SQL(
        "COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, HEADER TRUE)"
    ).format(
        table=identifier,
        cols=SQL(", ").join(Identifier(col) for col in columns),
    )
    copy_sql_no_header = SQL(
        "COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)"
    ).format(
        table=identifier,
        cols=SQL(", ").join(Identifier(col) for col in columns),
    )

    with connection.cursor() as cursor:
        if truncate:
            cursor.execute(SQL("TRUNCATE TABLE {table}").format(table=identifier))

        if file_type == "csv":
            opener = gzip.open if compressed else open
            with opener(file_path, "rt") as f, cursor.copy(copy_sql) as copy:
                while data := f.read(8192):
                    copy.write(data)

        elif file_type == "parquet":
            parquet_file = pq.ParquetFile(file_path)
            with cursor.copy(copy_sql_no_header) as copy:
                for batch in parquet_file.iter_batches(
                    batch_size=chunk_size, columns=columns
                ):
                    chunk = batch.to_pandas()
                    buffer = io.StringIO()
                    chunk.to_csv(buffer, index=False, header=False)
                    copy.write(buffer.getvalue())
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    connection.commit()
