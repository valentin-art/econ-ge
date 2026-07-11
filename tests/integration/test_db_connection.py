"""Test connection to local Postgres.

Use:
    pytest tests/integration/test_db_connection.py -v
"""

import os
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

# Load env vars from .env in project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@pytest.fixture(scope="module")
def conn_str() -> str:
    required = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "POSTGRES_HOST_PORT",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        pytest.fail(f"Missing env vars: {missing}. Did you create .env?")
    return (
        f"host=localhost "
        f"port={os.environ['POSTGRES_HOST_PORT']} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']} "
        f"connect_timeout=5"
    )


def test_connects_and_selects_one(conn_str: str) -> None:
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1;")
        assert cur.fetchone() == (1,)


def test_server_version(conn_str: str) -> None:
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute("SHOW server_version;")
        version = cur.fetchone()[0]  # type: ignore
        assert version.startswith("17."), f"Expected Postgres 17.x, got {version}"


def test_expected_extensions_loaded(conn_str: str) -> None:
    """Test that expected extensions are loaded."""
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute("SELECT extname FROM pg_extension;")
        extensions = {row[0] for row in cur.fetchall()}
        assert {"uuid-ossp", "pgcrypto"}.issubset(extensions)
