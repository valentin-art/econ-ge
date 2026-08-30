from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from src.config.settings import settings
from src.jobs.ipums_bronze import main


def _bronze_year(root: Path, collection: str, year: int, columns: list[str]) -> None:
    path = root / "bronze" / "ipums" / collection
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({c: [1] for c in columns}).to_parquet(path / f"{year}.parquet")


def test_check_fails_when_bronze_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.paths, "root", tmp_path)

    result = CliRunner().invoke(main, ["check", "--collection", "cps"])

    # Not just "it failed": a broken parsing config fails the command too,
    # without ever reaching the guard under test.
    assert result.exit_code != 0
    assert "no bronze year found" in result.output


def test_check_exits_nonzero_when_a_year_deviates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No config for "probe", so the contract is derived: 2005/2007 outvote 2006.
    monkeypatch.setattr(settings.paths, "root", tmp_path)
    _bronze_year(tmp_path, "probe", 2005, ["YEAR", "AGE", "SEX"])
    _bronze_year(tmp_path, "probe", 2006, ["YEAR", "AGE"])
    _bronze_year(tmp_path, "probe", 2007, ["YEAR", "AGE", "SEX"])

    result = CliRunner().invoke(main, ["check", "--collection", "probe"])

    assert result.exit_code == 1
    assert "2006" in result.output
    assert "SEX" in result.output
