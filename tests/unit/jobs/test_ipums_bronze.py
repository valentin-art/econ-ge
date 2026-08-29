from pathlib import Path

import pytest
from click.testing import CliRunner

from src.config.settings import settings
from src.jobs.ipums_bronze import main


def test_check_fails_when_bronze_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.paths, "root", tmp_path)

    result = CliRunner().invoke(main, ["check", "--collection", "cps"])

    # Not just "it failed": a broken parsing config fails the command too,
    # without ever reaching the guard under test.
    assert result.exit_code != 0
    assert "no bronze year found" in result.output
