from click.testing import CliRunner

from src.config.settings import settings
from src.jobs.ipums_bronze import main


def test_check_fails_when_bronze_is_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings.paths, "root", tmp_path)
    result = CliRunner().invoke(main, ["check", "--collection", "cps"])
    assert result.exit_code != 0
