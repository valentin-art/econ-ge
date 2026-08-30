from pathlib import Path

import pytest
import yaml

from src.config.parsing import (
    load_collection_expected_columns,
    load_expected_columns,
    parsing_config_path,
)

PARSING_CONFIG_ROOT = Path(__file__).parents[3] / "config" / "parsing"


def _write_config(root: Path, payload: dict) -> Path:
    path = parsing_config_path(root, "ipums", "cps")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_loads_declared_columns(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path, {"collection": "cps", "expected_columns": ["YEAR", "AGE", "SEX"]}
    )

    assert load_expected_columns(path) == frozenset({"YEAR", "AGE", "SEX"})


def test_missing_config_is_not_an_error(tmp_path: Path) -> None:
    # No contract declared means "derive it from bronze", not "expect nothing".
    assert load_expected_columns(tmp_path / "absent.yaml") is None


def test_rejects_an_empty_expected_columns_list(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"collection": "cps", "expected_columns": []})

    with pytest.raises(ValueError, match="declares an empty"):
        load_expected_columns(path)


def test_rejects_a_falsy_non_list_expected_columns(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"collection": "cps", "expected_columns": []})

    with pytest.raises(ValueError, match="declares an empty"):
        load_expected_columns(path)

    path2 = _write_config(tmp_path, {"collection": "cps", "expected_columns": {}})
    with pytest.raises(ValueError, match="not a list of strings"):
        load_expected_columns(path2)


def test_config_without_expected_columns_returns_none(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"collection": "cps"})

    assert load_expected_columns(path) is None


def test_rejects_expected_columns_that_is_not_a_list_of_strings(
    tmp_path: Path,
) -> None:
    # Silently reading a malformed contract as a narrower one would refuse
    # valid extracts and mark good years as damaged.
    path = _write_config(tmp_path, {"expected_columns": {"YEAR": 1}})

    with pytest.raises(ValueError, match="not a list of strings"):
        load_expected_columns(path)


def test_load_collection_expected_columns_rejects_a_mismatched_collection(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, {"collection": "asec", "expected_columns": ["YEAR"]})

    with pytest.raises(ValueError, match="declares collection"):
        load_collection_expected_columns(tmp_path, "ipums", "cps")


def test_rejects_duplicate_columns(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"expected_columns": ["YEAR", "AGE", "YEAR"]})

    with pytest.raises(ValueError, match="duplicate columns"):
        load_expected_columns(path)


def test_load_collection_expected_columns_reads_columns(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, {"expected_columns": ["YEAR", "AGE"]})

    assert load_collection_expected_columns(tmp_path, "ipums", "cps") == frozenset(
        {"YEAR", "AGE"}
    )


def test_shipped_cps_config_matches_what_the_pipeline_expects() -> None:
    # The committed contract must stay loadable and non-empty: a typo here
    # silently changes what every parse run treats as damage.

    columns = load_collection_expected_columns(PARSING_CONFIG_ROOT, "ipums", "cps")
    assert columns is not None
    assert {"YEAR", "AGE", "SEX", "INCWAGE"} <= columns


def test_settings_default_points_at_the_shipped_parsing_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.settings import Settings

    monkeypatch.delenv("PARSING_CONFIG_ROOT", raising=False)
    config_root = Settings().parsing_config_root

    assert config_root == PARSING_CONFIG_ROOT


def test_rejects_a_config_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = parsing_config_path(tmp_path, "ipums", "cps")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("- YEAR\n- AGE\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_expected_columns(path)
