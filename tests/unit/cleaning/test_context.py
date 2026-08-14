from pathlib import Path

import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from src.cleaning.context import CleaningContext, SourceProfile, TopcodeConfig

FIXTURES = Path(__file__).parent / "fixtures" / "config"
PRODUCTION_CONFIG = Path(__file__).parents[3] / "config" / "cleaning" / "cps"


def _context(**overrides: object) -> CleaningContext:
    return CleaningContext.from_config(
        config_dir=FIXTURES / "cps",
        source="ipums_cps_asec",
        crosswalks_dir=FIXTURES / "crosswalks",
        overrides=overrides or None,
    )


def test_from_config_builds_context_from_fixture_yaml() -> None:
    context = _context()

    assert context.source_profile.kind == "ipums_cps_asec"
    assert "Stage A" in context.source_profile.notes


def test_from_config_loads_crosswalk_csv_into_polars_dataframe() -> None:
    context = _context()

    assert isinstance(context.crosswalks["sample"], pl.DataFrame)
    assert context.crosswalks["sample"].columns == ["code", "label"]


def test_from_config_raises_when_source_profile_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_profile.yaml"):
        CleaningContext.from_config(
            config_dir=tmp_path, source="ipums_cps_asec", crosswalks_dir=tmp_path
        )


def test_from_config_raises_when_source_profile_malformed(tmp_path: Path) -> None:
    (tmp_path / "source_profile.yaml").write_text(
        yaml.safe_dump({"kind": "not_a_real_source"})
    )

    with pytest.raises(ValueError):
        CleaningContext.from_config(
            config_dir=tmp_path, source="ipums_cps_asec", crosswalks_dir=tmp_path
        )


def test_from_config_raises_when_source_mismatch(tmp_path: Path) -> None:
    (tmp_path / "source_profile.yaml").write_text(yaml.safe_dump({"kind": "psid"}))

    with pytest.raises(ValueError, match="psid"):
        CleaningContext.from_config(
            config_dir=tmp_path, source="ipums_cps_asec", crosswalks_dir=tmp_path
        )


def test_context_is_frozen() -> None:
    context = CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))

    with pytest.raises(ValidationError):
        context.source_profile = SourceProfile(kind="psid")


def test_compute_hash_stable_for_identical_inputs() -> None:
    assert _context().compute_hash() == _context().compute_hash()


def test_compute_hash_changes_with_overrides() -> None:
    baseline = _context()
    overridden = _context(notes="a different note")

    assert baseline.compute_hash() != overridden.compute_hash()


def test_compute_hash_ignores_run_id() -> None:
    context_a = CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))
    context_b = CleaningContext(source_profile=SourceProfile(kind="ipums_cps_asec"))

    assert context_a.run_metadata.run_id != context_b.run_metadata.run_id
    assert context_a.compute_hash() == context_b.compute_hash()


def test_from_config_defaults_optional_subcontexts_when_yaml_absent() -> None:
    # The test fixture dir has no topcode/ directory -
    # from_config must tolerate that and return an empty dict, not raise.
    context = _context()

    assert context.topcode == {}


def test_from_config_loads_real_production_config() -> None:
    context = CleaningContext.from_config(
        config_dir=PRODUCTION_CONFIG,
        source="ipums_cps_asec",
        crosswalks_dir=PRODUCTION_CONFIG.parent / "crosswalks",
    )

    wage = context.topcode["wage"]
    assert wage.multiplier == 1.5
    # 5 pre-1988 bands + 22 individual 1988-2009 years, folded into one
    # canonical `thresholds` list (see TopcodeConfig._fold_threshold_blocks).
    assert len(wage.thresholds) == 27
    band_2009 = next(b for b in wage.thresholds if b.start_year == 2009)
    assert band_2009.end_year == 2009
    assert band_2009.threshold == 35000
    assert band_2009.match_mode == "gte"
    band_1962 = next(b for b in wage.thresholds if b.start_year == 1962)
    assert band_1962.match_mode == "exact"


def test_from_config_loads_multiple_named_topcode_instances(tmp_path: Path) -> None:
    # Two files under topcode/ become two independently-keyed TopcodeConfig
    # instances - no new CleaningContext field needed per instance.
    (tmp_path / "source_profile.yaml").write_text(
        yaml.safe_dump({"kind": "ipums_cps_asec"})
    )
    topcode_dir = tmp_path / "topcode"
    topcode_dir.mkdir()
    (topcode_dir / "wage.yaml").write_text(
        yaml.safe_dump(
            {
                "multiplier": 1.5,
                "uncovered_years": "skip",
                "per_year": {1996: 150000},
            }
        )
    )
    (topcode_dir / "income.yaml").write_text(
        yaml.safe_dump(
            {
                "multiplier": 2.0,
                "uncovered_years": "skip",
                "per_year": {1996: 150000},
            }
        )
    )

    context = CleaningContext.from_config(
        config_dir=tmp_path, source="ipums_cps_asec", crosswalks_dir=tmp_path
    )

    assert context.topcode["wage"].multiplier == 1.5
    assert context.topcode["income"].multiplier == 2.0


def test_topcode_config_folds_pre_and_post_1988_blocks_into_one_list() -> None:
    # Authoring shorthand: two shape-identified raw blocks (a list of bands,
    # a compact {year: value} mapping) fold into the single `thresholds`
    # field - TopcodeConfig itself exposes no separate attribute per block.
    config = TopcodeConfig.model_validate(
        {
            "uncovered_years": "skip",
            "bands": [{"start_year": 1962, "end_year": 1964, "threshold": 90000}],
            "per_year": {1988: 99999, 1989: 95000},
        }
    )

    assert not hasattr(config, "bands")
    assert not hasattr(config, "per_year")
    assert [
        (b.start_year, b.end_year, b.threshold, b.match_mode) for b in config.thresholds
    ] == [
        (1962, 1964, 90000, "exact"),
        (1988, 1988, 99999, "gte"),
        (1989, 1989, 95000, "gte"),
    ]


def test_from_config_raises_when_optional_subcontext_malformed(tmp_path: Path) -> None:
    (tmp_path / "source_profile.yaml").write_text(
        yaml.safe_dump({"kind": "ipums_cps_asec"})
    )
    topcode_dir = tmp_path / "topcode"
    topcode_dir.mkdir()
    (topcode_dir / "wage.yaml").write_text(
        yaml.safe_dump({"multiplier": "not_a_float"})
    )

    with pytest.raises(ValueError, match="wage.yaml"):
        CleaningContext.from_config(
            config_dir=tmp_path, source="ipums_cps_asec", crosswalks_dir=tmp_path
        )
