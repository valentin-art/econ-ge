import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
import structlog
import yaml
from pydantic import ValidationError

from src.cleaning.context import (
    CleaningContext,
    DeflatorTableConfig,
    SourceProfile,
    TopcodeConfig,
    YearBandThreshold,
)
from src.harmonization.cps_tables import CPI_DEFLATOR, GDP_PCE_DEFLATOR

FIXTURES = Path(__file__).parent / "fixtures" / "config"
PRODUCTION_CONFIG = Path(__file__).parents[3] / "config" / "cleaning" / "cps"
CONFIG_ROOT = Path(__file__).parents[3] / "config" / "cleaning"
KNOWN_UNCOVERED_YEARS = {"error", "warning"}


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


def test_context_survives_pickle_deepcopy_and_json_dump() -> None:
    # Regression: freezing topcode/crosswalks into MappingProxyType at
    # construction previously broke pickle (needed by multiprocess
    # executors like Dagster's) and model_dump(mode="json").
    import copy
    import pickle

    context = CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        topcode={
            "wage": TopcodeConfig(
                uncovered_years="warning",
                thresholds=[
                    YearBandThreshold(
                        start_year=2000,
                        end_year=2000,
                        threshold=1.0,
                        match_mode="gte",
                    )
                ],
            )
        },
    )

    assert pickle.loads(pickle.dumps(context)).compute_hash() == context.compute_hash()
    assert copy.deepcopy(context).compute_hash() == context.compute_hash()
    assert context.model_dump(mode="json")["source_profile"]["kind"] == "ipums_cps_asec"


def test_json_dump_works_once_a_crosswalk_is_actually_loaded() -> None:
    # Regression: pl.DataFrame is an arbitrary type pydantic cannot serialize,
    # so model_dump(mode="json") raised PydanticSerializationError the moment
    # any crosswalk was present. The test above cannot catch this - its
    # fixture has no crosswalks, which is exactly why the bug survived it.
    context = CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        crosswalks={"occ": pl.DataFrame({"code": [1, 2], "label": ["a", "b"]})},
    )

    dumped = context.model_dump(mode="json", by_alias=True)

    assert dumped["crosswalks"] == {
        "occ": [{"code": 1, "label": "a"}, {"code": 2, "label": "b"}]
    }
    # Must be genuinely JSON-encodable, not merely dumpable.
    assert json.loads(json.dumps(dumped))["crosswalks"]["occ"][0]["label"] == "a"


def test_aliased_dump_round_trips_back_through_model_validate() -> None:
    # populate_by_name: without it the by_alias dump carries `topcode`/
    # `deflators`/`crosswalks` while the fields are named with a trailing
    # underscore, so a dump could never be re-validated into a context.
    context = CleaningContext(
        source_profile=SourceProfile(kind="ipums_cps_asec"),
        deflators={
            "cpi": DeflatorTableConfig(values={1980: 1.5}, uncovered_years="warning")
        },
    )

    restored = CleaningContext.model_validate(context.model_dump(mode="json"))

    assert restored.compute_hash() == context.compute_hash()
    assert dict(restored.deflators["cpi"].values) == {1980: 1.5}


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
    # No crosswalks_dir: there is no config/cleaning/crosswalks/ directory.
    # Passing one anyway would be silently tolerated and assert nothing.
    context = CleaningContext.from_config(
        config_dir=PRODUCTION_CONFIG, source="ipums_cps_asec"
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

    # config/cleaning/cps/deflators/*.yaml is transcribed exactly from
    # src.harmonization.cps_tables, which stays the source of truth.
    assert dict(context.deflators["cpi"].values) == CPI_DEFLATOR
    assert dict(context.deflators["gdp_pce"].values) == GDP_PCE_DEFLATOR
    assert context.deflators["cpi"].uncovered_years == "warning"


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
                "uncovered_years": "warning",
                "per_year": {1996: 150000},
            }
        )
    )
    (topcode_dir / "income.yaml").write_text(
        yaml.safe_dump(
            {
                "multiplier": 2.0,
                "uncovered_years": "warning",
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
            "uncovered_years": "warning",
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


def test_empty_thresholds_is_rejected() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        TopcodeConfig(uncovered_years="warning", thresholds=[])


def test_inverted_single_band_is_rejected() -> None:
    # Regression: start_year > end_year must be caught even with only one
    # band - is_between(start, end) would silently match nothing.
    with pytest.raises(ValueError, match="start_year > end_year"):
        TopcodeConfig(
            uncovered_years="warning",
            thresholds=[
                YearBandThreshold(
                    start_year=1990, end_year=1980, threshold=5.0, match_mode="gte"
                )
            ],
        )


def test_inverted_last_band_among_several_is_rejected() -> None:
    with pytest.raises(ValueError, match="start_year > end_year"):
        TopcodeConfig(
            uncovered_years="warning",
            thresholds=[
                YearBandThreshold(
                    start_year=1980, end_year=1985, threshold=5.0, match_mode="gte"
                ),
                YearBandThreshold(
                    start_year=2000, end_year=1990, threshold=6.0, match_mode="gte"
                ),
            ],
        )


def test_overlapping_bands_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlapping bands"):
        TopcodeConfig(
            uncovered_years="warning",
            thresholds=[
                YearBandThreshold(
                    start_year=1980, end_year=1990, threshold=5.0, match_mode="gte"
                ),
                YearBandThreshold(
                    start_year=1985, end_year=1995, threshold=6.0, match_mode="gte"
                ),
            ],
        )


def test_thresholds_cannot_be_mutated_after_construction() -> None:
    config = TopcodeConfig(
        uncovered_years="warning",
        thresholds=[
            YearBandThreshold(
                start_year=1980, end_year=1990, threshold=5.0, match_mode="gte"
            )
        ],
    )

    assert isinstance(config.thresholds, tuple)
    with pytest.raises(AttributeError):
        config.thresholds.append(  # type: ignore[attr-defined]
            YearBandThreshold(
                start_year=2000, end_year=2000, threshold=6.0, match_mode="gte"
            )
        )


def test_deflator_table_config_empty_values_is_rejected() -> None:
    with pytest.raises(ValueError, match="values"):
        DeflatorTableConfig(values={}, uncovered_years="warning")


def test_deflator_table_config_values_view_cannot_be_mutated() -> None:
    config = DeflatorTableConfig(values={1980: 1.5}, uncovered_years="warning")

    assert dict(config.values) == {1980: 1.5}
    with pytest.raises(TypeError):
        config.values[1990] = 2.0  # type: ignore[index]


def test_deflator_table_config_survives_pickle_and_json_dump() -> None:
    import pickle

    config = DeflatorTableConfig(values={1980: 1.5, 1981: 1.6}, uncovered_years="error")

    assert pickle.loads(pickle.dumps(config)).values == config.values
    dumped = config.model_dump(mode="json", by_alias=True)
    assert dumped["values"] == {"1980": 1.5, "1981": 1.6}
    assert dumped["uncovered_years"] == "error"


def _iter_uncovered_years(node: object) -> Iterator[object]:
    """Every `uncovered_years` value anywhere in a parsed YAML tree."""
    if isinstance(node, dict):
        if "uncovered_years" in node:
            yield node["uncovered_years"]
        for value in node.values():
            yield from _iter_uncovered_years(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_uncovered_years(item)


def test_config_root_is_discoverable() -> None:
    # Without this, the parametrised test below would silently collect zero
    # cases (and pass) if CONFIG_ROOT ever moved.
    assert sorted(CONFIG_ROOT.rglob("*.yaml"))


@pytest.mark.parametrize(
    "config_path", sorted(CONFIG_ROOT.rglob("*.yaml")), ids=lambda p: str(p.name)
)
def test_every_config_declares_a_known_uncovered_years_policy(
    config_path: Path,
) -> None:
    # The model coerces an unrecognised policy to "error", so a stale ("skip")
    # or typo'd value would NOT fail to load - it would silently become strict.
    # Read the raw YAML, before any coercion.
    raw = yaml.safe_load(config_path.read_text()) or {}
    for value in _iter_uncovered_years(raw):
        assert value in KNOWN_UNCOVERED_YEARS, (
            f"{config_path} declares uncovered_years={value!r}; "
            f"allowed: {sorted(KNOWN_UNCOVERED_YEARS)}"
        )


@pytest.mark.parametrize(
    "given", ["skpi", "WARNING", " warning ", "", None, True, 123, [], {}]
)
def test_unknown_uncovered_years_falls_back_to_error_and_logs(given: object) -> None:
    # Fail closed: a policy that cannot be read must never become the
    # permissive one. The log line is the only trace, hence the assertion.
    with structlog.testing.capture_logs() as logs:
        config = DeflatorTableConfig(values={1980: 1.5}, uncovered_years=given)

    assert config.uncovered_years == "error"
    assert [entry["event"] for entry in logs] == ["uncovered_years_coerced"]


def test_renamed_skip_policy_is_rejected_not_coerced() -> None:
    # A stale `skip` must fail loudly: coercing it to "error" would turn a
    # tolerant production config strict with nothing in the diff to show it.
    with pytest.raises(ValidationError, match="renamed to 'warning'"):
        DeflatorTableConfig(values={1980: 1.5}, uncovered_years="skip")


def test_uncovered_years_stays_required() -> None:
    # Missing is not the same as unreadable: declaring the policy is a
    # methodology decision, so it is never inferred.
    with pytest.raises(ValidationError, match="uncovered_years"):
        DeflatorTableConfig(values={1980: 1.5})


def test_production_config_needs_no_uncovered_years_coercion() -> None:
    with structlog.testing.capture_logs() as logs:
        CleaningContext.from_config(
            config_dir=PRODUCTION_CONFIG, source="ipums_cps_asec"
        )

    assert [e for e in logs if e["event"] == "uncovered_years_coerced"] == []


def test_stale_skip_in_a_config_file_names_the_file(tmp_path: Path) -> None:
    # from_config wraps ValidationError as ValueError - the file path is what
    # makes the rename actionable.
    (tmp_path / "source_profile.yaml").write_text(
        yaml.safe_dump({"kind": "ipums_cps_asec"})
    )
    deflators_dir = tmp_path / "deflators"
    deflators_dir.mkdir()
    (deflators_dir / "cpi.yaml").write_text(
        yaml.safe_dump({"values": {1980: 1.5}, "uncovered_years": "skip"})
    )

    with pytest.raises(ValueError) as excinfo:
        CleaningContext.from_config(config_dir=tmp_path, source="ipums_cps_asec")

    message = str(excinfo.value)
    assert "cpi.yaml" in message
    assert "renamed to 'warning'" in message


def test_wage_topcode_has_no_unexplained_threshold_collapse() -> None:
    # FAILS TODAY, deliberately. wage.yaml drops 99999 (1995) -> 25000 (1996),
    # a 4x fall with no documented CPS methodology change, and holds it through
    # 2002. Against real bronze data that flags 1.6% of wage records in 1995 but
    # 37.3% in 1996 and 47.3% in 2001 - and each flagged row is multiplied by
    # 1.5, so the step is inflating the middle of the wage distribution rather
    # than the censored top tail. Verify the block against
    # aa_clean/clean7909km.do and correct it; this turns green then.
    context = CleaningContext.from_config(
        config_dir=PRODUCTION_CONFIG, source="ipums_cps_asec"
    )
    by_year = {
        year: band.threshold
        for band in context.topcode["wage"].thresholds
        for year in range(band.start_year, band.end_year + 1)
    }
    collapses = [
        (year, by_year[year - 1], by_year[year])
        for year in sorted(by_year)
        if year - 1 in by_year and by_year[year] * 2 < by_year[year - 1]
    ]

    assert collapses == []
