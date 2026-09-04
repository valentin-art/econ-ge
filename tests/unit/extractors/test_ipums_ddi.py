"""Tests for extractors.ipums_ddi.

The load-bearing property here is that flags are recognised by their DDI label
and never by their name: IPUMS flag names are irregular (QACTNLFL truncated,
QUHRSWORKLY not; QWKSWORK shared by two variables) and ordinary variables can
begin with Q or T, so any name-shaped shortcut would be wrong in both
directions.
"""

from pathlib import Path

import pytest
import structlog.testing
from ipumspy import readers

from src.extractors.base import build_extraction_record
from src.extractors.ipums_ddi import (
    FLAG_PARSER_VERSION,
    collection_flag_registry,
    flag_columns_for,
    merge_column_names,
    parse_flag_label,
    summarize_ddi,
    summary_from_metadata,
    try_summarize_ddi,
)
from src.extractors.manifest import append_to_manifest, read_manifest


def _write_ddi(
    tmp_path: Path, make_ddi_xml, variables, name: str = "cps_00001"
) -> Path:
    ddi_path = tmp_path / f"{name}.xml"
    ddi_path.write_text(make_ddi_xml(variables), encoding="utf-8")
    return ddi_path


# --- label parsing ---------------------------------------------------------


def test_parse_flag_label_quality_simple() -> None:
    assert parse_flag_label("Data quality flag for INCWAGE") == (
        "quality",
        ("INCWAGE",),
    )


def test_parse_flag_label_strips_version_qualifier() -> None:
    # SRCEARN has both a general and a detailed flag; the qualifier
    # distinguishes the flag columns, not the source variable.
    assert parse_flag_label("Data quality flag for SRCEARN [general version]") == (
        "quality",
        ("SRCEARN",),
    )
    assert parse_flag_label("Data quality flag for SRCEARN [detailed version]") == (
        "quality",
        ("SRCEARN",),
    )


def test_parse_flag_label_splits_multiple_sources() -> None:
    assert parse_flag_label("Data quality flag for WKSWORK1 and WKSWORK2") == (
        "quality",
        ("WKSWORK1", "WKSWORK2"),
    )
    assert parse_flag_label("Data quality flag for TEST1, TEST2 and TEST3") == (
        "quality",
        ("TEST1", "TEST2", "TEST3"),
    )
    assert parse_flag_label("Data quality flag for TEST1, TEST2, and TEST3") == (
        "quality",
        ("TEST1", "TEST2", "TEST3"),
    )
    assert parse_flag_label("Data quality flag for TEST1, TEST2, TEST3, and TEST4") == (
        "quality",
        ("TEST1", "TEST2", "TEST3", "TEST4"),
    )
    assert parse_flag_label("Data quality flag for TEST1, TEST2, TEST3 and TEST4") == (
        "quality",
        ("TEST1", "TEST2", "TEST3", "TEST4"),
    )


def test_parse_flag_label_topcode() -> None:
    assert parse_flag_label("Topcode Flag for INCFARM") == ("topcode", ("INCFARM",))


def test_parse_flag_label_strips_trailing_punctuation() -> None:
    # A label written as a full sentence ("... for INCFARM.") must not lose
    # the flag entirely just because of trailing punctuation.
    assert parse_flag_label("Data quality flag for INCFARM.") == (
        "quality",
        ("INCFARM",),
    )
    assert parse_flag_label("Data quality flag for TEST1, TEST2, and TEST3.") == (
        "quality",
        ("TEST1", "TEST2", "TEST3"),
    )


@pytest.mark.parametrize(
    "label",
    [
        "Flag for ASEC",
        "Flag for the 3/8 file 2014",
        "Age",
        "Means of transportation to work",
        "Survey year",
        "",
    ],
)
def test_parse_flag_label_ignores_non_flag_labels(label: str) -> None:
    assert parse_flag_label(label) is None


def test_parse_flag_label_rejects_prose_tail() -> None:
    # A label that matches the prefix but whose tail is prose rather than
    # variable names must not be read as a flag for a made-up variable.
    assert parse_flag_label("Data quality flag for the whole household") is None


# --- summarizing a codebook ------------------------------------------------


def test_summarize_ddi_lists_delivered_variables_in_file_order(
    tmp_path: Path, make_ddi_xml
) -> None:
    variables = [
        ("YEAR", "Survey year", 4),
        ("AGE", "Age", 2),
        ("QAGE", "Data quality flag for AGE", 1),
    ]
    summary = summarize_ddi(_write_ddi(tmp_path, make_ddi_xml, variables))

    assert summary.variables == ("YEAR", "AGE", "QAGE")


def test_summarize_ddi_maps_sources_to_quality_and_topcode_flags(
    tmp_path: Path, make_ddi_xml, cps_flag_vars
) -> None:
    summary = summarize_ddi(_write_ddi(tmp_path, make_ddi_xml, cps_flag_vars))

    assert summary.quality_flags == {
        "ACTNLFLY": ("QACTNLFL",),
        "INCLONGJ": ("QINCLONG", "QINCLONGD"),
        "WKSWORK1": ("QWKSWORK", "QWKSWORKTEST"),
        "WKSWORK2": ("QWKSWORK", "QWKSWORKTEST"),
        "WKSWORKTEST": ("QWKSWORKTEST",),
        "UHRSWORKLY": ("QUHRSWORKLY",),
        "TEST1": ("QTEST1", "QTEST2", "QTEST3", "QTEST4"),
        "TEST2": ("QTEST1", "QTEST2", "QTEST3", "QTEST4"),
        "TEST3": ("QTEST1", "QTEST2", "QTEST3", "QTEST4"),
        "TEST4": ("QTEST3", "QTEST4"),
    }
    assert summary.topcode_flags == {"INCLONGJ": ("TINCLONGJ",)}


def test_summarize_ddi_never_classifies_by_name(
    tmp_path: Path, make_ddi_xml, cps_flag_vars
) -> None:
    """ASECFLAG/HFLAG have flag-ish labels and TRANWORK/TRANTIME are ordinary
    T-prefixed variables. None of them may be reported as a flag."""
    summary = summarize_ddi(_write_ddi(tmp_path, make_ddi_xml, cps_flag_vars))

    for name in ("ASECFLAG", "HFLAG", "TRANWORK", "TRANTIME"):
        assert summary.kind_of(name) is None
        assert name not in summary.flag_names


def test_summarize_ddi_kind_of_distinguishes_quality_from_topcode(
    tmp_path: Path, make_ddi_xml, cps_flag_vars
) -> None:
    summary = summarize_ddi(_write_ddi(tmp_path, make_ddi_xml, cps_flag_vars))

    assert summary.kind_of("QINCLONG") == "quality"
    assert summary.kind_of("TINCLONGJ") == "topcode"


def test_summarize_ddi_caches_on_path_mtime_and_size(
    tmp_path: Path, make_ddi_xml, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    real_read = readers.read_ipums_ddi

    def counting_read(path):
        calls["n"] += 1
        return real_read(path)

    monkeypatch.setattr(
        "src.extractors.ipums_ddi.readers.read_ipums_ddi", counting_read
    )

    variables = [("AGE", "Age", 2), ("QAGE", "Data quality flag for AGE", 1)]
    ddi_path = _write_ddi(tmp_path, make_ddi_xml, variables)

    summarize_ddi(ddi_path)
    summarize_ddi(ddi_path)
    assert calls["n"] == 1

    # A codebook re-downloaded to the same filename must not be served stale.
    rewritten = [("AGE", "Age", 2), ("SEX", "Sex", 1)]
    ddi_path.write_text(make_ddi_xml(rewritten), encoding="utf-8")
    import os

    stat = ddi_path.stat()
    os.utime(ddi_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert summarize_ddi(ddi_path).variables == ("AGE", "SEX")
    assert calls["n"] == 2


def test_try_summarize_ddi_returns_none_for_stub_codebook(tmp_path: Path) -> None:
    stub = tmp_path / "stub.xml"
    stub.write_text("<codeBook/>")

    assert try_summarize_ddi(stub) is None


def test_try_summarize_ddi_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert try_summarize_ddi(tmp_path / "nope.xml") is None


def test_try_summarize_ddi_does_not_cache_an_environment_failure(
    tmp_path: Path, make_ddi_xml, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError says nothing about the file - a dropped mount, too many open
    handles - so the next call must parse instead of being served the None.
    """
    ddi_path = _write_ddi(tmp_path, make_ddi_xml, [("AGE", "Age", 2)])
    real_read = readers.read_ipums_ddi
    calls = {"n": 0}

    def failing_once(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("EMFILE")
        return real_read(path)

    monkeypatch.setattr("src.extractors.ipums_ddi.readers.read_ipums_ddi", failing_once)

    assert try_summarize_ddi(ddi_path) is None

    summary = try_summarize_ddi(ddi_path)
    assert summary is not None
    assert summary.variables == ("AGE",)


def test_try_summarize_ddi_caches_a_parse_failure(
    tmp_path: Path, make_ddi_xml, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contrast to the OSError case: a file that does not parse stays
    unparseable until it changes, and mtime/size are already in the cache key.
    """
    ddi_path = _write_ddi(tmp_path, make_ddi_xml, [("AGE", "Age", 2)])
    real_read = readers.read_ipums_ddi
    calls = {"n": 0}

    def failing_once(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("not a codebook")
        return real_read(path)

    monkeypatch.setattr("src.extractors.ipums_ddi.readers.read_ipums_ddi", failing_once)

    assert try_summarize_ddi(ddi_path) is None
    assert try_summarize_ddi(ddi_path) is None
    assert calls["n"] == 1


def test_summarize_ddi_raises_for_stub_codebook(tmp_path: Path) -> None:
    stub = tmp_path / "stub.xml"
    stub.write_text("<codeBook/>")

    with pytest.raises(Exception):
        summarize_ddi(stub)


# --- deriving column lists --------------------------------------------------


def test_flag_columns_for_includes_shared_flag_when_one_source_requested(
    tmp_path: Path, make_ddi_xml, cps_flag_vars
) -> None:
    summary = summarize_ddi(_write_ddi(tmp_path, make_ddi_xml, cps_flag_vars))

    assert flag_columns_for(summary, ["WKSWORK1"]) == ("QWKSWORK", "QWKSWORKTEST")


def test_flag_columns_for_excludes_topcode_when_disabled(
    tmp_path: Path, make_ddi_xml, cps_flag_vars
) -> None:
    summary = summarize_ddi(_write_ddi(tmp_path, make_ddi_xml, cps_flag_vars))

    with_topcode = flag_columns_for(summary, ["INCLONGJ"], include_topcode=True)
    without = flag_columns_for(summary, ["INCLONGJ"], include_topcode=False)

    assert with_topcode == ("QINCLONG", "QINCLONGD", "TINCLONGJ")
    assert without == ("QINCLONG", "QINCLONGD")


def test_merge_column_names_adds_flags_and_drops_technical_columns(
    tmp_path: Path, make_ddi_xml
) -> None:
    variables = [
        ("YEAR", "Survey year", 4),
        ("ASECWT", "ASEC weight", 5),
        ("INCWAGE", "Wage income", 7),
        ("QINCWAGE", "Data quality flag for INCWAGE", 1),
    ]
    summary = summarize_ddi(_write_ddi(tmp_path, make_ddi_xml, variables))

    columns = merge_column_names(summary, ["INCWAGE"])

    assert columns == ["INCWAGE", "QINCWAGE"]
    # The preselected technical/weight columns must stay out: bronze already
    # has them and merge_variables_into_bronze drops them for that reason.
    assert "YEAR" not in columns
    assert "ASECWT" not in columns


def test_merge_column_names_falls_back_to_requested_when_summary_is_none() -> None:
    assert merge_column_names(None, ["AGE", "SEX"]) == ["AGE", "SEX"]


def test_merge_column_names_does_not_duplicate_a_requested_flag(
    tmp_path: Path, make_ddi_xml
) -> None:
    variables = [
        ("INCWAGE", "Wage income", 7),
        ("QINCWAGE", "Data quality flag for INCWAGE", 1),
    ]
    summary = summarize_ddi(_write_ddi(tmp_path, make_ddi_xml, variables))

    assert merge_column_names(summary, ["INCWAGE", "QINCWAGE"]) == [
        "INCWAGE",
        "QINCWAGE",
    ]


# --- rebuilding a summary from a manifest entry ----------------------------


def test_summary_from_metadata_round_trips_recorded_keys() -> None:
    summary = summary_from_metadata(
        {
            "ddi_path": "/tmp/cps_00001.xml",
            "flag_parser_version": FLAG_PARSER_VERSION,
            "delivered_variables": ["INCWAGE", "QINCWAGE"],
            "quality_flags": {"INCWAGE": ["QINCWAGE"]},
            "topcode_flags": {},
        }
    )

    assert summary is not None
    assert summary.variables == ("INCWAGE", "QINCWAGE")
    assert summary.quality_flags == {"INCWAGE": ("QINCWAGE",)}


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("QINCWAGE", id="scalar_string"),
        pytest.param(None, id="null"),
        pytest.param({"nested": "map"}, id="mapping"),
        pytest.param(["QINCWAGE", 7], id="list_with_non_string"),
    ],
)
def test_summary_from_metadata_rejects_a_flag_map_value_that_is_not_a_name_list(
    bad_value: object,
) -> None:
    # tuple() on a scalar coerces rather than rejects: "QINCWAGE" would become
    # ('Q','I','N','C',...), which silently drops the real flag column out of
    # the registry and out of merge_column_names.
    summary = summary_from_metadata(
        {
            "delivered_variables": ["INCWAGE", "QINCWAGE"],
            "flag_parser_version": FLAG_PARSER_VERSION,
            "ddi_path": "/tmp/x.xml",
            "quality_flags": {"INCWAGE": bad_value},
        }
    )

    assert summary is not None
    assert dict(summary.quality_flags) == {}
    assert summary.flag_names == frozenset()
    assert summary.kind_of("Q") is None


def test_summary_from_metadata_keeps_the_well_formed_pairs_of_a_partial_map() -> None:
    summary = summary_from_metadata(
        {
            "delivered_variables": ["INCWAGE", "QINCWAGE", "INCFARM"],
            "flag_parser_version": FLAG_PARSER_VERSION,
            "ddi_path": "/tmp/x.xml",
            "quality_flags": {"INCWAGE": ["QINCWAGE"], "INCFARM": "QINCFARM"},
        }
    )

    assert summary is not None
    assert dict(summary.quality_flags) == {"INCWAGE": ("QINCWAGE",)}


def test_summary_from_metadata_is_none_for_legacy_entry() -> None:
    assert summary_from_metadata({"variables": ["AGE"]}) is None


@pytest.mark.parametrize(
    "stamp", [{}, {"flag_parser_version": FLAG_PARSER_VERSION - 1}]
)
def test_summary_from_metadata_is_none_for_stale_parser_version(stamp: dict) -> None:
    """An unstamped entry predates the constant; a lower stamp predates the
    current regexes. Both must be re-derived from the codebook, not trusted.
    """
    assert (
        summary_from_metadata(
            {
                "ddi_path": "/tmp/cps_00001.xml",
                "delivered_variables": ["INCWAGE", "QINCWAGE"],
                "quality_flags": {"INCWAGE": ["QINCWAGE"]},
                "topcode_flags": {},
                **stamp,
            }
        )
        is None
    )


# --- the collection registry -----------------------------------------------


def _seed_manifest(
    collection_dir: Path, ddi_path: Path, extraction_id: str, metadata_extra: dict
) -> None:
    data_path = collection_dir / f"{extraction_id}.dat.gz"
    data_path.write_bytes(b"data")
    record = build_extraction_record(
        source="ipums_api",
        extraction_id=extraction_id,
        file_path=data_path,
        metadata={
            "collection": "cps",
            "samples": ("cps2006_09s",),
            "variables": ("INCWAGE",),
            "ddi_path": str(ddi_path),
            "extract_id": 1,
            # what extract() writes today; metadata_extra can override it to
            # stand in for an entry written by an older parser
            "flag_parser_version": FLAG_PARSER_VERSION,
            **metadata_extra,
        },
    )
    append_to_manifest(collection_dir, record)


def test_collection_flag_registry_reads_manifest_without_parsing_ddis(
    tmp_path: Path, make_ddi_xml, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    ddi_path = _write_ddi(
        collection_dir,
        make_ddi_xml,
        [
            ("INCWAGE", "Wage income", 7),
            ("QINCWAGE", "Data quality flag for INCWAGE", 1),
        ],
    )
    _seed_manifest(
        collection_dir,
        ddi_path,
        "cps_00001",
        {
            "delivered_variables": ["INCWAGE", "QINCWAGE"],
            "quality_flags": {"INCWAGE": ["QINCWAGE"]},
            "topcode_flags": {},
        },
    )

    def must_not_parse(path):
        raise AssertionError("registry should not parse a DDI it already has recorded")

    monkeypatch.setattr(
        "src.extractors.ipums_ddi.readers.read_ipums_ddi", must_not_parse
    )

    registry = collection_flag_registry(collection_dir)

    assert registry.kind_of("QINCWAGE") == "quality"
    assert registry.sources_of("QINCWAGE") == ("INCWAGE",)


def test_collection_flag_registry_rederives_when_parser_version_is_stale(
    tmp_path: Path, make_ddi_xml
) -> None:
    """The recorded map is a cached parse, not a fact. An entry stamped by an
    older parser is ignored in favour of the codebook sitting next to it, so a
    fix to parse_flag_label reaches extracts downloaded before it.
    """
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    ddi_path = _write_ddi(
        collection_dir,
        make_ddi_xml,
        [
            ("INCWAGE", "Wage income", 7),
            ("QINCWAGE", "Data quality flag for INCWAGE", 1),
        ],
    )
    _seed_manifest(
        collection_dir,
        ddi_path,
        "cps_00001",
        {
            "flag_parser_version": FLAG_PARSER_VERSION - 1,
            "delivered_variables": ["INCWAGE", "QINCWAGE"],
            # what a buggy older parser might have concluded
            "quality_flags": {},
            "topcode_flags": {"INCWAGE": ["QINCWAGE"]},
        },
    )

    registry = collection_flag_registry(collection_dir)

    assert registry.kind_of("QINCWAGE") == "quality"
    assert registry.sources_of("QINCWAGE") == ("INCWAGE",)


def test_collection_flag_registry_uses_a_stale_map_when_the_codebook_is_gone(
    tmp_path: Path, make_ddi_xml
) -> None:
    """Last tier of resolution: the stamp says the map is old, but the codebook
    it would be re-derived from no longer exists. A possibly-outdated answer
    beats "not a flag", which costs a rejected API round trip - and the entry
    beside it must not mask the loss either, so this directory has two.
    """
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    good_ddi = _write_ddi(
        collection_dir, make_ddi_xml, [("AGE", "Age", 2)], name="cps_00001"
    )
    _seed_manifest(
        collection_dir,
        good_ddi,
        "cps_00001",
        {"delivered_variables": ["AGE"], "quality_flags": {}, "topcode_flags": {}},
    )
    gone_ddi = _write_ddi(
        collection_dir, make_ddi_xml, [("INCWAGE", "Wage income", 7)], name="cps_00002"
    )
    _seed_manifest(
        collection_dir,
        gone_ddi,
        "cps_00002",
        {
            "flag_parser_version": FLAG_PARSER_VERSION - 1,
            "delivered_variables": ["INCWAGE", "QINCWAGE"],
            "quality_flags": {"INCWAGE": ["QINCWAGE"]},
            "topcode_flags": {},
        },
    )
    gone_ddi.unlink()

    with structlog.testing.capture_logs() as logs:
        registry = collection_flag_registry(collection_dir)

    assert registry.kind_of("QINCWAGE") == "quality"
    assert [entry["event"] for entry in logs if "stale" in entry["event"]] == [
        "ipums_flag_map_stale_but_used"
    ]


def test_collection_flag_registry_uses_a_stale_map_when_the_codebook_is_unreadable(
    tmp_path: Path,
) -> None:
    """Broken codebook, stale stamp, and a map that would be correct if the codebook
    were readable.

    Test case:
    - cps_00001.xml is a broken cannot be parsed;
    - cps_00001 is stamped with an old parser version
    - cps_00001 has a map that would be correct if the codebook were readable;
    Expected behavior:
        The registry must use the map and log that it is stale, rather than
        returning "not a flag" for QINCWAGE.
    """
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    broken_ddi = collection_dir / "cps_00001.xml"
    broken_ddi.write_text("<codeBook/>")  # exists, but unreadable - same stub
    # as test_try_summarize_ddi_returns_none_for_stub_codebook
    _seed_manifest(
        collection_dir,
        broken_ddi,
        "cps_00001",
        {
            "flag_parser_version": FLAG_PARSER_VERSION - 1,
            "delivered_variables": ["INCWAGE", "QINCWAGE"],
            "quality_flags": {"INCWAGE": ["QINCWAGE"]},
            "topcode_flags": {},
        },
    )

    with structlog.testing.capture_logs() as logs:
        registry = collection_flag_registry(collection_dir)

    assert registry.kind_of("QINCWAGE") == "quality"
    assert [entry["event"] for entry in logs if "stale" in entry["event"]] == [
        "ipums_flag_map_stale_but_used"
    ]


def test_summary_from_metadata_can_opt_out_of_the_version_check() -> None:
    metadata = {
        "ddi_path": "/tmp/cps_00001.xml",
        "flag_parser_version": FLAG_PARSER_VERSION - 1,
        "delivered_variables": ["INCWAGE", "QINCWAGE"],
        "quality_flags": {"INCWAGE": ["QINCWAGE"]},
        "topcode_flags": {},
    }

    assert summary_from_metadata(metadata) is None
    stale = summary_from_metadata(metadata, require_current_flag_parser=False)
    assert stale is not None
    assert stale.quality_flags == {"INCWAGE": ("QINCWAGE",)}


def test_collection_flag_registry_backfills_legacy_entry_from_ddi(
    tmp_path: Path, make_ddi_xml
) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    ddi_path = _write_ddi(
        collection_dir,
        make_ddi_xml,
        [
            ("INCWAGE", "Wage income", 7),
            ("QINCWAGE", "Data quality flag for INCWAGE", 1),
        ],
    )
    # A manifest entry from before delivered_variables was recorded.
    _seed_manifest(collection_dir, ddi_path, "cps_00001", {})

    registry = collection_flag_registry(collection_dir)

    assert registry.kind_of("QINCWAGE") == "quality"


def test_collection_flag_registry_falls_back_to_loose_codebooks(
    tmp_path: Path, make_ddi_xml
) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _write_ddi(
        collection_dir,
        make_ddi_xml,
        [("INCFARM", "Farm income", 7), ("TINCFARM", "Topcode Flag for INCFARM", 1)],
    )

    registry = collection_flag_registry(collection_dir)

    assert registry.kind_of("TINCFARM") == "topcode"


def test_collection_flag_registry_is_empty_for_unknown_collection(
    tmp_path: Path,
) -> None:
    registry = collection_flag_registry(tmp_path / "nothing-here")

    assert not registry
    assert registry.kind_of("QINCWAGE") is None


def test_collection_flag_registry_falls_back_when_recorded_ddi_is_unreadable(
    tmp_path: Path, make_ddi_xml
) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    broken = collection_dir / "cps_00001.xml"
    broken.write_text("<codeBook/>")  # unreadable
    _seed_manifest(collection_dir, broken, "cps_00001", {})
    _write_ddi(
        collection_dir,
        make_ddi_xml,  # a good, loose codebook
        [("INCFARM", "Farm income", 7), ("TINCFARM", "Topcode Flag for INCFARM", 1)],
    )

    assert collection_flag_registry(collection_dir).kind_of("TINCFARM") == "topcode"


def test_collection_flag_registry_accepts_preread_entries(
    tmp_path: Path, make_ddi_xml
) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    ddi_path = _write_ddi(
        collection_dir,
        make_ddi_xml,
        [
            ("INCWAGE", "Wage income", 7),
            ("QINCWAGE", "Data quality flag for INCWAGE", 1),
        ],
    )
    _seed_manifest(
        collection_dir,
        ddi_path,
        "cps_00001",
        {
            "delivered_variables": ["INCWAGE", "QINCWAGE"],
            "quality_flags": {"INCWAGE": ["QINCWAGE"]},
            "topcode_flags": {},
        },
    )
    entries = read_manifest(collection_dir)

    registry = collection_flag_registry(collection_dir, entries)

    assert registry.kind_of("QINCWAGE") == "quality"
    assert registry.sources_of("QINCWAGE") == ("INCWAGE",)


@pytest.mark.parametrize(
    "manifest_text",
    [
        "- extraction_id: cps_00001\n  metadata:\n",  # empty
        "- extraction_id: cps_00001\n  metadata: not-a-mapping\n",  # scalar
        "- just-a-string\n",  # non-dict entry
    ],
)
def test_collection_flag_registry_survives_a_malformed_entry(
    tmp_path: Path, make_ddi_xml, manifest_text: str
) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    (collection_dir / "_MANIFEST.yaml").write_text(manifest_text)
    _write_ddi(
        collection_dir,
        make_ddi_xml,
        [
            ("INCWAGE", "Wage income", 7),
            ("QINCWAGE", "Data quality flag for INCWAGE", 1),
        ],
    )
    with structlog.testing.capture_logs() as logs:
        registry = collection_flag_registry(collection_dir)

    # the bad entry is skipped and the loose codebook still answers
    assert registry.kind_of("QINCWAGE") == "quality"
    # ...and it is skipped loudly. Warning for the empty and scalar shapes as
    # well as the non-dict entry is what keeps a corrupted manifest from
    # looking like an empty one.
    skipped = [entry for entry in logs if entry["event"] == "manifest_entry_skipped"]
    assert [entry["reason"] for entry in skipped] == ["metadata_not_a_mapping"]
