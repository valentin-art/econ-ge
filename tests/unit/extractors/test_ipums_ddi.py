"""Tests for extractors.ipums_ddi.

The load-bearing property here is that flags are recognised by their DDI label
and never by their name: IPUMS flag names are irregular (QACTNLFL truncated,
QUHRSWORKLY not; QWKSWORK shared by two variables) and ordinary variables can
begin with Q or T, so any name-shaped shortcut would be wrong in both
directions.
"""

from pathlib import Path

import pytest
from ipumspy import readers

from src.extractors.base import build_extraction_record
from src.extractors.ipums_ddi import (
    FLAG_PARSER_VERSION,
    collection_flag_registry,
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


def test_summarize_ddi_raises_for_stub_codebook(tmp_path: Path) -> None:
    stub = tmp_path / "stub.xml"
    stub.write_text("<codeBook/>")

    with pytest.raises(Exception):
        summarize_ddi(stub)


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
