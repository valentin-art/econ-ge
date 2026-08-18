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

from src.extractors.ipums_ddi import parse_flag_label, summarize_ddi, try_summarize_ddi


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
        "WKSWORK1": ("QWKSWORK",),
        "WKSWORK2": ("QWKSWORK",),
        "UHRSWORKLY": ("QUHRSWORKLY",),
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
