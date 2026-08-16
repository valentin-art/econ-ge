"""Tests for extractors.ipums_ddi.

The load-bearing property here is that flags are recognised by their DDI label
and never by their name: IPUMS flag names are irregular (QACTNLFL truncated,
QUHRSWORKLY not; QWKSWORK shared by two variables) and ordinary variables can
begin with Q or T, so any name-shaped shortcut would be wrong in both
directions.
"""

import pytest

from src.extractors.ipums_ddi import parse_flag_label

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
