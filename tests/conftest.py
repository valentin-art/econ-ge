"""Shared test fixtures.

The IPUMS-facing ones build real DDI codebooks rather than stubs: ipumspy's
Codebook.read requires the full stdyDscr/fileDscr skeleton and raises on
anything less, and the extractor now reads codebooks to learn which columns an
extract really delivered. A test that fakes that away would not exercise the
behaviour it claims to.
"""

import gzip
from collections.abc import Callable, Sequence

import pytest

from src.extractors.ipums_ddi import clear_ddi_summary_cache

_DDI_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<codeBook ID="ddi2-test" version="2.5" xmlns="ddi:codebook:2_5">
  <stdyDscr>
    <citation>
      <serStmt>
        <serName abbr="cps">IPUMS CPS</serName>
        <serInfo>DOI:10.18128/D030.V13.0</serInfo>
      </serStmt>
    </citation>
    <stdyInfo/>
    <dataAccs>
      <useStmt>
        <citReq>Cite IPUMS appropriately.</citReq>
        <conditions>Use it for GOOD -- never for EVIL.</conditions>
      </useStmt>
    </dataAccs>
  </stdyDscr>
  <fileDscr ID="ExtractData">
    <fileTxt>
      <fileName>{filename}</fileName>
      <fileCont>Microdata records</fileCont>
      <fileStrc type="rectangular"/>
      <fileType charset="ISO-8859-1">ISO-8859-1 data file</fileType>
      <format>fixed length fields</format>
      <filePlac>IPUMS</filePlac>
    </fileTxt>
  </fileDscr>
  <dataDscr>
{vars}
  </dataDscr>
</codeBook>
"""

_VAR_TEMPLATE = """    <var ID="{name}" dcml="0" files="ExtractData" intrvl="discrete" name="{name}">
      <location EndPos="{end}" StartPos="{start}" width="{width}"/>
      <labl>{label}</labl>
      <txt>{label}.</txt>
      <concept vocab="IPUMS">{concept}</concept>
      <varFormat schema="other" type="numeric"/>
    </var>"""

# (name, label, width). Modelled on the real labels in cps_00030/33/35, which
# is what makes the flag-detection tests meaningful:
#   - QACTNLFL is truncated, QUHRSWORKLY is not - names are not derivable;
#   - QWKSWORK is one flag shared by two source variables;
#   - INCLONGJ has both a general and a detailed quality flag, plus a topcode
#     flag that arrives regardless of data_quality_flags.
# The last four are decoys that must never be read as flags: ASECFLAG and
# HFLAG have flag-ish labels, and TRANWORK/TRANTIME are ordinary IPUMS USA
# variables whose names begin with T. They pin the guarantee that detection
# reads the label and never the name.
CPS_FLAG_VARS: list[tuple[str, str, int]] = [
    ("ACTNLFLY", "Activities not in labor force last year", 1),
    ("QACTNLFL", "Data quality flag for ACTNLFLY", 1),
    ("INCLONGJ", "Earnings from longest job", 7),
    ("QINCLONG", "Data quality flag for INCLONGJ [general version]", 1),
    ("QINCLONGD", "Data quality flag for INCLONGJ [detailed version]", 2),
    ("TINCLONGJ", "Topcode Flag for INCLONGJ", 1),
    ("WKSWORK1", "Weeks worked last year", 2),
    ("WKSWORK2", "Weeks worked last year, intervalled", 1),
    ("QWKSWORK", "Data quality flag for WKSWORK1 and WKSWORK2", 1),
    ("UHRSWORKLY", "Usual hours worked per week last year", 2),
    ("QUHRSWORKLY", "Data quality flag for UHRSWORKLY", 1),
    ("ASECFLAG", "Flag for ASEC", 1),
    ("HFLAG", "Flag for the 3/8 file 2014", 1),
    ("TRANWORK", "Means of transportation to work", 2),
    ("TRANTIME", "Travel time to work", 3),
]


@pytest.fixture
def cps_flag_vars() -> list[tuple[str, str, int]]:
    """CPS_FLAG_VARS as a fixture, for tests that want the whole set."""
    return list(CPS_FLAG_VARS)


@pytest.fixture(autouse=True)
def _clear_ddi_cache():
    """Summaries are cached on (path, mtime, size); a test that rewrites a
    codebook to the same path within one mtime tick would otherwise see the
    previous test's answer.
    """
    clear_ddi_summary_cache()
    yield
    clear_ddi_summary_cache()


@pytest.fixture
def make_ddi_xml() -> Callable[..., str]:
    """Build a valid DDI codebook from (name, label, width) triples.

    Column positions are accumulated from the widths, so a caller never has to
    keep StartPos/EndPos consistent by hand.
    """

    def _make(
        variables: Sequence[tuple[str, str, int]],
        filename: str = "test.dat",
    ) -> str:
        rendered = []
        position = 1
        for name, label, width in variables:
            rendered.append(
                _VAR_TEMPLATE.format(
                    name=name,
                    label=label,
                    start=position,
                    end=position + width - 1,
                    width=width,
                    concept=(
                        "Data Quality Flags Variables -- PERSON"
                        if "flag for" in label.lower()
                        else "Technical Variables"
                    ),
                )
            )
            position += width
        return _DDI_TEMPLATE.format(filename=filename, vars="\n".join(rendered))

    return _make


@pytest.fixture
def make_fixed_width_dat() -> Callable[..., bytes]:
    """Gzip a fixed-width data file from per-row values, right-aligning each
    value in its variable's width so it lines up with a make_ddi_xml codebook.
    """

    def _make(
        rows: Sequence[Sequence[object]],
        variables: Sequence[tuple[str, str, int]],
    ) -> bytes:
        lines = []
        for row in rows:
            lines.append(
                "".join(
                    str(value).rjust(width, "0")
                    for value, (_, _, width) in zip(row, variables, strict=True)
                )
            )
        text = "\n".join(lines) + "\n"
        return gzip.compress(text.encode("iso-8859-1"))

    return _make
