import gzip
import warnings
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import structlog.testing
from ipumspy import readers

from src.parsers.ipums import (
    bronze_columns_by_year,
    bronze_coverage,
    bronze_path,
    build_and_save_variable_dictionary,
    build_variable_dictionary,
    load_variable_dictionary,
    merge_variables_into_bronze,
    parse_ipums_extract,
    parse_to_bronze,
    save_variable_dictionary,
    variable_dictionary_path,
)

# Minimal but structurally valid IPUMS DDI 2.5 codebook (trimmed version of a
# real IPUMS CPS extract's codebook - same elements/namespace as the ones
# ipumspy actually downloads, with only 3 variables): YEAR (numeric, no value
# labels), MONTH and SEX (numeric with catgry value-label blocks).
_DDI_XML = """<?xml version="1.0" encoding="UTF-8"?>
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
      <fileName>cps_00027.dat</fileName>
      <fileCont>Microdata records</fileCont>
      <fileStrc type="rectangular"/>
      <fileType charset="ISO-8859-1">ISO-8859-1 data file</fileType>
      <format>fixed length fields</format>
      <filePlac>IPUMS</filePlac>
    </fileTxt>
  </fileDscr>
  <dataDscr>
    <var ID="YEAR" dcml="0" files="ExtractData" intrvl="contin" name="YEAR">
      <location EndPos="4" StartPos="1" width="4"/>
      <labl>Survey year</labl>
      <txt>YEAR reports the year in which the survey was conducted.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
    <var ID="MONTH" dcml="0" files="ExtractData" intrvl="discrete" name="MONTH">
      <location EndPos="6" StartPos="5" width="2"/>
      <labl>Month</labl>
      <txt>MONTH indicates the calendar month of the interview.</txt>
      <catgry>
        <catValu>01</catValu>
        <labl>January</labl>
      </catgry>
      <catgry>
        <catValu>02</catValu>
        <labl>February</labl>
      </catgry>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
    <var ID="SEX" dcml="0" files="ExtractData" intrvl="discrete" name="SEX">
      <location EndPos="7" StartPos="7" width="1"/>
      <labl>Sex</labl>
      <txt>SEX reports whether the person was male or female.</txt>
      <catgry>
        <catValu>1</catValu>
        <labl>Male</labl>
      </catgry>
      <catgry>
        <catValu>2</catValu>
        <labl>Female</labl>
      </catgry>
      <concept vocab="IPUMS">Demographic Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
  </dataDscr>
</codeBook>
"""

# YEAR(4) + MONTH(2) + SEX(1), matching the <location> specs above.
_DAT_TEXT = "2006011\n2006022\n"


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    ddi_path = tmp_path / "cps_00027.xml"
    ddi_path.write_text(_DDI_XML, encoding="utf-8")
    data_path = tmp_path / "cps_00027.dat.gz"
    data_path.write_bytes(gzip.compress(_DAT_TEXT.encode("iso-8859-1")))
    return data_path, ddi_path


def test_build_variable_dictionary_matches_cps_shape(tmp_path: Path) -> None:
    _, ddi_path = _write_fixture(tmp_path)
    ddi_codebook = readers.read_ipums_ddi(ddi_path)

    variable_dictionary = build_variable_dictionary(ddi_codebook)

    assert variable_dictionary["YEAR"]["numeric"] is True
    assert variable_dictionary["YEAR"]["Description"] == "Survey year"
    assert variable_dictionary["YEAR"]["Values"] == {}
    assert variable_dictionary["MONTH"]["Values"] == {"1": "January", "2": "February"}
    assert variable_dictionary["SEX"]["Values"] == {"1": "Male", "2": "Female"}


def test_save_and_load_variable_dictionary_roundtrip(tmp_path: Path) -> None:
    _, ddi_path = _write_fixture(tmp_path)
    ddi_codebook = readers.read_ipums_ddi(ddi_path)
    variable_dictionary = build_variable_dictionary(ddi_codebook)

    out_path = save_variable_dictionary(variable_dictionary, tmp_path, 2006)

    assert out_path == variable_dictionary_path(tmp_path, 2006)
    assert load_variable_dictionary(tmp_path, 2006) == variable_dictionary


def test_save_variable_dictionary_unions_with_existing_year(tmp_path: Path) -> None:
    save_variable_dictionary({"AGE": {"Description": "Age"}}, tmp_path, 2006)

    save_variable_dictionary({"RACE": {"Description": "Race"}}, tmp_path, 2006)

    merged = load_variable_dictionary(tmp_path, 2006)
    assert set(merged) == {"AGE", "RACE"}


def test_save_variable_dictionary_keeps_existing_on_key_collision(
    tmp_path: Path,
) -> None:
    save_variable_dictionary({"AGE": {"Description": "Age (old)"}}, tmp_path, 2006)

    save_variable_dictionary({"AGE": {"Description": "Age (new)"}}, tmp_path, 2006)

    merged = load_variable_dictionary(tmp_path, 2006)
    assert merged["AGE"]["Description"] == "Age (old)"


def test_save_variable_dictionary_logs_warning_on_drift(tmp_path: Path) -> None:
    save_variable_dictionary({"AGE": {"Description": "Age (old)"}}, tmp_path, 2006)

    with structlog.testing.capture_logs() as drifted_logs:
        save_variable_dictionary({"AGE": {"Description": "Age (new)"}}, tmp_path, 2006)
    assert [log["event"] for log in drifted_logs] == ["ipums_variable_definition_drift"]

    with structlog.testing.capture_logs() as identical_logs:
        # Identical re-save of the same entry - not drift, no warning.
        save_variable_dictionary({"AGE": {"Description": "Age (old)"}}, tmp_path, 2006)
    assert identical_logs == []


def test_save_variable_dictionary_force_overwrites_on_key_collision(
    tmp_path: Path,
) -> None:
    save_variable_dictionary({"AGE": {"Description": "Age (old)"}}, tmp_path, 2006)

    with structlog.testing.capture_logs() as logs:
        save_variable_dictionary(
            {"AGE": {"Description": "Age (new)"}}, tmp_path, 2006, force=True
        )

    merged = load_variable_dictionary(tmp_path, 2006)
    assert merged["AGE"]["Description"] == "Age (new)"
    # Still logged - force overwrites the definition, it doesn't hide that
    # a drift happened.
    assert [log["event"] for log in logs] == ["ipums_variable_definition_drift"]
    assert logs[0]["overwritten"] is True


def test_build_and_save_variable_dictionary(tmp_path: Path) -> None:
    _, ddi_path = _write_fixture(tmp_path)

    out_paths = build_and_save_variable_dictionary(
        ddi_path, tmp_path, years=[2005, 2006]
    )

    assert [p.exists() for p in out_paths] == [True, True]
    assert "MONTH" in load_variable_dictionary(tmp_path, 2005)
    assert "MONTH" in load_variable_dictionary(tmp_path, 2006)


def test_parse_ipums_extract_returns_tidy_dataframe(tmp_path: Path) -> None:
    data_path, ddi_path = _write_fixture(tmp_path)

    df = parse_ipums_extract(data_path, ddi_path)

    assert list(df.columns) == ["YEAR", "MONTH", "SEX"]
    assert len(df) == 2
    assert df["MONTH"].tolist() == [1, 2]
    assert df["SEX"].tolist() == [1, 2]


def test_parse_to_bronze_writes_parquet(tmp_path: Path) -> None:
    data_path, ddi_path = _write_fixture(tmp_path)
    bronze_dir = tmp_path / "bronze"

    out_paths = parse_to_bronze(data_path, ddi_path, "cps", bronze_dir)

    assert out_paths == [bronze_path(bronze_dir, "cps", 2006)]
    df = pd.read_parquet(out_paths[0])
    assert len(df) == 2
    assert df["YEAR"].tolist() == [2006, 2006]


# 6 rows across 3 years, ordered so chunksize=2 puts each year's 2 rows in
# two different, non-adjacent chunks:
#   chunk 1 (rows 0-1): YEAR 2005, 2006
#   chunk 2 (rows 2-3): YEAR 2005, 2007
#   chunk 3 (rows 4-5): YEAR 2006, 2007
_MULTI_YEAR_DAT_TEXT = "2005011\n2006012\n2005021\n2007011\n2006022\n2007022\n"


def _write_multi_year_fixture(tmp_path: Path) -> tuple[Path, Path]:
    ddi_path = tmp_path / "cps_00027.xml"
    ddi_path.write_text(_DDI_XML, encoding="utf-8")
    data_path = tmp_path / "cps_00027.dat.gz"
    data_path.write_bytes(gzip.compress(_MULTI_YEAR_DAT_TEXT.encode("iso-8859-1")))
    return data_path, ddi_path


def test_parse_to_bronze_splits_by_year_across_chunk_boundaries(
    tmp_path: Path,
) -> None:
    data_path, ddi_path = _write_multi_year_fixture(tmp_path)
    bronze_dir = tmp_path / "bronze"

    out_paths = parse_to_bronze(data_path, ddi_path, "cps", bronze_dir, chunksize=2)

    expected_paths = [
        bronze_path(bronze_dir, "cps", year) for year in (2005, 2006, 2007)
    ]
    assert out_paths == expected_paths

    total_rows = 0
    for year, path in zip((2005, 2006, 2007), out_paths):
        df = pd.read_parquet(path)
        assert len(df) == 2, f"year {year} should have accumulated 2 rows"
        assert df["YEAR"].unique().tolist() == [year]
        total_rows += len(df)
    assert total_rows == 6


def test_parse_to_bronze_raises_on_empty_extract(tmp_path: Path) -> None:
    ddi_path = tmp_path / "cps_00027.xml"
    ddi_path.write_text(_DDI_XML, encoding="utf-8")
    data_path = tmp_path / "cps_00027.dat.gz"
    data_path.write_bytes(gzip.compress(b""))
    bronze_dir = tmp_path / "bronze"

    with pytest.raises(ValueError, match="no rows"):
        parse_to_bronze(data_path, ddi_path, "cps", bronze_dir)


def test_parse_to_bronze_does_not_clobber_existing_year_on_mid_stream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path, ddi_path = _write_multi_year_fixture(tmp_path)  # 2005/2006/2007
    bronze_dir = tmp_path / "bronze"
    good_path = bronze_path(bronze_dir, "cps", 2005)
    good_path.parent.mkdir(parents=True, exist_ok=True)
    good_path.write_bytes(b"not-really-parquet-but-must-survive")
    before = good_path.read_bytes()

    calls = {"n": 0}
    import src.parsers.ipums as ipums_mod

    orig = ipums_mod.check_no_duplicate_columns

    def flaky(df):
        calls["n"] += 1
        orig(df)
        if calls["n"] == 2:
            raise ValueError("boom")

    monkeypatch.setattr(ipums_mod, "check_no_duplicate_columns", flaky)
    with pytest.raises(ValueError, match="boom"):
        # replace=True so the run gets far enough to fail mid-stream: even
        # with overwriting permitted, a year is only ever replaced by a
        # complete rewrite.
        parse_to_bronze(
            data_path, ddi_path, "cps", bronze_dir, chunksize=2, replace=True
        )

    assert good_path.read_bytes() == before
    assert list((bronze_dir / "cps").glob("*.tmp.parquet")) == []


def test_merge_variables_into_bronze_raises_when_no_merge_columns_present(
    tmp_path: Path,
) -> None:
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(_EXISTING_DAT_TEXT.encode("iso-8859-1"))
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)

    delta_ddi_path = tmp_path / "delta.xml"
    delta_ddi_path.write_text(_DELTA_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "delta.dat.gz"
    delta_data_path.write_bytes(gzip.compress(_DELTA_DAT_TEXT.encode("iso-8859-1")))

    with pytest.raises(RuntimeError, match="no columns in common"):
        merge_variables_into_bronze(
            delta_data_path,
            delta_ddi_path,
            "cps",
            bronze_dir,
            new_variables=["RACE"],
            merge_keys=("SERIAL", "PERNUM"),  # absent from both fixtures
        )


# Same DDI as _DELTA_DDI_XML; both rows share MONTH=01 so they collide on
# the effective merge key (YEAR, MONTH) once SERIAL/PERNUM are absent.
_DUPLICATE_KEY_DELTA_DAT_TEXT = "2006011000000001100\n2006011000000002200\n"


def test_merge_variables_into_bronze_raises_on_row_count_mismatch(
    tmp_path: Path,
) -> None:
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(_EXISTING_DAT_TEXT.encode("iso-8859-1"))
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)

    delta_ddi_path = tmp_path / "delta.xml"
    delta_ddi_path.write_text(_DELTA_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "delta.dat.gz"
    delta_data_path.write_bytes(
        gzip.compress(_DUPLICATE_KEY_DELTA_DAT_TEXT.encode("iso-8859-1"))
    )

    with pytest.raises(RuntimeError, match="changed the number of rows"):
        merge_variables_into_bronze(
            delta_data_path, delta_ddi_path, "cps", bronze_dir, new_variables=["RACE"]
        )


# DDI + data for the "already in bronze" extract: YEAR, MONTH, CPSIDP (merge
# keys) plus SEX (an already-covered variable, mirrors what a prior
# "new_samples" parse_to_bronze run would have produced).
_EXISTING_DDI_XML = """<?xml version="1.0" encoding="UTF-8"?>
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
      <fileName>existing.dat</fileName>
      <fileCont>Microdata records</fileCont>
      <fileStrc type="rectangular"/>
      <fileType charset="ISO-8859-1">ISO-8859-1 data file</fileType>
      <format>fixed length fields</format>
      <filePlac>IPUMS</filePlac>
    </fileTxt>
  </fileDscr>
  <dataDscr>
    <var ID="YEAR" dcml="0" files="ExtractData" intrvl="contin" name="YEAR">
      <location EndPos="4" StartPos="1" width="4"/>
      <labl>Survey year</labl>
      <txt>Survey year.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
    <var ID="MONTH" dcml="0" files="ExtractData" intrvl="discrete" name="MONTH">
      <location EndPos="6" StartPos="5" width="2"/>
      <labl>Month</labl>
      <txt>Month.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
    <var ID="CPSIDP" dcml="0" files="ExtractData" intrvl="contin" name="CPSIDP">
      <location EndPos="16" StartPos="7" width="10"/>
      <labl>Person id</labl>
      <txt>Uniquely identifies a person across CPS samples.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
    <var ID="SEX" dcml="0" files="ExtractData" intrvl="discrete" name="SEX">
      <location EndPos="17" StartPos="17" width="1"/>
      <labl>Sex</labl>
      <txt>Sex.</txt>
      <concept vocab="IPUMS">Demographic Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
  </dataDscr>
</codeBook>
"""
# YEAR(4) + MONTH(2) + CPSIDP(10) + SEX(1)
_EXISTING_DAT_TEXT = "20060110000000011\n20060210000000022\n"

# DDI + data for the variable-delta extract: same merge keys, RACE only -
# what extract_incremental would submit as a "variable_delta" for samples
# already covered on other variables.
_DELTA_DDI_XML = """<?xml version="1.0" encoding="UTF-8"?>
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
      <fileName>delta.dat</fileName>
      <fileCont>Microdata records</fileCont>
      <fileStrc type="rectangular"/>
      <fileType charset="ISO-8859-1">ISO-8859-1 data file</fileType>
      <format>fixed length fields</format>
      <filePlac>IPUMS</filePlac>
    </fileTxt>
  </fileDscr>
  <dataDscr>
    <var ID="YEAR" dcml="0" files="ExtractData" intrvl="contin" name="YEAR">
      <location EndPos="4" StartPos="1" width="4"/>
      <labl>Survey year</labl>
      <txt>Survey year.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
    <var ID="MONTH" dcml="0" files="ExtractData" intrvl="discrete" name="MONTH">
      <location EndPos="6" StartPos="5" width="2"/>
      <labl>Month</labl>
      <txt>Month.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
    <var ID="CPSIDP" dcml="0" files="ExtractData" intrvl="contin" name="CPSIDP">
      <location EndPos="16" StartPos="7" width="10"/>
      <labl>Person id</labl>
      <txt>Uniquely identifies a person across CPS samples.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
    <var ID="RACE" dcml="0" files="ExtractData" intrvl="discrete" name="RACE">
      <location EndPos="19" StartPos="17" width="3"/>
      <labl>Race</labl>
      <txt>Race.</txt>
      <catgry>
        <catValu>100</catValu>
        <labl>White</labl>
      </catgry>
      <catgry>
        <catValu>200</catValu>
        <labl>Black</labl>
      </catgry>
      <concept vocab="IPUMS">Demographic Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
  </dataDscr>
</codeBook>
"""
# YEAR(4) + MONTH(2) + CPSIDP(10) + RACE(3)
_DELTA_DAT_TEXT = "2006011000000001100\n2006021000000002200\n"


def test_merge_variables_into_bronze_attaches_new_column(tmp_path: Path) -> None:
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(_EXISTING_DAT_TEXT.encode("iso-8859-1"))
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)

    delta_ddi_path = tmp_path / "delta.xml"
    delta_ddi_path.write_text(_DELTA_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "delta.dat.gz"
    delta_data_path.write_bytes(gzip.compress(_DELTA_DAT_TEXT.encode("iso-8859-1")))

    updated_paths = merge_variables_into_bronze(
        delta_data_path, delta_ddi_path, "cps", bronze_dir, new_variables=["RACE"]
    )

    assert updated_paths == [bronze_path(bronze_dir, "cps", 2006)]
    merged = pd.read_parquet(updated_paths[0])
    assert set(merged.columns) == {"YEAR", "MONTH", "CPSIDP", "SEX", "RACE"}
    assert len(merged) == 2  # no row duplication from the join
    by_cpsidp = merged.set_index("CPSIDP")
    assert by_cpsidp.loc[1000000001, "RACE"] == 100
    assert by_cpsidp.loc[1000000001, "SEX"] == 1
    assert by_cpsidp.loc[1000000002, "RACE"] == 200


def test_merge_variables_into_bronze_default_skips_already_present_column(
    tmp_path: Path,
) -> None:
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(_EXISTING_DAT_TEXT.encode("iso-8859-1"))
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)
    before = pd.read_parquet(bronze_path(bronze_dir, "cps", 2006))

    delta_ddi_path = tmp_path / "delta.xml"
    delta_ddi_path.write_text(_DELTA_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "delta.dat.gz"
    delta_data_path.write_bytes(gzip.compress(_DELTA_DAT_TEXT.encode("iso-8859-1")))

    # SEX is already present in bronze - re-merging it (unforced) is a no-op.
    updated_paths = merge_variables_into_bronze(
        delta_data_path, delta_ddi_path, "cps", bronze_dir, new_variables=["SEX"]
    )

    after = pd.read_parquet(updated_paths[0])
    pd.testing.assert_frame_equal(before, after)


# Same DDI/layout as _EXISTING_DDI_XML (YEAR, MONTH, CPSIDP, SEX) but staged
# as a forced re-pull with SEX values flipped from _EXISTING_DAT_TEXT's - the
# scenario force=True exists for (e.g. correcting a value from a bad pull).
_FORCE_DELTA_DAT_TEXT = "20060110000000012\n20060210000000021\n"


def test_merge_variables_into_bronze_force_replaces_existing_column(
    tmp_path: Path,
) -> None:
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(_EXISTING_DAT_TEXT.encode("iso-8859-1"))
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)
    before = pd.read_parquet(bronze_path(bronze_dir, "cps", 2006)).set_index("CPSIDP")
    assert before.loc[1000000001, "SEX"] == 1
    assert before.loc[1000000002, "SEX"] == 2

    delta_ddi_path = tmp_path / "force_delta.xml"
    delta_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "force_delta.dat.gz"
    delta_data_path.write_bytes(
        gzip.compress(_FORCE_DELTA_DAT_TEXT.encode("iso-8859-1"))
    )

    updated_paths = merge_variables_into_bronze(
        delta_data_path,
        delta_ddi_path,
        "cps",
        bronze_dir,
        new_variables=["SEX"],
        force=True,
    )

    after = pd.read_parquet(updated_paths[0]).set_index("CPSIDP")
    assert set(after.columns) == {"YEAR", "MONTH", "SEX"}
    # SEX was replaced from the staged extract...
    assert after.loc[1000000001, "SEX"] == 2
    assert after.loc[1000000002, "SEX"] == 1
    # ...while every other existing column is untouched.
    pd.testing.assert_series_equal(before["YEAR"], after["YEAR"])
    pd.testing.assert_series_equal(before["MONTH"], after["MONTH"])


# Same layout as _EXISTING_DAT_TEXT, plus a third row (CPSIDP 1000000003,
# SEX=1) that the forced delta below deliberately does not cover - e.g. a
# third sample/month sharing this year's bronze file that this particular
# forced request isn't touching.
_EXISTING_DAT_TEXT_3ROWS = "20060110000000011\n20060210000000022\n20060310000000031\n"
# Forces SEX for CPSIDP 1000000001/1000000002 only (flipped values);
# CPSIDP 1000000003 is absent from this extract entirely.
_PARTIAL_FORCE_DELTA_DAT_TEXT = "20060110000000012\n20060210000000021\n"


def test_merge_variables_into_bronze_force_does_not_clobber_rows_outside_delta(
    tmp_path: Path,
) -> None:
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(_EXISTING_DAT_TEXT_3ROWS.encode("iso-8859-1"))
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)

    delta_ddi_path = tmp_path / "partial_force_delta.xml"
    delta_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "partial_force_delta.dat.gz"
    delta_data_path.write_bytes(
        gzip.compress(_PARTIAL_FORCE_DELTA_DAT_TEXT.encode("iso-8859-1"))
    )

    updated_paths = merge_variables_into_bronze(
        delta_data_path,
        delta_ddi_path,
        "cps",
        bronze_dir,
        new_variables=["SEX"],
        force=True,
    )

    after = pd.read_parquet(updated_paths[0]).set_index("CPSIDP")
    assert len(after) == 3  # no rows dropped
    # Rows covered by the forced delta are replaced...
    assert after.loc[1000000001, "SEX"] == 2
    assert after.loc[1000000002, "SEX"] == 1
    # ...but a row the forced delta doesn't mention keeps its prior value,
    # rather than being wiped to NaN.
    assert after.loc[1000000003, "SEX"] == 1


# Two rows sharing MONTH=01 - a duplicate merge key (YEAR, MONTH, since
# CPSIDP isn't a default merge_keys column) on the staged/forced side.
_DUPLICATE_KEY_FORCE_DELTA_DAT_TEXT = "20060100000000019\n20060100000000028\n"


def test_merge_variables_into_bronze_force_raises_on_duplicate_staged_key(
    tmp_path: Path,
) -> None:
    # A duplicate merge key on the staged/forced side used to reach
    # DataFrame.update() before the row-count guard could catch it, crashing
    # with pandas' own "cannot handle a non-unique multi-index" ValueError
    # instead of the intended, more informative RuntimeError.
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(_EXISTING_DAT_TEXT.encode("iso-8859-1"))
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)

    delta_ddi_path = tmp_path / "dup_force_delta.xml"
    delta_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "dup_force_delta.dat.gz"
    delta_data_path.write_bytes(
        gzip.compress(_DUPLICATE_KEY_FORCE_DELTA_DAT_TEXT.encode("iso-8859-1"))
    )

    with pytest.raises(RuntimeError, match="changed the number of rows"):
        merge_variables_into_bronze(
            delta_data_path,
            delta_ddi_path,
            "cps",
            bronze_dir,
            new_variables=["SEX"],
            force=True,
        )


def test_merge_variables_into_bronze_force_preserves_column_order_and_dtype(
    tmp_path: Path,
) -> None:
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(_EXISTING_DAT_TEXT.encode("iso-8859-1"))
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)

    # Simulate an existing bronze file whose SEX column ended up float64
    # (e.g. from an older parse where some other row had a missing value) -
    # a dtype that diverges from the forced delta's int64 SEX below.
    existing_path = bronze_path(bronze_dir, "cps", 2006)
    existing_df = pd.read_parquet(existing_path)
    existing_df["SEX"] = existing_df["SEX"].astype("float64")
    existing_df.to_parquet(existing_path)
    original_columns = list(existing_df.columns)

    delta_ddi_path = tmp_path / "force_delta.xml"
    delta_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "force_delta.dat.gz"
    delta_data_path.write_bytes(
        gzip.compress(_FORCE_DELTA_DAT_TEXT.encode("iso-8859-1"))
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        updated_paths = merge_variables_into_bronze(
            delta_data_path,
            delta_ddi_path,
            "cps",
            bronze_dir,
            new_variables=["SEX"],
            force=True,
        )

    after = pd.read_parquet(updated_paths[0])
    # Column order matches the existing file, not merge_columns-first (what
    # set_index()/reset_index() would otherwise produce).
    assert list(after.columns) == original_columns
    # The existing column's dtype (float64) wins, not the staged delta's
    # (int64) - values are cast to fit rather than the write silently
    # upcasting the whole column.
    assert after["SEX"].dtype == existing_df["SEX"].dtype
    by_cpsidp = after.set_index("CPSIDP")
    assert by_cpsidp.loc[1000000001, "SEX"] == 2
    assert by_cpsidp.loc[1000000002, "SEX"] == 1


def test_merge_variables_into_bronze_raises_when_bronze_year_missing(
    tmp_path: Path,
) -> None:
    delta_ddi_path = tmp_path / "delta.xml"
    delta_ddi_path.write_text(_DELTA_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "delta.dat.gz"
    delta_data_path.write_bytes(gzip.compress(_DELTA_DAT_TEXT.encode("iso-8859-1")))
    bronze_dir = tmp_path / "bronze"  # no existing bronze files at all

    with pytest.raises(RuntimeError, match="missing bronze"):
        merge_variables_into_bronze(
            delta_data_path, delta_ddi_path, "cps", bronze_dir, new_variables=["RACE"]
        )


def test_bronze_coverage_empty_dir_returns_empty_dict(tmp_path: Path) -> None:
    assert bronze_coverage(tmp_path / "does-not-exist") == {}


def test_bronze_coverage_reads_variables_per_year(tmp_path: Path) -> None:
    save_variable_dictionary({"AGE": {}, "SEX": {}}, tmp_path, 2005)
    save_variable_dictionary({"AGE": {}, "RACE": {}}, tmp_path, 2006)

    coverage = bronze_coverage(tmp_path)

    assert coverage == {2005: {"AGE", "SEX"}, 2006: {"AGE", "RACE"}}


def test_bronze_coverage_ignores_non_year_filenames(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "cps_00027.json").write_text("{}", encoding="utf-8")
    save_variable_dictionary({"AGE": {}}, tmp_path, 2006)

    coverage = bronze_coverage(tmp_path)

    assert coverage == {2006: {"AGE"}}


def test_build_and_save_variable_dictionary_filters_to_given_variables(
    tmp_path: Path, make_ddi_xml
) -> None:
    # A variable_delta merge writes only its own columns to bronze, so the
    # dictionary must not claim the rest of the codebook - bronze_coverage
    # reads these files as the record of what bronze actually holds.
    ddi_path = tmp_path / "cps_00001.xml"
    ddi_path.write_text(
        make_ddi_xml(
            [
                ("YEAR", "Survey year", 4),
                ("ASECWT", "ASEC weight", 5),
                ("INCWAGE", "Wage income", 7),
                ("QINCWAGE", "Data quality flag for INCWAGE", 1),
            ]
        ),
        encoding="utf-8",
    )
    dictionaries_dir = tmp_path / "reference"

    build_and_save_variable_dictionary(
        ddi_path, dictionaries_dir, [2006], variables=["INCWAGE", "QINCWAGE"]
    )

    dictionary = load_variable_dictionary(dictionaries_dir, 2006)
    assert set(dictionary) == {"INCWAGE", "QINCWAGE"}


def test_build_and_save_variable_dictionary_keeps_everything_by_default(
    tmp_path: Path, make_ddi_xml
) -> None:
    # A new_samples pull writes the whole file, so the whole codebook is right.
    ddi_path = tmp_path / "cps_00001.xml"
    ddi_path.write_text(
        make_ddi_xml(
            [
                ("YEAR", "Survey year", 4),
                ("INCWAGE", "Wage income", 7),
                ("QINCWAGE", "Data quality flag for INCWAGE", 1),
            ]
        ),
        encoding="utf-8",
    )
    dictionaries_dir = tmp_path / "reference"

    build_and_save_variable_dictionary(ddi_path, dictionaries_dir, [2006])

    assert set(load_variable_dictionary(dictionaries_dir, 2006)) == {
        "YEAR",
        "INCWAGE",
        "QINCWAGE",
    }


def _write_bronze_year(
    bronze_dir: Path, collection: str, year: int, columns: list[str]
) -> Path:
    out_path = bronze_path(bronze_dir, collection, year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({name: [1] for name in columns}).to_parquet(out_path, index=False)
    return out_path


def test_bronze_columns_by_year_reads_columns_from_parquet_footers(
    tmp_path: Path,
) -> None:
    _write_bronze_year(tmp_path, "cps", 2005, ["YEAR", "AGE", "SEX"])
    _write_bronze_year(tmp_path, "cps", 2006, ["YEAR", "AGE"])

    assert bronze_columns_by_year(tmp_path, "cps") == {
        2005: {"YEAR", "AGE", "SEX"},
        2006: {"YEAR", "AGE"},
    }


def test_bronze_columns_by_year_is_empty_when_collection_has_no_bronze(
    tmp_path: Path,
) -> None:
    assert bronze_columns_by_year(tmp_path, "cps") == {}


def test_bronze_columns_by_year_reports_what_the_parquet_holds_not_the_dictionary(
    tmp_path: Path,
) -> None:
    # A dictionary accumulates and never shrinks, so a year whose parquet was
    # overwritten by a narrower extract still has a wide dictionary. This is
    # the disagreement that hid the cps2006_09s overwrite.
    bronze_dir = tmp_path / "bronze"
    dictionaries_dir = tmp_path / "reference"
    _write_bronze_year(bronze_dir, "cps", 2006, ["YEAR", "AGE"])
    save_variable_dictionary(
        {name: {"Description": name} for name in ("YEAR", "AGE", "SEX", "EDUC")},
        dictionaries_dir,
        2006,
    )

    assert bronze_columns_by_year(bronze_dir, "cps") == {2006: {"YEAR", "AGE"}}
    assert bronze_coverage(dictionaries_dir) == {
        2006: {"YEAR", "AGE", "SEX", "EDUC"},
    }


def test_bronze_columns_by_year_skips_and_logs_non_year_parquets(
    tmp_path: Path,
) -> None:
    _write_bronze_year(tmp_path, "cps", 2006, ["YEAR", "AGE"])
    _write_bronze_year(tmp_path, "cps", 2005, ["YEAR"]).rename(
        bronze_path(tmp_path, "cps", 2005).with_suffix(".tmp.parquet")
    )
    (tmp_path / "cps" / "junk.parquet").write_bytes(
        (tmp_path / "cps" / "2006.parquet").read_bytes()
    )

    with structlog.testing.capture_logs() as logs:
        columns_by_year = bronze_columns_by_year(tmp_path, "cps")

    assert columns_by_year == {2006: {"YEAR", "AGE"}}
    reasons = {
        entry["reason"]
        for entry in logs
        if entry["event"] == "ipums_bronze_parquet_skipped"
    }
    assert reasons == {"leftover_tmp_file", "non_year_filename"}


def test_bronze_columns_by_year_skips_and_logs_unreachable_parquet(
    tmp_path: Path,
) -> None:
    _write_bronze_year(tmp_path, "cps", 2005, ["YEAR", "SEX"])
    _write_bronze_year(tmp_path, "cps", 2007, ["YEAR", "AGE"])
    (tmp_path / "cps" / "2006.parquet").write_bytes(b"not a parquet file")

    with structlog.testing.capture_logs() as logs:
        columns_by_year = bronze_columns_by_year(tmp_path, "cps")

    # The unreadable year is left out rather than taking down the other years
    assert columns_by_year == {2005: {"YEAR", "SEX"}, 2007: {"YEAR", "AGE"}}
    reasons = {
        entry["reason"]
        for entry in logs
        if entry["event"] == "ipums_bronze_parquet_skipped"
    }
    assert reasons == {"unreadable_parquet"}


def test_parse_to_bronze_refuses_existing_year_without_replace(tmp_path: Path) -> None:
    # The cps2006_09s regression: a new_samples extract landing on a year that
    # already has bronze used to overwrite every column that year held.
    data_path, ddi_path = _write_fixture(tmp_path)  # 2006 only
    bronze_dir = tmp_path / "bronze"
    existing = bronze_path(bronze_dir, "cps", 2006)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"existing-bronze")

    with pytest.raises(FileExistsError, match="replace=True"):
        parse_to_bronze(data_path, ddi_path, "cps", bronze_dir)

    assert existing.read_bytes() == b"existing-bronze"
    assert list((bronze_dir / "cps").glob("*.tmp.parquet")) == []


def test_parse_to_bronze_replace_overwrites_existing_year(tmp_path: Path) -> None:
    data_path, ddi_path = _write_fixture(tmp_path)
    bronze_dir = tmp_path / "bronze"
    existing = bronze_path(bronze_dir, "cps", 2006)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"existing-bronze")

    out_paths = parse_to_bronze(data_path, ddi_path, "cps", bronze_dir, replace=True)

    assert out_paths == [existing]
    assert bronze_columns_by_year(bronze_dir, "cps") == {2006: {"YEAR", "MONTH", "SEX"}}


def test_parse_to_bronze_years_filter_writes_only_requested_years(
    tmp_path: Path,
) -> None:
    data_path, ddi_path = _write_multi_year_fixture(tmp_path)  # 2005/2006/2007
    bronze_dir = tmp_path / "bronze"

    out_paths = parse_to_bronze(data_path, ddi_path, "cps", bronze_dir, years=[2006])

    assert out_paths == [bronze_path(bronze_dir, "cps", 2006)]
    assert not bronze_path(bronze_dir, "cps", 2005).exists()
    assert not bronze_path(bronze_dir, "cps", 2007).exists()


def test_parse_to_bronze_years_filter_matching_nothing_returns_empty(
    tmp_path: Path,
) -> None:
    data_path, ddi_path = _write_multi_year_fixture(tmp_path)
    bronze_dir = tmp_path / "bronze"

    with structlog.testing.capture_logs() as logs:
        out_paths = parse_to_bronze(
            data_path, ddi_path, "cps", bronze_dir, years=[1999]
        )

    assert out_paths == []
    assert "ipums_parse_no_years_written" in [entry["event"] for entry in logs]


def test_parse_to_bronze_empty_years_filter_writes_nothing(tmp_path: Path) -> None:
    # An empty filter means "no year", not "every year" - a truthiness check
    # here would turn it into a full rebuild.
    data_path, ddi_path = _write_multi_year_fixture(tmp_path)
    bronze_dir = tmp_path / "bronze"

    assert parse_to_bronze(data_path, ddi_path, "cps", bronze_dir, years=[]) == []
    assert not (bronze_dir / "cps").exists()


def test_parse_to_bronze_still_raises_on_empty_extract_under_years_filter(
    tmp_path: Path,
) -> None:
    # An extract with no rows stays a failure; only a filtered-out one is a
    # no-op. The two must not collapse into each other.
    ddi_path = tmp_path / "cps_00027.xml"
    ddi_path.write_text(_DDI_XML, encoding="utf-8")
    data_path = tmp_path / "cps_00027.dat.gz"
    data_path.write_bytes(gzip.compress(b""))

    with pytest.raises(ValueError, match="no rows"):
        parse_to_bronze(data_path, ddi_path, "cps", tmp_path / "bronze", years=[2006])


def test_parse_to_bronze_guard_leavs_other_years_unwritten(tmp_path: Path) -> None:
    # All-or-nothing: a collision on one year must not leave the extract's other years half-landed in bronze
    data_path, ddi_path = _write_multi_year_fixture(tmp_path)
    bronze_dir = tmp_path / "bronze"
    existing = bronze_path(bronze_dir, "cps", 2006)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"existing-bronze")

    with pytest.raises(FileExistsError, match="replace=True"):
        parse_to_bronze(data_path, ddi_path, "cps", bronze_dir, chunksize=1)

    assert existing.read_bytes() == b"existing-bronze"
    assert sorted(p.name for p in (bronze_dir / "cps").iterdir()) == ["2006.parquet"]


def test_parse_to_bronze_removes_tmp_files_when_a_writer_fails_to_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The RuntimeError promises the partials were removed; hold it to that.
    data_path, ddi_path = _write_fixture(tmp_path)
    bronze_dir = tmp_path / "bronze"
    monkeypatch.setattr(
        pq.ParquetWriter,
        "close",
        lambda self: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(RuntimeError, match="were removed"):
        parse_to_bronze(data_path, ddi_path, "cps", bronze_dir)

    assert list((bronze_dir / "cps").glob("*.tmp.parquet")) == []
    assert not bronze_path(bronze_dir, "cps", 2006).exists()


def test_merge_variables_into_bronze_years_filter_restricts_touched_years(
    tmp_path: Path,
) -> None:
    # Repairing one year must not re-merge a delta into every other year it
    # happens to cover.
    existing_ddi_path = tmp_path / "existing.xml"
    existing_ddi_path.write_text(_EXISTING_DDI_XML, encoding="utf-8")
    existing_data_path = tmp_path / "existing.dat.gz"
    existing_data_path.write_bytes(
        gzip.compress(
            # YEAR(4) + MONTH(2) + CPSIDP(10) + SEX(1), one row per year.
            b"20050110000000011\n20060110000000021\n20070110000000031\n"
        )
    )
    bronze_dir = tmp_path / "bronze"
    parse_to_bronze(existing_data_path, existing_ddi_path, "cps", bronze_dir)
    untouched_before = bronze_path(bronze_dir, "cps", 2005).read_bytes()

    delta_ddi_path = tmp_path / "delta.xml"
    delta_ddi_path.write_text(_DELTA_DDI_XML, encoding="utf-8")
    delta_data_path = tmp_path / "delta.dat.gz"
    delta_data_path.write_bytes(
        gzip.compress(
            # YEAR(4) + MONTH(2) + CPSIDP(10) + RACE(3)
            b"20050110000000011 00\n20060110000000021 00\n20070110000000031 00\n".replace(
                b" ", b"1"
            )
        )
    )

    updated = merge_variables_into_bronze(
        delta_data_path,
        delta_ddi_path,
        "cps",
        bronze_dir,
        new_variables=["RACE"],
        years=[2006],
    )

    assert updated == [bronze_path(bronze_dir, "cps", 2006)]
    assert "RACE" in bronze_columns_by_year(bronze_dir, "cps")[2006]
    assert "RACE" not in bronze_columns_by_year(bronze_dir, "cps")[2007]
    assert bronze_path(bronze_dir, "cps", 2005).read_bytes() == untouched_before
