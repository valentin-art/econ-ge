import gzip
from pathlib import Path

import pandas as pd
import pytest
from ipumspy import readers

from src.parsers.ipums import (
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
