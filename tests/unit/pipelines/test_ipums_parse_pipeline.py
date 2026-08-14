import gzip
from pathlib import Path

import pandas as pd
import pytest

from src.config.sources import IPUMSExtractRequest
from src.extractors.base import build_extraction_record
from src.extractors.manifest import append_to_manifest
from src.parsers.ipums import (
    bronze_path,
    load_variable_dictionary,
    parse_to_bronze,
    save_variable_dictionary,
)
from src.pipelines.ipums_parse_pipeline import (
    _entry_needs_processing,
    parse_ipums_extracts,
)


def test_needs_processing_when_year_fully_missing() -> None:
    coverage: dict[int, set[str]] = {}

    assert _entry_needs_processing(coverage, {2006}, {"AGE", "SEX"}) is True


def test_needs_processing_when_variable_missing_for_a_covered_year() -> None:
    coverage = {2006: {"AGE"}}

    assert _entry_needs_processing(coverage, {2006}, {"AGE", "SEX"}) is True


def test_no_processing_needed_when_fully_covered() -> None:
    coverage = {2006: {"AGE", "SEX", "YEAR"}}

    assert _entry_needs_processing(coverage, {2006}, {"AGE", "SEX"}) is False


def test_needs_processing_when_only_some_years_covered() -> None:
    coverage = {2006: {"AGE", "SEX"}}

    assert _entry_needs_processing(coverage, {2006, 2007}, {"AGE", "SEX"}) is True


def test_needs_processing_when_no_year_could_be_parsed() -> None:
    # Fail safe: an entry we can't determine years for is always processed
    # rather than silently skipped.
    coverage = {2006: {"AGE", "SEX"}}

    assert _entry_needs_processing(coverage, set(), {"AGE", "SEX"}) is True


def test_needs_processing_force_true_even_when_fully_covered() -> None:
    # The whole reason force exists: a fully-covered entry must still be
    # (re)processed when it was a deliberate forced refresh.
    coverage = {2006: {"AGE", "SEX", "YEAR"}}

    assert _entry_needs_processing(coverage, {2006}, {"AGE", "SEX"}, force=True) is True


# --- End-to-end: force-refreshing one variable must not clobber the rest of
# --- the year's bronze columns. This is the scenario that motivated the
# --- whole "force" plan.

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

_YEAR_VAR = """
    <var ID="YEAR" dcml="0" files="ExtractData" intrvl="contin" name="YEAR">
      <location EndPos="4" StartPos="1" width="4"/>
      <labl>Survey year</labl>
      <txt>Survey year.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
"""
_MONTH_VAR = """
    <var ID="MONTH" dcml="0" files="ExtractData" intrvl="discrete" name="MONTH">
      <location EndPos="6" StartPos="5" width="2"/>
      <labl>Month</labl>
      <txt>Month.</txt>
      <concept vocab="IPUMS">Technical Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
"""
_AGE_VAR = """
    <var ID="AGE" dcml="0" files="ExtractData" intrvl="contin" name="AGE">
      <location EndPos="8" StartPos="7" width="2"/>
      <labl>Age</labl>
      <txt>Age.</txt>
      <concept vocab="IPUMS">Demographic Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
"""
_SEX_VAR = """
    <var ID="SEX" dcml="0" files="ExtractData" intrvl="discrete" name="SEX">
      <location EndPos="9" StartPos="9" width="1"/>
      <labl>Sex</labl>
      <txt>Sex.</txt>
      <concept vocab="IPUMS">Demographic Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
"""
# Same shape as _AGE_VAR but with a different label - stands in for a
# forced re-pull correcting a bad prior variable definition, not just bad
# prior bronze values.
_AGE_VAR_CORRECTED = """
    <var ID="AGE" dcml="0" files="ExtractData" intrvl="contin" name="AGE">
      <location EndPos="8" StartPos="7" width="2"/>
      <labl>Age (corrected)</labl>
      <txt>Age.</txt>
      <concept vocab="IPUMS">Demographic Variables</concept>
      <varFormat schema="other" type="numeric"/>
    </var>
"""

# YEAR(4) MONTH(2) AGE(2) SEX(1): full "new_samples" pull.
_FULL_DAT_TEXT = "200601251\n200602302\n"
# YEAR(4) MONTH(2) AGE(2): forced re-pull of AGE alone, corrected values.
_FORCED_AGE_DAT_TEXT = "20060199\n20060288\n"


def _write_extract(
    collection_dir: Path, name: str, ddi_xml: str, dat_text: str
) -> tuple[Path, Path]:
    collection_dir.mkdir(parents=True, exist_ok=True)
    ddi_path = collection_dir / f"{name}.xml"
    ddi_path.write_text(ddi_xml, encoding="utf-8")
    data_path = collection_dir / f"{name}.dat.gz"
    data_path.write_bytes(gzip.compress(dat_text.encode("iso-8859-1")))
    return data_path, ddi_path


def test_force_refresh_replaces_variable_without_clobbering_other_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setattr(
        "src.pipelines.ipums_parse_pipeline.settings.paths.root", data_root
    )
    external_dir = data_root / "external" / "ipums"
    bronze_dir = data_root / "bronze" / "ipums"
    reference_dir = data_root / "reference" / "ipums" / "cps"
    collection_dir = external_dir / "cps"

    full_ddi_xml = _DDI_TEMPLATE.format(
        filename="full.dat", vars=_YEAR_VAR + _MONTH_VAR + _AGE_VAR + _SEX_VAR
    )
    full_data_path, full_ddi_path = _write_extract(
        collection_dir, "full", full_ddi_xml, _FULL_DAT_TEXT
    )
    full_record = build_extraction_record(
        source="ipums_api",
        extraction_id="cps_00001",
        file_path=full_data_path,
        metadata={
            "collection": "cps",
            "samples": ("cps2006_09s",),
            "variables": ("AGE", "SEX"),
            "ddi_path": str(full_ddi_path),
            "extract_id": 1,
            "request_kind": "new_samples",
            "force": False,
        },
    )
    append_to_manifest(collection_dir, full_record)
    # Seed the state a prior, already-completed pipeline run would have left:
    # the bronze parquet for 2006 (AGE, SEX) plus its variable dictionary.
    # The "full" manifest entry above is thus already covered and will be
    # skipped by _entry_needs_processing - it's the forced entry that must
    # still be processed despite that.
    parse_to_bronze(full_data_path, full_ddi_path, "cps", bronze_dir)
    save_variable_dictionary(
        {"AGE": {"Description": "Age (bad pull)"}, "SEX": {}}, reference_dir, 2006
    )

    forced_ddi_xml = _DDI_TEMPLATE.format(
        filename="forced.dat", vars=_YEAR_VAR + _MONTH_VAR + _AGE_VAR_CORRECTED
    )
    forced_data_path, forced_ddi_path = _write_extract(
        collection_dir, "forced", forced_ddi_xml, _FORCED_AGE_DAT_TEXT
    )
    forced_record = build_extraction_record(
        source="ipums_api",
        extraction_id="cps_00002",
        file_path=forced_data_path,
        metadata={
            "collection": "cps",
            "samples": ("cps2006_09s",),
            "variables": ("AGE",),
            "ddi_path": str(forced_ddi_path),
            "extract_id": 2,
            "request_kind": "variable_delta",
            "force": True,
        },
    )
    append_to_manifest(collection_dir, forced_record)

    parse_ipums_extracts(
        external_dir,
        bronze_dir,
        extracts=[
            IPUMSExtractRequest(
                collection="cps", samples=("cps2006_09s",), variables=("AGE", "SEX")
            )
        ],
    )

    result = pd.read_parquet(bronze_path(bronze_dir, "cps", 2006)).sort_values("MONTH")
    assert set(result.columns) == {"YEAR", "MONTH", "AGE", "SEX"}
    assert result["SEX"].tolist() == [1, 2]  # untouched
    assert result["AGE"].tolist() == [99, 88]  # replaced by the forced extract

    dictionary = load_variable_dictionary(reference_dir, 2006)
    # force also overrides a drifted variable *definition*, not just bronze
    # values - otherwise the dictionary would keep describing AGE using the
    # bad prior pull's (possibly wrong) definition forever.
    assert dictionary["AGE"]["Description"] == "Age (corrected)"


def test_parse_ipums_extracts_calls_parse_to_bronze_with_expected_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression guard for a positional-arg mismatch that silently slipped
    # an unwanted 5th argument (extract_id) into a parse_to_bronze call -
    # asserting the exact arg list (not just that it was called) is what
    # would have caught it.
    data_root = tmp_path / "data"
    monkeypatch.setattr(
        "src.pipelines.ipums_parse_pipeline.settings.paths.root", data_root
    )
    external_dir = data_root / "external" / "ipums"
    bronze_dir = data_root / "bronze" / "ipums"
    collection_dir = external_dir / "cps"

    full_ddi_xml = _DDI_TEMPLATE.format(
        filename="full.dat", vars=_YEAR_VAR + _MONTH_VAR + _AGE_VAR + _SEX_VAR
    )
    data_path, ddi_path = _write_extract(
        collection_dir, "full", full_ddi_xml, _FULL_DAT_TEXT
    )
    record = build_extraction_record(
        source="ipums_api",
        extraction_id="cps_00001",
        file_path=data_path,
        metadata={
            "collection": "cps",
            "samples": ("cps2006_09s",),
            "variables": ("AGE", "SEX"),
            "ddi_path": str(ddi_path),
            "extract_id": 1,
            "request_kind": "new_samples",
            "force": False,
        },
    )
    append_to_manifest(collection_dir, record)

    calls: list[tuple] = []

    def fake_parse_to_bronze(
        data_path: Path, ddi_path: Path, collection: str, bronze_dir: Path
    ):
        calls.append((data_path, ddi_path, collection, bronze_dir))
        return [bronze_path(bronze_dir, collection, 2006)]

    monkeypatch.setattr(
        "src.pipelines.ipums_parse_pipeline.parse_to_bronze", fake_parse_to_bronze
    )

    parse_ipums_extracts(
        external_dir,
        bronze_dir,
        extracts=[
            IPUMSExtractRequest(
                collection="cps", samples=("cps2006_09s",), variables=("AGE", "SEX")
            )
        ],
    )

    assert calls == [(data_path, ddi_path, "cps", bronze_dir)]
