import gzip
from pathlib import Path

import pandas as pd
import pytest
import structlog.testing
import yaml

from src.extractors.base import build_extraction_record
from src.extractors.manifest import MANIFEST_FILENAME, append_to_manifest, read_manifest
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
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
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
        collection="cps",
        dictionaries_dir=reference_dir,
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


@pytest.mark.parametrize(
    ("bad_entry", "reason"),
    [
        pytest.param(
            {"extraction_id": "cps_00002", "metadata": None},
            "missing_required_metadata_keys",
            id="metadata_scalar",
        ),
        pytest.param(
            {"extraction_id": "cps_00002", "metadata": {"collection": "cps"}},
            "missing_required_metadata_keys",
            id="metadata_missing_required_keys",
        ),
        pytest.param(
            {
                "extraction_id": "cps_00002",
                "metadata": {
                    "samples": ["cps2006_09s"],
                    "variables": ["AGE"],
                    "ddi_path": "/nonexistent/delta.xml",
                    "extract_id": 2,
                },
            },
            "no_file_path",
            id="entry_missing_file_path",
        ),
    ],
)
def test_malformed_manifest_entry_is_skipped_not_raised(
    tmp_path: Path, bad_entry: dict, reason: str
) -> None:
    # A hand-edited/truncated manifest can carry an entry with `metadata:`
    # missing or scalar - extractors.ipums_ddi.collection_flag_registry and
    # extractors.ipums_coverage.build_coverage already warn-and-skip this
    # rather than raise; _collection_manifest_entries must do the same
    # instead of aborting the whole pipeline run over one bad entry.
    data_root = tmp_path / "data"
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

    # Simulate a hand-truncated manifest: append a malformed entry directly,
    # bypassing append_to_manifest (which always writes well-formed records).
    manifest_path = collection_dir / MANIFEST_FILENAME
    entries = read_manifest(collection_dir)
    entries.append(bad_entry)
    manifest_path.write_text(yaml.safe_dump(entries, sort_keys=False))

    with structlog.testing.capture_logs() as logs:
        bronze_paths = parse_ipums_extracts(
            external_dir,
            bronze_dir,
            collection="cps",
            dictionaries_dir=reference_dir,
        )

    assert bronze_paths == [bronze_path(bronze_dir, "cps", 2006)]
    result = pd.read_parquet(bronze_path(bronze_dir, "cps", 2006))
    assert set(result.columns) == {"YEAR", "MONTH", "AGE", "SEX"}
    # Skipped *loudly* - a silent skip makes a corrupted manifest look like a
    # short one, so the entry is never parsed and nothing says why.
    assert [
        entry["reason"]
        for entry in logs
        if entry["event"] == "ipums_manifest_entry_skipped"
    ] == [reason]


def test_raises_when_no_manifest_entry_is_backed_by_files(tmp_path: Path) -> None:
    # An empty/absent manifest must raise rather than return [] - a silent
    # empty result reads as "already up to date" and hides a failed extract.
    data_root = tmp_path / "data"
    external_dir = data_root / "external" / "ipums"

    with pytest.raises(RuntimeError, match="No downloaded IPUMS extract found"):
        parse_ipums_extracts(
            external_dir,
            data_root / "bronze" / "ipums",
            collection="cps",
            dictionaries_dir=data_root / "reference" / "ipums" / "cps",
        )


def test_dictionaries_dir_routes_reads_and_writes_away_from_the_default(
    tmp_path: Path,
) -> None:
    # The injected directory must carry both halves: the dictionary is written
    # there, and bronze_coverage reads it back from there on the next run.
    data_root = tmp_path / "data"
    external_dir = data_root / "external" / "ipums"
    bronze_dir = data_root / "bronze" / "ipums"
    elsewhere = tmp_path / "elsewhere" / "dictionaries"
    collection_dir = external_dir / "cps"

    full_ddi_xml = _DDI_TEMPLATE.format(
        filename="full.dat", vars=_YEAR_VAR + _MONTH_VAR + _AGE_VAR + _SEX_VAR
    )
    data_path, ddi_path = _write_extract(
        collection_dir, "full", full_ddi_xml, _FULL_DAT_TEXT
    )
    append_to_manifest(
        collection_dir,
        build_extraction_record(
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
        ),
    )

    parse_ipums_extracts(
        external_dir, bronze_dir, collection="cps", dictionaries_dir=elsewhere
    )

    # Written to the injected directory, not the settings-derived default.
    assert load_variable_dictionary(elsewhere, 2006)["AGE"]

    # Second run: coverage read back from `elsewhere` marks 2006 done, so
    # nothing is reparsed. This is the half that fails if only the write is
    # routed and bronze_coverage still reads the default.
    assert (
        parse_ipums_extracts(
            external_dir, bronze_dir, collection="cps", dictionaries_dir=elsewhere
        )
        == []
    )


def test_parse_ipums_extracts_calls_parse_to_bronze_with_expected_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression guard for a positional-arg mismatch that silently slipped
    # an unwanted 5th argument (extract_id) into a parse_to_bronze call -
    # asserting the exact arg list (not just that it was called) is what
    # would have caught it.
    data_root = tmp_path / "data"
    external_dir = data_root / "external" / "ipums"
    bronze_dir = data_root / "bronze" / "ipums"
    reference_dir = data_root / "reference" / "ipums" / "cps"
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
        collection="cps",
        dictionaries_dir=reference_dir,
    )

    assert calls == [(data_path, ddi_path, "cps", bronze_dir)]


# --- Data quality flags reaching bronze -------------------------------------
#
# IPUMS attaches a flag column to every requested variable that has one. A
# "new_samples" pull carries them into bronze automatically (parse_to_bronze
# writes the whole file), but a "variable_delta" merge keeps only the columns
# it is told about - so the flags have to be named explicitly or they are
# dropped while sitting in the .dat.gz.

_DELTA_MERGE_KEYS: list[tuple[str, str, int]] = [
    ("YEAR", "Survey year", 4),
    ("MONTH", "Month", 2),
    ("SERIAL", "Household serial number", 3),
    ("PERNUM", "Person number in sample unit", 2),
]


def _seed_bronze_and_delta(
    tmp_path: Path,
    make_ddi_xml,
    make_fixed_width_dat,
    delta_vars: list[tuple[str, str, int]],
    requested: tuple[str, ...],
) -> tuple[Path, Path, Path]:
    """A bronze year written by a full pull, plus a pending delta entry."""
    data_root = tmp_path / "data"
    external_dir = data_root / "external" / "ipums"
    bronze_dir = data_root / "bronze" / "ipums"
    reference_dir = data_root / "reference" / "ipums" / "cps"
    collection_dir = external_dir / "cps"
    collection_dir.mkdir(parents=True, exist_ok=True)

    full_vars = [*_DELTA_MERGE_KEYS, ("AGE", "Age", 2), ("SEX", "Sex", 1)]
    full_rows = [[2006, 1, 1, 1, 25, 1], [2006, 2, 3, 1, 30, 2]]
    (collection_dir / "full.xml").write_text(make_ddi_xml(full_vars), encoding="utf-8")
    (collection_dir / "full.dat.gz").write_bytes(
        make_fixed_width_dat(full_rows, full_vars)
    )
    append_to_manifest(
        collection_dir,
        build_extraction_record(
            source="ipums_api",
            extraction_id="cps_00001",
            file_path=collection_dir / "full.dat.gz",
            metadata={
                "collection": "cps",
                "samples": ("cps2006_09s",),
                "variables": ("AGE", "SEX"),
                "ddi_path": str(collection_dir / "full.xml"),
                "extract_id": 1,
                "request_kind": "new_samples",
                "force": False,
            },
        ),
    )

    delta_rows = [
        [2006, 1, 1, 1, *[7] * (len(delta_vars) - 4)],
        [2006, 2, 3, 1, *[8] * (len(delta_vars) - 4)],
    ]
    (collection_dir / "delta.xml").write_text(
        make_ddi_xml(delta_vars), encoding="utf-8"
    )
    (collection_dir / "delta.dat.gz").write_bytes(
        make_fixed_width_dat(delta_rows, delta_vars)
    )
    append_to_manifest(
        collection_dir,
        build_extraction_record(
            source="ipums_api",
            extraction_id="cps_00002",
            file_path=collection_dir / "delta.dat.gz",
            metadata={
                "collection": "cps",
                "samples": ("cps2006_09s",),
                "variables": requested,
                "ddi_path": str(collection_dir / "delta.xml"),
                "extract_id": 2,
                "request_kind": "variable_delta",
                "force": False,
            },
        ),
    )
    return external_dir, bronze_dir, reference_dir


def _run_parse(external_dir: Path, bronze_dir: Path, reference_dir: Path) -> None:
    parse_ipums_extracts(
        external_dir,
        bronze_dir,
        collection="cps",
        dictionaries_dir=reference_dir,
    )


def test_variable_delta_merge_includes_quality_flag_columns(
    tmp_path: Path, make_ddi_xml, make_fixed_width_dat
) -> None:
    delta_vars = [
        *_DELTA_MERGE_KEYS,
        ("INCWAGE", "Wage income", 6),
        ("QINCWAGE", "Data quality flag for INCWAGE", 1),
    ]
    external_dir, bronze_dir, reference_dir = _seed_bronze_and_delta(
        tmp_path,
        make_ddi_xml,
        make_fixed_width_dat,
        delta_vars,
        requested=("INCWAGE",),
    )

    _run_parse(external_dir, bronze_dir, reference_dir)

    columns = set(pd.read_parquet(bronze_path(bronze_dir, "cps", 2006)).columns)
    assert "INCWAGE" in columns
    # The flag was never in the request's `variables` - it has to come from
    # the codebook or it is silently lost.
    assert "QINCWAGE" in columns


def test_variable_delta_merge_includes_topcode_flag_column(
    tmp_path: Path, make_ddi_xml, make_fixed_width_dat
) -> None:
    delta_vars = [
        *_DELTA_MERGE_KEYS,
        ("INCFARM", "Farm income", 6),
        ("TINCFARM", "Topcode Flag for INCFARM", 1),
    ]
    external_dir, bronze_dir, reference_dir = _seed_bronze_and_delta(
        tmp_path,
        make_ddi_xml,
        make_fixed_width_dat,
        delta_vars,
        requested=("INCFARM",),
    )

    _run_parse(external_dir, bronze_dir, reference_dir)

    columns = set(pd.read_parquet(bronze_path(bronze_dir, "cps", 2006)).columns)
    assert "TINCFARM" in columns


def test_variable_delta_merge_still_drops_untouched_technical_columns(
    tmp_path: Path, make_ddi_xml, make_fixed_width_dat
) -> None:
    # ASECWT rides along in the delta extract but was not requested and is not
    # a flag; merging it would collide with what bronze already holds.
    delta_vars = [
        *_DELTA_MERGE_KEYS,
        ("ASECWT", "ASEC weight", 5),
        ("INCWAGE", "Wage income", 6),
        ("QINCWAGE", "Data quality flag for INCWAGE", 1),
    ]
    external_dir, bronze_dir, reference_dir = _seed_bronze_and_delta(
        tmp_path,
        make_ddi_xml,
        make_fixed_width_dat,
        delta_vars,
        requested=("INCWAGE",),
    )

    _run_parse(external_dir, bronze_dir, reference_dir)

    columns = set(pd.read_parquet(bronze_path(bronze_dir, "cps", 2006)).columns)
    assert "QINCWAGE" in columns
    assert "ASECWT" not in columns


def test_variable_delta_dictionary_records_only_merged_columns(
    tmp_path: Path, make_ddi_xml, make_fixed_width_dat
) -> None:
    # The reference dictionary is read back as the record of what bronze holds
    # (bronze_coverage). If it claims columns the merge dropped, the entry is
    # reported as covered and can never be reprocessed to add them.
    delta_vars = [
        *_DELTA_MERGE_KEYS,
        ("ASECWT", "ASEC weight", 5),
        ("INCWAGE", "Wage income", 6),
        ("QINCWAGE", "Data quality flag for INCWAGE", 1),
    ]
    external_dir, bronze_dir, reference_dir = _seed_bronze_and_delta(
        tmp_path,
        make_ddi_xml,
        make_fixed_width_dat,
        delta_vars,
        requested=("INCWAGE",),
    )

    _run_parse(external_dir, bronze_dir, reference_dir)

    dictionary = load_variable_dictionary(reference_dir, 2006)
    parquet_columns = set(pd.read_parquet(bronze_path(bronze_dir, "cps", 2006)).columns)
    assert "QINCWAGE" in dictionary
    assert "ASECWT" not in dictionary
    assert set(dictionary) <= parquet_columns


def test_variable_delta_is_reprocessed_when_only_its_flag_is_missing(
    tmp_path: Path, make_ddi_xml, make_fixed_width_dat
) -> None:
    # The regression that made the flag gap unbackfillable: with the entry's
    # column set taken from `variables` alone, a bronze file already holding
    # INCWAGE but not QINCWAGE looked fully covered and was skipped forever.
    delta_vars = [
        *_DELTA_MERGE_KEYS,
        ("INCWAGE", "Wage income", 6),
        ("QINCWAGE", "Data quality flag for INCWAGE", 1),
    ]
    external_dir, bronze_dir, reference_dir = _seed_bronze_and_delta(
        tmp_path,
        make_ddi_xml,
        make_fixed_width_dat,
        delta_vars,
        requested=("INCWAGE",),
    )
    # Pretend a previous (pre-fix) run merged INCWAGE but not its flag.
    save_variable_dictionary(
        {"YEAR": {}, "MONTH": {}, "AGE": {}, "SEX": {}, "INCWAGE": {}},
        reference_dir,
        2006,
    )

    _run_parse(external_dir, bronze_dir, reference_dir)

    columns = set(pd.read_parquet(bronze_path(bronze_dir, "cps", 2006)).columns)
    assert "QINCWAGE" in columns


def test_unparseable_delta_ddi_surfaces_rather_than_half_merging(
    tmp_path: Path, make_ddi_xml, make_fixed_width_dat
) -> None:
    delta_vars = [
        *_DELTA_MERGE_KEYS,
        ("INCWAGE", "Wage income", 6),
        ("QINCWAGE", "Data quality flag for INCWAGE", 1),
    ]
    external_dir, bronze_dir, reference_dir = _seed_bronze_and_delta(
        tmp_path,
        make_ddi_xml,
        make_fixed_width_dat,
        delta_vars,
        requested=("INCWAGE",),
    )
    (external_dir / "cps" / "delta.xml").write_text("<codeBook/>")

    # Column selection falls back to the requested list (merge_column_names
    # with no summary), but merge_variables_into_bronze needs the same
    # codebook to read the fixed-width data at all - so an unreadable DDI
    # fails loudly instead of quietly writing a year with missing columns.
    with pytest.raises(Exception):
        _run_parse(external_dir, bronze_dir, reference_dir)
