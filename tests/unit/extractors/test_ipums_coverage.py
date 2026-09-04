from pathlib import Path

import pytest
import structlog.testing
import yaml

from src.extractors.base import build_extraction_record
from src.extractors.ipums_coverage import (
    CollectionCoverage,
    SampleCoverage,
    build_coverage,
    parse_sample_year,
    plan_delta_requests,
    plan_force_requests,
    save_coverage,
)
from src.extractors.manifest import append_to_manifest


def _add_manifest_entry(
    collection_dir: Path,
    extraction_id: str,
    samples: list[str],
    variables: list[str],
    make_files: bool = True,
    delivered: list[str] | None = None,
    ddi_xml: str | None = None,
) -> None:
    data_path = collection_dir / f"{extraction_id}.dat.gz"
    ddi_path = collection_dir / f"{extraction_id}.xml"
    # build_extraction_record reads the file to checksum it, so it must exist
    # at record-build time regardless; make_files=False removes it afterward
    # to simulate a manifest entry whose files were later deleted.
    data_path.write_bytes(b"data")
    # A deliberately unparseable stub unless the caller supplies a real
    # codebook: most of these tests are about manifest bookkeeping, and the
    # stub proves the fallbacks work when a codebook can't be read.
    ddi_path.write_text(ddi_xml if ddi_xml is not None else "<codeBook/>")
    metadata = {
        "collection": "cps",
        "samples": samples,
        "variables": variables,
        "extract_id": int(extraction_id.rsplit("_", 1)[-1]),
        "ddi_path": str(ddi_path),
        "cached": False,
        "request_kind": "new_samples",
    }
    if delivered is not None:
        metadata["delivered_variables"] = delivered
    record = build_extraction_record(
        source="ipums_api",
        extraction_id=extraction_id,
        file_path=data_path,
        metadata=metadata,
    )
    append_to_manifest(collection_dir, record)
    if not make_files:
        data_path.unlink()
        ddi_path.unlink()


def test_parse_sample_year() -> None:
    assert parse_sample_year("cps2006_09s") == 2006
    assert parse_sample_year("cps1962_03s") == 1962
    assert parse_sample_year("no-year-here") is None


def test_build_coverage_unions_variables_per_sample(tmp_path: Path) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(collection_dir, "cps_00029", ["cps2006_09s"], ["AGE", "SEX"])
    _add_manifest_entry(
        collection_dir,
        "cps_00030",
        ["cps2025_03s", "cps2024_03s"],
        ["YEAR", "SEX", "RACE"],
    )

    coverage = build_coverage(collection_dir, "cps")

    assert set(coverage.samples) == {"cps2006_09s", "cps2025_03s", "cps2024_03s"}
    assert coverage.samples["cps2006_09s"].requested_variables == frozenset(
        {"AGE", "SEX"}
    )
    assert coverage.samples["cps2025_03s"].extraction_ids == ("cps_00030",)
    assert coverage.requested_variables == frozenset({"AGE", "SEX", "YEAR", "RACE"})
    assert coverage.years == frozenset({2006, 2025, 2024})


def test_build_coverage_skips_entries_with_missing_files(tmp_path: Path) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(
        collection_dir, "cps_00029", ["cps2006_09s"], ["AGE"], make_files=False
    )

    coverage = build_coverage(collection_dir, "cps")

    assert coverage.samples == {}


def test_build_coverage_accumulates_extraction_ids_for_same_sample(
    tmp_path: Path,
) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(collection_dir, "cps_00030", ["cps2025_03s"], ["AGE"])
    _add_manifest_entry(collection_dir, "cps_00032", ["cps2025_03s"], ["AGE"])

    coverage = build_coverage(collection_dir, "cps")

    assert coverage.samples["cps2025_03s"].extraction_ids == ("cps_00030", "cps_00032")


def test_build_coverage_dedupes_repeated_extraction_id_for_same_sample(
    tmp_path: Path,
) -> None:
    # extract() appends a manifest entry on every call, including cache hits
    # that just re-point at an already-recorded extraction_id.
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(collection_dir, "cps_00030", ["cps2025_03s"], ["AGE"])
    _add_manifest_entry(collection_dir, "cps_00030", ["cps2025_03s"], ["AGE"])

    coverage = build_coverage(collection_dir, "cps")

    assert coverage.samples["cps2025_03s"].extraction_ids == ("cps_00030",)


def test_build_coverage_prefers_recorded_delivered_variables(tmp_path: Path) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(
        collection_dir,
        "cps_00029",
        ["cps2006_09s"],
        ["INCWAGE"],
        delivered=["YEAR", "INCWAGE", "QINCWAGE"],
    )

    coverage = build_coverage(collection_dir, "cps")

    sample = coverage.samples["cps2006_09s"]
    assert sample.requested_variables == frozenset({"INCWAGE"})
    assert sample.delivered_variables == frozenset({"YEAR", "INCWAGE", "QINCWAGE"})


def test_build_coverage_backfills_delivered_variables_from_ddi(
    tmp_path: Path, make_ddi_xml
) -> None:
    # A manifest entry written before delivered_variables existed: the
    # codebook is on disk and is the ground truth for what the file holds.
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(
        collection_dir,
        "cps_00029",
        ["cps2006_09s"],
        ["INCWAGE"],
        ddi_xml=make_ddi_xml(
            [
                ("YEAR", "Survey year", 4),
                ("INCWAGE", "Wage income", 7),
                ("QINCWAGE", "Data quality flag for INCWAGE", 1),
            ]
        ),
    )

    coverage = build_coverage(collection_dir, "cps")

    sample = coverage.samples["cps2006_09s"]
    assert sample.requested_variables == frozenset({"INCWAGE"})
    assert sample.delivered_variables == frozenset({"YEAR", "INCWAGE", "QINCWAGE"})


def test_build_coverage_falls_back_to_requested_when_ddi_unparseable(
    tmp_path: Path,
) -> None:
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(collection_dir, "cps_00029", ["cps2006_09s"], ["AGE", "SEX"])

    coverage = build_coverage(collection_dir, "cps")

    sample = coverage.samples["cps2006_09s"]
    assert sample.delivered_variables == frozenset({"AGE", "SEX"})


def test_save_coverage_writes_summary(tmp_path: Path) -> None:
    coverage = CollectionCoverage(
        collection="cps",
        samples={
            "cps2006_09s": SampleCoverage(
                requested_variables=frozenset({"AGE", "SEX"}),
                delivered_variables=frozenset({"AGE", "SEX"}),
                extraction_ids=("cps_00029",),
            )
        },
    )

    out_path = save_coverage(coverage, tmp_path)

    assert out_path == tmp_path / "_COVERAGE.yaml"
    payload = yaml.safe_load(out_path.read_text())
    assert payload["collection"] == "cps"
    assert payload["years"] == [2006]
    assert payload["variables"] == ["AGE", "SEX"]
    assert payload["samples"]["cps2006_09s"]["extraction_ids"] == ["cps_00029"]


def test_save_coverage_reports_delivered_and_requested_separately(
    tmp_path: Path,
) -> None:
    coverage = CollectionCoverage(
        collection="cps",
        samples={
            "cps2006_09s": SampleCoverage(
                requested_variables=frozenset({"INCWAGE"}),
                delivered_variables=frozenset({"YEAR", "INCWAGE", "QINCWAGE"}),
                extraction_ids=("cps_00029",),
            )
        },
    )

    payload = yaml.safe_load(save_coverage(coverage, tmp_path).read_text())

    # `variables` is what the files really contain, flags included.
    assert payload["variables"] == ["INCWAGE", "QINCWAGE", "YEAR"]
    assert payload["requested_variables"] == ["INCWAGE"]
    sample = payload["samples"]["cps2006_09s"]
    assert sample["variables"] == ["INCWAGE", "QINCWAGE", "YEAR"]
    assert sample["requested_variables"] == ["INCWAGE"]


def _coverage_with_one_sample(
    variables: set[str], delivered: set[str] | None = None
) -> CollectionCoverage:
    return CollectionCoverage(
        collection="cps",
        samples={
            "cps2006_09s": SampleCoverage(
                requested_variables=frozenset(variables),
                delivered_variables=frozenset(
                    delivered if delivered is not None else variables
                ),
                extraction_ids=("cps_00029",),
            )
        },
    )


def test_plan_delta_requests_nothing_new() -> None:
    coverage = _coverage_with_one_sample({"AGE", "SEX"})

    planned = plan_delta_requests(coverage, ["cps2006_09s"], ["AGE"])

    assert planned == []


def test_plan_delta_requests_new_samples_only() -> None:
    coverage = _coverage_with_one_sample({"AGE", "SEX"})

    planned = plan_delta_requests(
        coverage, ["cps2006_09s", "cps2007_09s"], ["AGE", "SEX"]
    )

    assert len(planned) == 1
    assert planned[0].request_kind == "new_samples"
    assert planned[0].samples == ("cps2007_09s",)
    assert planned[0].variables == ("AGE", "SEX")


def test_plan_delta_requests_variable_delta_only() -> None:
    coverage = _coverage_with_one_sample({"AGE", "SEX"})

    planned = plan_delta_requests(coverage, ["cps2006_09s"], ["AGE", "RACE"])

    assert len(planned) == 1
    assert planned[0].request_kind == "variable_delta"
    assert planned[0].samples == ("cps2006_09s",)
    assert planned[0].variables == ("RACE",)


def test_plan_delta_requests_both_new_samples_and_variable_delta() -> None:
    coverage = _coverage_with_one_sample({"AGE", "SEX"})

    planned = plan_delta_requests(
        coverage, ["cps2006_09s", "cps2007_09s"], ["AGE", "RACE"]
    )

    assert len(planned) == 2
    assert planned[0].request_kind == "new_samples"
    assert planned[0].samples == ("cps2007_09s",)
    assert planned[0].variables == ("AGE", "RACE")
    assert planned[1].request_kind == "variable_delta"
    assert planned[1].samples == ("cps2006_09s",)
    assert planned[1].variables == ("RACE",)


def test_plan_delta_requests_diffs_requested_not_delivered() -> None:
    # QINCWAGE and YEAR are in the files but were never requested. Planning
    # must ignore them: a caller cannot ask for a flag column by name, and a
    # delivered column is not evidence that a *requested* variable is covered.
    coverage = _coverage_with_one_sample(
        {"INCWAGE"}, delivered={"YEAR", "INCWAGE", "QINCWAGE"}
    )

    planned = plan_delta_requests(coverage, ["cps2006_09s"], ["INCWAGE", "AGE"])

    assert len(planned) == 1
    assert planned[0].request_kind == "variable_delta"
    assert planned[0].variables == ("AGE",)


def test_plan_force_requests_splits_known_from_new_samples() -> None:
    coverage = _coverage_with_one_sample({"AGE", "SEX"})

    planned = plan_force_requests(
        coverage, ["cps2006_09s", "cps2007_09s"], ["AGE", "SEX"]
    )

    # Both halves carry the full variable list - force re-pulls everything -
    # but the split still tells the parse stage which half merges onto columns
    # already in bronze rather than overwriting them.
    assert [(p.samples, p.request_kind) for p in planned] == [
        (("cps2007_09s",), "new_samples"),
        (("cps2006_09s",), "variable_delta"),
    ]
    assert all(p.variables == ("AGE", "SEX") for p in planned)


def test_plan_force_requests_omits_an_empty_half() -> None:
    coverage = _coverage_with_one_sample({"AGE"})

    planned = plan_force_requests(coverage, ["cps2006_09s"], ["AGE", "SEX"])

    assert [p.request_kind for p in planned] == ["variable_delta"]
    # Unlike plan_delta_requests, force does not diff - SEX being missing from
    # coverage changes nothing about what is pulled.
    assert planned[0].variables == ("AGE", "SEX")


def test_build_coverage_skips_an_entry_whose_samples_are_not_a_list(
    tmp_path: Path,
) -> None:
    """`samples: cps2007_09s` - a bare string where a list belongs - would be
    iterated one character at a time into a "sample" per letter.
    """
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(collection_dir, "cps_00001", ["cps2006_09s"], ["AGE"])
    _add_manifest_entry(collection_dir, "cps_00002", ["cps2007_09s"], ["AGE"])
    manifest = collection_dir / "_MANIFEST.yaml"
    entries = yaml.safe_load(manifest.read_text())
    entries[1]["metadata"]["samples"] = "cps2007_09s"
    manifest.write_text(yaml.safe_dump(entries, sort_keys=False))

    with structlog.testing.capture_logs() as logs:
        coverage = build_coverage(collection_dir, "cps")

    assert set(coverage.samples) == {"cps2006_09s"}
    skipped = [
        entry for entry in logs if entry["event"] == "ipums_manifest_entry_skipped"
    ]
    assert [entry["reason"] for entry in skipped] == [
        "samples_or_variables_not_a_list_of_names"
    ]


def _append_raw_entry(collection_dir: Path, entry: dict) -> None:
    """Append a hand-built entry, bypassing build_extraction_record.

    A truncated or hand-edited _MANIFEST.yaml is the case build_coverage has to
    survive, and the record builder cannot produce those shapes.
    """
    manifest = collection_dir / "_MANIFEST.yaml"
    entries = yaml.safe_load(manifest.read_text()) if manifest.exists() else []
    manifest.write_text(yaml.safe_dump((entries or []) + [entry], sort_keys=False))


@pytest.mark.parametrize(
    ("bad_entry", "reason"),
    [
        pytest.param(
            {"extraction_id": "cps_00002", "metadata": "not-a-mapping"},
            "metadata_not_a_mapping",
            id="scalar_metadata",
        ),
        pytest.param(
            {"extraction_id": "cps_00002", "metadata": None},
            "metadata_not_a_mapping",
            id="empty_metadata",
        ),
        pytest.param(
            "just-a-string",
            "metadata_not_a_mapping",
            id="non_dict_entry",
        ),
        pytest.param(
            {"extraction_id": "cps_00002", "metadata": {"samples": ["cps2007_09s"]}},
            "missing_metadata_keys",
            id="partial_metadata",
        ),
        pytest.param(
            # Path("") is Path("."), which exists - so without an explicit guard
            # this shape reaches entry["extraction_id"] and raises KeyError.
            {
                "extraction_id": "cps_00002",
                "metadata": {
                    "samples": ["cps2007_09s"],
                    "variables": ["AGE"],
                    "ddi_path": "/nonexistent/cps_00002.xml",
                },
            },
            "missing_entry_keys",
            id="no_file_path",
        ),
        pytest.param(
            {
                "file_path": "/nonexistent/cps_00002.dat.gz",
                "metadata": {
                    "samples": ["cps2007_09s"],
                    "variables": ["AGE"],
                    "ddi_path": "/nonexistent/cps_00002.xml",
                },
            },
            "missing_entry_keys",
            id="no_extraction_id",
        ),
    ],
)
def test_build_coverage_skips_a_malformed_entry(
    tmp_path: Path, bad_entry, reason: str
) -> None:
    """A malformed entry is skipped with a warning, and - the point - the sound
    entries around it still produce coverage. Asserting only `samples == {}`
    would pass just as well if the whole manifest had been dropped.
    """
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    _add_manifest_entry(collection_dir, "cps_00001", ["cps2006_09s"], ["AGE", "SEX"])
    _append_raw_entry(collection_dir, bad_entry)

    with structlog.testing.capture_logs() as logs:
        coverage = build_coverage(collection_dir, "cps")

    assert set(coverage.samples) == {"cps2006_09s"}
    assert coverage.samples["cps2006_09s"].requested_variables == frozenset(
        {"AGE", "SEX"}
    )
    skipped = [entry for entry in logs if entry["event"] == "manifest_entry_skipped"]
    assert [entry["reason"] for entry in skipped] == [reason]
