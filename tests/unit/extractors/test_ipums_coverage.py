from pathlib import Path

import yaml

from src.extractors.base import build_extraction_record
from src.extractors.ipums_coverage import (
    CollectionCoverage,
    SampleCoverage,
    build_coverage,
    parse_sample_year,
    plan_delta_requests,
    save_coverage,
)
from src.extractors.manifest import append_to_manifest


def _add_manifest_entry(
    collection_dir: Path,
    extraction_id: str,
    samples: list[str],
    variables: list[str],
    make_files: bool = True,
) -> None:
    data_path = collection_dir / f"{extraction_id}.dat.gz"
    ddi_path = collection_dir / f"{extraction_id}.xml"
    # build_extraction_record reads the file to checksum it, so it must exist
    # at record-build time regardless; make_files=False removes it afterward
    # to simulate a manifest entry whose files were later deleted.
    data_path.write_bytes(b"data")
    ddi_path.write_text("<codeBook/>")
    metadata = {
        "collection": "cps",
        "samples": samples,
        "variables": variables,
        "extract_id": int(extraction_id.rsplit("_", 1)[-1]),
        "ddi_path": str(ddi_path),
        "cached": False,
        "request_kind": "new_samples",
    }
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
    assert coverage.samples["cps2006_09s"].variables == frozenset({"AGE", "SEX"})
    assert coverage.samples["cps2025_03s"].extraction_ids == ("cps_00030",)
    assert coverage.variables == frozenset({"AGE", "SEX", "YEAR", "RACE"})
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


def test_save_coverage_writes_summary(tmp_path: Path) -> None:
    coverage = CollectionCoverage(
        collection="cps",
        samples={
            "cps2006_09s": SampleCoverage(
                variables=frozenset({"AGE", "SEX"}), extraction_ids=("cps_00029",)
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


def _coverage_with_one_sample(variables: set[str]) -> CollectionCoverage:
    return CollectionCoverage(
        collection="cps",
        samples={
            "cps2006_09s": SampleCoverage(
                variables=frozenset(variables), extraction_ids=("cps_00029",)
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
    assert planned[0].samples == ["cps2007_09s"]
    assert planned[0].variables == ["AGE", "SEX"]


def test_plan_delta_requests_variable_delta_only() -> None:
    coverage = _coverage_with_one_sample({"AGE", "SEX"})

    planned = plan_delta_requests(coverage, ["cps2006_09s"], ["AGE", "RACE"])

    assert len(planned) == 1
    assert planned[0].request_kind == "variable_delta"
    assert planned[0].samples == ["cps2006_09s"]
    assert planned[0].variables == ["RACE"]


def test_plan_delta_requests_both_new_samples_and_variable_delta() -> None:
    coverage = _coverage_with_one_sample({"AGE", "SEX"})

    planned = plan_delta_requests(
        coverage, ["cps2006_09s", "cps2007_09s"], ["AGE", "RACE"]
    )

    assert len(planned) == 2
    assert planned[0].request_kind == "new_samples"
    assert planned[0].samples == ["cps2007_09s"]
    assert planned[0].variables == ["AGE", "RACE"]
    assert planned[1].request_kind == "variable_delta"
    assert planned[1].samples == ["cps2006_09s"]
    assert planned[1].variables == ["RACE"]
