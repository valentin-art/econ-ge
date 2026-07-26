import zipfile
from pathlib import Path

import pytest

from src.extractors.base import ExtractionRecord
from src.pipelines.cps_pipeline import run_cps_basic_pipeline, run_cpsmw_pipeline

_MW_SPS_TEXT = """\
data list file='c:\\cpsmw64.raw' /
            hhid       1-5
            state      6-7         (a)
            age        8-9
.
"""

# hhid=12345, state="CA", age=34
_DAT_TEXT = "12345CA34\n12346TX25\n"


def _write_zip_and_sps(
    dir_path: Path, zip_filename: str, member_name: str, sps_filename: str
) -> tuple[Path, Path]:
    dir_path.mkdir(parents=True, exist_ok=True)
    zip_path = dir_path / zip_filename
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(member_name, _DAT_TEXT)
    sps_path = dir_path / sps_filename
    sps_path.write_text(_MW_SPS_TEXT)
    return zip_path, sps_path


def test_run_cpsmw_pipeline_extracts_parses_and_concatenates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_dir = tmp_path / "external"
    data_root = tmp_path / "data"
    monkeypatch.setattr("src.pipelines.cps_pipeline.settings.paths.root", data_root)

    def fake_extract(self, year: int, month: int | None = None) -> ExtractionRecord:
        zip_filename = f"cpsmw{year % 100:02d}.zip"
        zip_path, sps_path = _write_zip_and_sps(
            external_dir, zip_filename, f"cpsmw{year % 100:02d}", "cpsmw64_88.sps"
        )
        return ExtractionRecord(
            source="mw",
            extraction_id=f"mw_{year}",
            extracted_at="2026-01-01T00:00:00+00:00",  # type: ignore[arg-type]
            file_path=zip_path,
            size_bytes=zip_path.stat().st_size,
            sha256="deadbeef",
            metadata={"sps_path": str(sps_path)},
        )

    monkeypatch.setattr("src.extractors.cps.CPSMWExtractor.extract", fake_extract)

    result = run_cpsmw_pipeline(years=[1964])

    assert list(result.frame.columns) == ["Year", "hhid", "state", "age"]
    assert result.frame["hhid"].tolist() == [12345, 12346]
    assert (data_root / "bronze" / "cps" / "mw" / "1964.parquet").exists()
    assert (data_root / "reference" / "cps" / "mw" / "1964.json").exists()


def test_run_cpsmw_pipeline_concatenates_multiple_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_dir = tmp_path / "external"
    data_root = tmp_path / "data"
    monkeypatch.setattr("src.pipelines.cps_pipeline.settings.paths.root", data_root)

    def fake_extract(self, year: int, month: int | None = None) -> ExtractionRecord:
        zip_filename = f"cpsmw{year % 100:02d}.zip"
        zip_path, sps_path = _write_zip_and_sps(
            external_dir / str(year),
            zip_filename,
            f"cpsmw{year % 100:02d}",
            "cpsmw64_88.sps",
        )
        return ExtractionRecord(
            source="mw",
            extraction_id=f"mw_{year}",
            extracted_at="2026-01-01T00:00:00+00:00",  # type: ignore[arg-type]
            file_path=zip_path,
            size_bytes=zip_path.stat().st_size,
            sha256="deadbeef",
            metadata={"sps_path": str(sps_path)},
        )

    monkeypatch.setattr("src.extractors.cps.CPSMWExtractor.extract", fake_extract)

    result = run_cpsmw_pipeline(years=[1964, 1965])

    assert sorted(result.frame["Year"].unique().tolist()) == [1964, 1965]
    assert len(result.frame) == 4


def test_run_cps_basic_pipeline_extracts_parses_and_concatenates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_dir = tmp_path / "external"
    data_root = tmp_path / "data"
    monkeypatch.setattr("src.pipelines.cps_pipeline.settings.paths.root", data_root)

    def fake_extract(self, year: int, month: int | None = None) -> ExtractionRecord:
        assert month is not None
        zip_filename = f"cpsb{year}{month:02d}_dat.zip"
        zip_path, sps_path = _write_zip_and_sps(
            external_dir, zip_filename, "cpsb199102", "cpsb198901.sps"
        )
        return ExtractionRecord(
            source="basic",
            extraction_id=f"basic_{year}_{month}",
            extracted_at="2026-01-01T00:00:00+00:00",  # type: ignore[arg-type]
            file_path=zip_path,
            size_bytes=zip_path.stat().st_size,
            sha256="deadbeef",
            metadata={"sps_path": str(sps_path)},
        )

    monkeypatch.setattr("src.extractors.cps.CPSBasicExtractor.extract", fake_extract)

    result = run_cps_basic_pipeline(periods=[(1991, 2)])

    assert list(result.frame.columns) == ["Year", "hhid", "state", "age"]
    assert (data_root / "bronze" / "cps" / "basic" / "1991" / "199102.parquet").exists()
    assert (data_root / "reference" / "cps" / "basic" / "199102.json").exists()
