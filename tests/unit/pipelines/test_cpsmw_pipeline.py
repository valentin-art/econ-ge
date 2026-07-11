import zipfile
from pathlib import Path

import pytest

from src.extractors.base import ExtractionRecord
from src.pipelines.cpsmw_pipeline import run_cpsmw_pipeline

_SPS_TEXT = """\
data list file='c:\\cpsmw64.raw' /
            hhid       1-5
            state      6-7         (a)
            age        8-9
.
"""

# hhid=12345, state="CA", age=34
_DAT_TEXT = "12345CA34\n12346TX25\n"


def _write_zip_and_sps(dir_path: Path, year: int) -> tuple[Path, Path]:
    dir_path.mkdir(parents=True, exist_ok=True)
    zip_path = dir_path / f"cpsmw{year % 100:02d}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"cpsmw{year % 100:02d}", _DAT_TEXT)
    sps_path = dir_path / "cpsmw64_88.sps"
    sps_path.write_text(_SPS_TEXT)
    return zip_path, sps_path


def test_run_cpsmw_pipeline_extracts_parses_and_concatenates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_dir = tmp_path / "external"
    data_root = tmp_path / "data"
    monkeypatch.setattr("src.pipelines.cpsmw_pipeline.settings.paths.root", data_root)
    # build_and_save_variable_dictionary's default dictionaries_dir points at
    # the real, committed src/parsers/dictionaries/cpsmw/ — stub it out so
    # this test can't overwrite the real generated dictionaries with fixture
    # data.
    dictionary_calls = []
    monkeypatch.setattr(
        "src.pipelines.cpsmw_pipeline.build_and_save_variable_dictionary",
        lambda sps_path, year: dictionary_calls.append((sps_path, year)),
    )

    def fake_extract(self, year: int) -> ExtractionRecord:
        zip_path, sps_path = _write_zip_and_sps(external_dir, year)
        return ExtractionRecord(
            source="cps_mw",
            extraction_id=f"cps_mw_{year}",
            extracted_at="2026-01-01T00:00:00+00:00",  # type: ignore[arg-type]
            file_path=zip_path,
            size_bytes=zip_path.stat().st_size,
            sha256="deadbeef",
            metadata={"sps_path": str(sps_path)},
        )

    monkeypatch.setattr("src.extractors.cps_mw.CPSMWExtractor.extract", fake_extract)

    result = run_cpsmw_pipeline(years=[1964])

    assert list(result.cps_mw.columns) == ["Year", "hhid", "state", "age"]
    assert result.cps_mw["hhid"].tolist() == [12345, 12346]
    assert (data_root / "bronze" / "cps" / "mw" / "1964.parquet").exists()
    assert len(dictionary_calls) == 1
    assert dictionary_calls[0][1] == 1964


def test_run_cpsmw_pipeline_concatenates_multiple_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_dir = tmp_path / "external"
    data_root = tmp_path / "data"
    monkeypatch.setattr("src.pipelines.cpsmw_pipeline.settings.paths.root", data_root)
    monkeypatch.setattr(
        "src.pipelines.cpsmw_pipeline.build_and_save_variable_dictionary",
        lambda sps_path, year: None,
    )

    def fake_extract(self, year: int) -> ExtractionRecord:
        zip_path, sps_path = _write_zip_and_sps(external_dir / str(year), year)
        return ExtractionRecord(
            source="cps_mw",
            extraction_id=f"cps_mw_{year}",
            extracted_at="2026-01-01T00:00:00+00:00",  # type: ignore[arg-type]
            file_path=zip_path,
            size_bytes=zip_path.stat().st_size,
            sha256="deadbeef",
            metadata={"sps_path": str(sps_path)},
        )

    monkeypatch.setattr("src.extractors.cps_mw.CPSMWExtractor.extract", fake_extract)

    result = run_cpsmw_pipeline(years=[1964, 1965])

    assert sorted(result.cps_mw["Year"].unique().tolist()) == [1964, 1965]
    assert len(result.cps_mw) == 4
