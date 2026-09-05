from pathlib import Path

import pytest

from extractors.cps.cps import CPSMWExtractor
from src.extractors.manifest import read_manifest


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        pass


def test_extract_writes_zip_dictionaries_sps_and_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_urls = []

    def fake_get(url, headers: dict[str, str] | None = None, timeout=None):
        requested_urls.append(url)
        assert headers is not None and "User-Agent" in headers
        if url.endswith(".zip"):
            return _FakeResponse(b"fake-zip-bytes")
        return _FakeResponse(b"fake-sps-bytes")

    monkeypatch.setattr("src.extractors.cps.cps.requests.get", fake_get)
    extractor = CPSMWExtractor(
        base_url="https://data.nber.org/mare_winship", storage_dir=tmp_path
    )

    record = extractor.extract(year=1964)

    assert record.file_path == tmp_path / "cpsmw64.zip"
    assert record.file_path.exists()
    assert record.file_path.read_bytes() == b"fake-zip-bytes"

    sps_path = tmp_path / "dictionaries" / "cpsmw64_88.sps"
    assert sps_path.exists()
    assert sps_path.read_bytes() == b"fake-sps-bytes"

    assert record.metadata == {
        "year": 1964,
        "zip_filename": "cpsmw64.zip",
        "sps_filename": "cpsmw64_88.sps",
        "zip_url": "https://data.nber.org/mare_winship/cpsmw64.zip",
        "sps_path": str(sps_path),
        "cached": False,
    }
    assert requested_urls == [
        "https://data.nber.org/mare_winship/cpsmw64.zip",
        "https://data.nber.org/mare_winship/cpsmw64_88.sps",
    ]

    manifest_entries = read_manifest(tmp_path)
    assert len(manifest_entries) == 1
    assert manifest_entries[0]["extraction_id"] == record.extraction_id


def test_extract_skips_sps_download_if_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dictionaries_dir = tmp_path / "dictionaries"
    dictionaries_dir.mkdir(parents=True, exist_ok=True)
    sps_path = dictionaries_dir / "cpsmw64_88.sps"
    sps_path.write_bytes(b"already-here")

    requested_urls = []

    def fake_get(url, headers=None, timeout=None):
        requested_urls.append(url)
        return _FakeResponse(b"fake-zip-bytes")

    monkeypatch.setattr("src.extractors.cps.cps.requests.get", fake_get)
    extractor = CPSMWExtractor(
        base_url="https://data.nber.org/mare_winship", storage_dir=tmp_path
    )

    extractor.extract(year=1964)

    assert requested_urls == ["https://data.nber.org/mare_winship/cpsmw64.zip"]
    assert sps_path.read_bytes() == b"already-here"


def test_extract_skips_zip_download_if_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_path / "cpsmw64.zip"
    zip_path.write_bytes(b"already-here")

    requested_urls = []

    def fake_get(url, headers=None, timeout=None):
        requested_urls.append(url)
        return _FakeResponse(b"fake-sps-bytes")

    monkeypatch.setattr("src.extractors.cps.cps.requests.get", fake_get)
    extractor = CPSMWExtractor(
        base_url="https://data.nber.org/mare_winship", storage_dir=tmp_path
    )

    record = extractor.extract(year=1964)

    assert requested_urls == ["https://data.nber.org/mare_winship/cpsmw64_88.sps"]
    assert zip_path.read_bytes() == b"already-here"
    assert record.metadata["cached"] is True
    assert record.metadata["zip_url"] is None


def test_extract_skips_both_downloads_if_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "cpsmw64.zip").write_bytes(b"already-here-zip")
    dictionaries_dir = tmp_path / "dictionaries"
    dictionaries_dir.mkdir(parents=True, exist_ok=True)
    (dictionaries_dir / "cpsmw64_88.sps").write_bytes(b"already-here-sps")

    def fake_get(url, headers=None, timeout=None):
        raise AssertionError(f"should not hit the network, but got {url}")

    monkeypatch.setattr("src.extractors.cps.cps.requests.get", fake_get)
    extractor = CPSMWExtractor(
        base_url="https://data.nber.org/mare_winship", storage_dir=tmp_path
    )

    record = extractor.extract(year=1964)

    assert record.metadata["cached"] is True


def test_download_retries_on_403_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.extractors.cps.cps.time.sleep", lambda _seconds: None)
    responses = iter(
        [_FakeResponse(b"", status_code=403), _FakeResponse(b"fake-zip-bytes")]
    )
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return next(responses)

    monkeypatch.setattr("src.extractors.cps.cps.requests.get", fake_get)
    extractor = CPSMWExtractor(
        base_url="https://data.nber.org/mare_winship", storage_dir=tmp_path
    )

    content, url = extractor._download("cpsmw64.zip")

    assert content == b"fake-zip-bytes"
    assert len(calls) == 2


def test_download_raises_after_exhausting_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.extractors.cps.cps.time.sleep", lambda _seconds: None)

    def fake_get(url, headers=None, timeout=None):
        response = _FakeResponse(b"", status_code=403)

        def _raise() -> None:
            raise RuntimeError("403 Forbidden")

        response.raise_for_status = _raise  # type: ignore[method-assign]
        return response

    monkeypatch.setattr("src.extractors.cps.cps.requests.get", fake_get)
    extractor = CPSMWExtractor(
        base_url="https://data.nber.org/mare_winship", storage_dir=tmp_path
    )

    with pytest.raises(RuntimeError, match="403 Forbidden"):
        extractor._download("cpsmw64.zip")


def test_cps_basic_extractor_requires_month(tmp_path: Path) -> None:
    from extractors.cps.cps import CPSBasicExtractor

    extractor = CPSBasicExtractor(
        base_url="https://data.nber.org/cps-basic3/dat/",
        storage_dir=tmp_path,
        sps_base_url="https://data.nber.org/cps-basic3/programs/",
    )

    with pytest.raises(ValueError, match="requires month"):
        extractor.extract(year=1991)


def test_cps_basic_extractor_writes_zip_and_sps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from extractors.cps.cps import CPSBasicExtractor

    requested_urls = []

    def fake_get(url, headers=None, timeout=None):
        requested_urls.append(url)
        if url.endswith(".zip"):
            return _FakeResponse(b"fake-zip-bytes")
        return _FakeResponse(b"fake-sps-bytes")

    monkeypatch.setattr("src.extractors.cps.cps.requests.get", fake_get)
    extractor = CPSBasicExtractor(
        base_url="https://data.nber.org/cps-basic3/dat/",
        storage_dir=tmp_path,
        sps_base_url="https://data.nber.org/cps-basic3/programs/",
    )

    record = extractor.extract(year=1991, month=2)

    zip_path = tmp_path / "cpsb199102_dat.zip"
    assert record.file_path == zip_path
    assert zip_path.read_bytes() == b"fake-zip-bytes"
    assert record.metadata["month"] == 2
    assert requested_urls == [
        "https://data.nber.org/cps-basic3/dat/1991/cpsb199102_dat.zip",
        "https://data.nber.org/cps-basic3/programs/cpsb198901.sps",
    ]
