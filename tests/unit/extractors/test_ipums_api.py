"""Tests for IPUMSExtractor.

Deliberately excludes a live-API test from the collected suite: submitting an
extract against the real IPUMS API adds one more entry to the user's account
history/extract-request quota every time it runs, so it must never execute
automatically. A real test is written out at the bottom of this file but kept
commented out - uncomment it and run manually (with a real IPUMS_API_KEY
configured) to confirm live connectivity beyond what the tests below cover.
"""

from pathlib import Path

from src.extractors.ipums_api import IPUMSExtractor
from src.extractors.manifest import read_manifest

_FAKE_API_KEY = "fake-key"  # pragma: allowlist secret


class _ClientMustNotBeCalled:
    """Fails the test if the extractor tries to hit the IPUMS API at all."""

    def submit_extract(self, *args, **kwargs):
        raise AssertionError(
            "submit_extract should not be called when a matching extract exists"
        )

    def wait_for_extract(self, *args, **kwargs):
        raise AssertionError(
            "wait_for_extract should not be called when a matching extract exists"
        )

    def download_extract(self, *args, **kwargs):
        raise AssertionError(
            "download_extract should not be called when a matching extract exists"
        )


class _SequentialFakeClient:
    """Simulates completed extracts with an auto-incrementing extract_id,
    without hitting the network - supports multiple submissions in one test.
    """

    def __init__(self, start_id: int = 100) -> None:
        self._next_id = start_id
        self.submit_calls = 0

    def submit_extract(self, extract) -> None:
        self.submit_calls += 1
        extract._id = self._next_id
        self._next_id += 1

    def wait_for_extract(self, extract) -> None:
        pass

    def download_extract(self, extract, download_dir: Path) -> None:
        collection = extract.collection
        extract_id = extract.extract_id
        (download_dir / f"{collection}_{extract_id:05d}.dat.gz").write_bytes(b"data")
        (download_dir / f"{collection}_{extract_id:05d}.xml").write_text("<codeBook/>")


def _seed_manifest_entry(
    extractor: IPUMSExtractor,
    collection: str,
    samples: list[str],
    variables: list[str],
) -> None:
    """Populate collection_dir/_MANIFEST.yaml with one real, matchable entry
    by running a real extract() call through whatever (faked) client
    `extractor` already has - avoids hand-building manifest YAML that could
    drift from the real schema.
    """
    extractor.extract(collection=collection, samples=samples, variables=variables)


def test_extract_reuses_matching_manifest_entry_without_hitting_api(
    tmp_path: Path,
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    extractor.client = _ClientMustNotBeCalled()

    # Subset of the seeded variables - still counts as fully covered.
    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert record.metadata["cached"] is True
    assert record.metadata["extract_id"] == 1

    # append_to_manifest still runs on a cache hit (audit trail of every call,
    # not just real submissions) - both entries point at the same extract_id.
    manifest_entries = read_manifest(tmp_path / "cps")
    assert len(manifest_entries) == 2
    assert all(e["metadata"]["extract_id"] == 1 for e in manifest_entries)


def test_extract_falls_through_when_variables_not_a_subset(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE", "SEX"]
    )

    assert record.metadata["cached"] is False
    assert record.metadata["extract_id"] == 2


def test_extract_requires_exact_samples_match(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(
        extractor, "cps", ["cps2006_09s", "cps2007_09s"], ["AGE", "SEX"]
    )

    # A strict subset of a covered sample list doesn't match - the entry
    # covers *both* samples together, not either one individually.
    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert record.metadata["cached"] is False


def test_extract_ignores_manifest_entry_whose_files_were_deleted(
    tmp_path: Path,
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])
    (tmp_path / "cps" / "cps_00001.dat.gz").unlink()

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert record.metadata["cached"] is False
    assert record.metadata["extract_id"] == 2


def test_extract_force_resubmits_even_when_a_matching_entry_exists(
    tmp_path: Path,
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])

    client = _SequentialFakeClient(start_id=99)
    extractor.client = client
    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"], force=True
    )

    assert client.submit_calls == 1
    assert record.metadata["extract_id"] == 99
    assert record.metadata["cached"] is False
    # the original matching extract is untouched, not deleted
    assert (tmp_path / "cps" / "cps_00001.dat.gz").exists()


def test_extract_records_request_kind_in_metadata(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )

    default_record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )
    delta_record = extractor.extract(
        collection="cps",
        samples=["cps2007_09s"],
        variables=["RACE"],
        request_kind="variable_delta",
    )

    assert default_record.metadata["request_kind"] == "new_samples"
    assert delta_record.metadata["request_kind"] == "variable_delta"


def test_extract_incremental_returns_empty_when_already_covered(
    tmp_path: Path,
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    extractor.client = _ClientMustNotBeCalled()

    records = extractor.extract_incremental(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert records == []


def test_extract_incremental_submits_new_samples_extract(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    client = _SequentialFakeClient(start_id=50)
    extractor.client = client

    records = extractor.extract_incremental(
        collection="cps",
        samples=["cps2006_09s", "cps2007_09s"],
        variables=["AGE", "SEX"],
    )

    assert client.submit_calls == 1
    assert len(records) == 1
    assert records[0].metadata["samples"] == ("cps2007_09s",)
    assert records[0].metadata["variables"] == ("AGE", "SEX")
    assert records[0].metadata["request_kind"] == "new_samples"
    assert (tmp_path / "cps" / "_COVERAGE.yaml").exists()


def test_extract_incremental_submits_variable_delta_extract(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    client = _SequentialFakeClient(start_id=50)
    extractor.client = client

    records = extractor.extract_incremental(
        collection="cps", samples=["cps2006_09s"], variables=["AGE", "SEX", "RACE"]
    )

    assert client.submit_calls == 1
    assert len(records) == 1
    assert records[0].metadata["samples"] == ("cps2006_09s",)
    assert records[0].metadata["variables"] == ("RACE",)
    assert records[0].metadata["request_kind"] == "variable_delta"


def test_extract_incremental_submits_both_when_years_and_variables_differ(
    tmp_path: Path,
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    client = _SequentialFakeClient(start_id=50)
    extractor.client = client

    records = extractor.extract_incremental(
        collection="cps",
        samples=["cps2006_09s", "cps2007_09s"],
        variables=["AGE", "SEX", "RACE"],
    )

    assert client.submit_calls == 2
    assert len(records) == 2
    new_samples_records = [
        r for r in records if r.metadata["request_kind"] == "new_samples"
    ]
    delta_records = [
        r for r in records if r.metadata["request_kind"] == "variable_delta"
    ]
    assert new_samples_records[0].metadata["samples"] == ("cps2007_09s",)
    assert new_samples_records[0].metadata["variables"] == ("AGE", "RACE", "SEX")
    assert delta_records[0].metadata["samples"] == ("cps2006_09s",)
    assert delta_records[0].metadata["variables"] == ("RACE",)


def test_extract_incremental_force_bypasses_planning(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    client = _SequentialFakeClient(start_id=50)
    extractor.client = client

    records = extractor.extract_incremental(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"], force=True
    )

    assert client.submit_calls == 1
    assert len(records) == 1
    assert records[0].metadata["samples"] == ("cps2006_09s",)
    assert records[0].metadata["variables"] == ("AGE",)
    assert records[0].metadata["force"] is True
    # cps2006_09s is already a known sample - forcing it is a re-pull of
    # existing coverage, not a brand-new sample, so it's labeled accordingly.
    assert records[0].metadata["request_kind"] == "variable_delta"


def test_extract_incremental_force_labels_unknown_sample_as_new_samples(
    tmp_path: Path,
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    client = _SequentialFakeClient(start_id=50)
    extractor.client = client

    records = extractor.extract_incremental(
        collection="cps", samples=["cps2007_09s"], variables=["AGE"], force=True
    )

    assert records[0].metadata["request_kind"] == "new_samples"
    assert records[0].metadata["force"] is True


def test_extract_incremental_force_splits_mixed_known_and_new_samples(
    tmp_path: Path,
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    client = _SequentialFakeClient(start_id=50)
    extractor.client = client

    # cps2006_09s is already known, cps2007_09s is not - a forced pull
    # spanning both must not collapse to a single "new_samples" extract, or
    # the parse stage would overwrite (not merge) 2006's existing columns.
    records = extractor.extract_incremental(
        collection="cps",
        samples=["cps2006_09s", "cps2007_09s"],
        variables=["AGE"],
        force=True,
    )

    assert client.submit_calls == 2
    assert len(records) == 2
    by_kind = {r.metadata["request_kind"]: r for r in records}
    assert set(by_kind) == {"new_samples", "variable_delta"}
    assert by_kind["new_samples"].metadata["samples"] == ("cps2007_09s",)
    assert by_kind["variable_delta"].metadata["samples"] == ("cps2006_09s",)
    assert all(r.metadata["force"] is True for r in records)
    assert (tmp_path / "cps" / "_COVERAGE.yaml").exists()


def test_extract_records_force_in_metadata(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )

    default_record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )
    forced_record = extractor.extract(
        collection="cps",
        samples=["cps2007_09s"],
        variables=["AGE"],
        force=True,
    )

    assert default_record.metadata["force"] is False
    assert forced_record.metadata["force"] is True


# --- Real-API test: commented out on purpose, see module docstring. ---
#
# def test_extract_hits_real_ipums_api(tmp_path: Path) -> None:
#     extractor = IPUMSExtractor(storage_dir=tmp_path)
#     record = extractor.extract(
#         collection="cps",
#         samples=["cps2006_09s"],
#         variables=["AGE"],
#         description="econ-ge extractor smoke test",
#     )
#     assert record.file_path.exists()
