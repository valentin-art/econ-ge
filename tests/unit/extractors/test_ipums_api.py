"""Tests for IPUMSExtractor.

Deliberately excludes a live-API test from the collected suite: submitting an
extract against the real IPUMS API adds one more entry to the user's account
history/extract-request quota every time it runs, so it must never execute
automatically. A real test is written out at the bottom of this file but kept
commented out - uncomment it and run manually (with a real IPUMS_API_KEY
configured) to confirm live connectivity beyond what the tests below cover.
"""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import structlog.testing
import yaml
from ipumspy.api.exceptions import BadIpumsApiRequest

from extractors.ipums.ipums_api import (
    IPUMSExtractor,
    _default_data_structure,
    find_matching_extract,
)
from extractors.ipums.ipums_ddi import FLAG_PARSER_VERSION, summary_from_metadata
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

    Writes a real DDI codebook describing what IPUMS would actually deliver:
    the requested variables, a flag column for each variable in `flags`, and
    the technical columns IPUMS preselects. That superset is the whole point -
    the extractor records delivered columns, and a `<codeBook/>` stub could
    never exercise it.
    """

    def __init__(
        self,
        start_id: int = 100,
        flags: dict[str, list[str]] | None = None,
        technical: tuple[str, ...] = ("YEAR", "SERIAL", "PERNUM"),
        make_ddi_xml: Callable[[Sequence[tuple[str, str, int]]], str] | None = None,
    ) -> None:
        self._next_id = start_id
        self.submit_calls = 0
        self.submitted: list[dict] = []
        self.flags = flags or {}
        self.technical = technical
        self._make_ddi_xml = make_ddi_xml

    def submit_extract(self, extract) -> None:
        self.submit_calls += 1
        self.submitted.append(extract.build())
        extract._id = self._next_id
        self._next_id += 1

    def wait_for_extract(self, extract) -> None:
        pass

    def _ddi_xml(self, extract) -> str:
        assert self._make_ddi_xml is not None
        requested = [v.name for v in extract.variables]
        variables = [(name, name.title(), 2) for name in self.technical]
        for name in requested:
            variables.append((name, name.title(), 2))
            for flag in self.flags.get(name, []):
                variables.append((flag, f"Data quality flag for {name}", 1))
        return self._make_ddi_xml(variables)

    def download_extract(self, extract, download_dir: Path) -> None:
        collection = extract.collection
        extract_id = extract.extract_id
        (download_dir / f"{collection}_{extract_id:05d}.dat.gz").write_bytes(b"data")
        ddi_path = download_dir / f"{collection}_{extract_id:05d}.xml"
        if self._make_ddi_xml is None:
            # Deliberately unreadable: exercises the path where a codebook
            # cannot be summarized and the entry is recorded without
            # delivered_variables.
            ddi_path.write_text("<codeBook/>")
        else:
            ddi_path.write_text(self._ddi_xml(extract), encoding="utf-8")


class _BadRequestFakeClient:
    """Rejects every submission the way the IPUMS API rejects an unknown
    variable name: HTTP 400 -> BadIpumsApiRequest carrying the server's text.
    """

    # Verbatim from the live API on 2026-08-16 for
    # variables=["QOINCWAGE"], samples=["cps1989_03s"].
    detail = "Invalid mnemonic: QOINCWAGE"

    def submit_extract(self, extract) -> None:
        raise BadIpumsApiRequest(self.detail)

    def wait_for_extract(self, extract) -> None:
        raise AssertionError("wait_for_extract should not run after a failed submit")

    def download_extract(self, extract, download_dir: Path) -> None:
        raise AssertionError("download_extract should not run after a failed submit")


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

    # A cache hit appends nothing: nothing was downloaded, so there is no new
    # file to record, and a second entry would only re-point at the same
    # extract_id while growing a file that append_to_manifest rewrites whole.
    manifest_entries = read_manifest(tmp_path / "cps")
    assert len(manifest_entries) == 1
    assert manifest_entries[0]["metadata"]["extract_id"] == 1


def _corrupt_manifest_checksum(collection_dir: Path, **fields: object) -> None:
    """Rewrite the sole manifest entry's size/checksum fields in place."""
    manifest_path = collection_dir / "_MANIFEST.yaml"
    entries = yaml.safe_load(manifest_path.read_text())
    entries[0].update(fields)
    manifest_path.write_text(yaml.safe_dump(entries, sort_keys=False))


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param({"size_bytes": "not-a-number"}, id="unparseable_size"),
        pytest.param({"sha256": None}, id="null_checksum"),
        pytest.param({"size_bytes": None, "sha256": None}, id="both_null"),
        # A checksum with no size is unusable on its own: nothing has been
        # checked against the file, so the recorded hash is not evidence.
        pytest.param({"size_bytes": None}, id="null_size_valid_checksum"),
    ],
)
def test_extract_recomputes_checksum_rather_than_resubmitting(
    tmp_path: Path, corruption: dict
) -> None:
    # An entry that matches on everything but carries no usable checksum is
    # still a cache hit: re-hashing the local file beats spending an IPUMS
    # extract on a re-download of a file already on disk.
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])
    _corrupt_manifest_checksum(tmp_path / "cps", **corruption)
    extractor.client = _ClientMustNotBeCalled()

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert record.metadata["cached"] is True
    # Recomputed from the file on disk, not carried over from the entry.
    data_path = tmp_path / "cps" / "cps_00001.dat.gz"
    assert record.size_bytes == data_path.stat().st_size
    assert record.sha256 == hashlib.sha256(data_path.read_bytes()).hexdigest()
    # Still a cache hit, so still no new manifest entry.
    assert len(read_manifest(tmp_path / "cps")) == 1


def test_extract_resubmits_when_the_extract_id_itself_is_unusable(
    tmp_path: Path,
) -> None:
    # extract_id is not recoverable from the file - it names the download and
    # the extraction_id - so an unusable one is still a hard skip.
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])
    manifest_path = tmp_path / "cps" / "_MANIFEST.yaml"
    entries = yaml.safe_load(manifest_path.read_text())
    entries[0]["metadata"]["extract_id"] = "not-a-number"
    manifest_path.write_text(yaml.safe_dump(entries, sort_keys=False))

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert record.metadata["cached"] is False
    assert len(read_manifest(tmp_path / "cps")) == 2


def test_extract_resubmits_when_recorded_size_mismatches_size(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )

    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])

    # Truncate the file
    data_path = tmp_path / "cps" / "cps_00001.dat.gz"
    data_path.write_bytes(b"f")

    # resubmit
    extractor.client = _SequentialFakeClient(start_id=2)
    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert record.metadata["cached"] is False
    assert record.metadata["extract_id"] == 2
    # Both entries survive - the first is not dropped
    assert len(read_manifest(tmp_path / "cps")) == 2


def test_extract_resubmits_when_size_mismatches_and_checksum_is_null(
    tmp_path: Path,
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])
    _corrupt_manifest_checksum(tmp_path / "cps", sha256=None)
    (tmp_path / "cps" / "cps_00001.dat.gz").write_bytes(b"f")

    extractor.client = _SequentialFakeClient(start_id=2)
    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )
    assert record.metadata["cached"] is False


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


# --- Data quality flags -----------------------------------------------------


def test_extract_records_delivered_variables_and_flag_maps(
    tmp_path: Path, make_ddi_xml
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(
            start_id=1, flags={"INCWAGE": ["QINCWAGE"]}, make_ddi_xml=make_ddi_xml
        ),
    )

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["INCWAGE"]
    )

    # What was asked for stays exactly what was asked for...
    assert record.metadata["variables"] == ("INCWAGE",)
    assert "QINCWAGE" in record.metadata["delivered_variables"]
    assert "YEAR" in record.metadata["delivered_variables"]
    assert record.metadata["quality_flags"] == {"INCWAGE": ["QINCWAGE"]}
    # The stamp is what lets collection_flag_registry trust this map instead of
    # re-parsing the codebook - see FLAG_PARSER_VERSION.
    assert record.metadata["flag_parser_version"] == FLAG_PARSER_VERSION
    assert summary_from_metadata(record.metadata) is not None  # write/read round trip


def test_extract_records_delivered_variables_on_cache_hit(
    tmp_path: Path, make_ddi_xml
) -> None:
    # A cache hit re-reads the codebook of the extract it is reusing, so an
    # entry written before these keys existed is backfilled rather than
    # propagating the gap forward.
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(
            start_id=1, flags={"INCWAGE": ["QINCWAGE"]}, make_ddi_xml=make_ddi_xml
        ),
    )
    extractor.extract(collection="cps", samples=["cps2006_09s"], variables=["INCWAGE"])

    extractor.client = _ClientMustNotBeCalled()
    cached_record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["INCWAGE"]
    )

    assert cached_record.metadata["cached"] is True
    assert "QINCWAGE" in cached_record.metadata["delivered_variables"]


def test_extract_tolerates_unparseable_ddi(tmp_path: Path) -> None:
    # A codebook we cannot read must not fail an extract that has already been
    # downloaded and already cost account quota.
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),  # writes a <codeBook/> stub
    )

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert "delivered_variables" not in record.metadata
    assert len(read_manifest(tmp_path / "cps")) == 1


def test_extract_rejects_known_quality_flag_before_submitting(
    tmp_path: Path, make_ddi_xml
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(
            start_id=1, flags={"INCWAGE": ["QINCWAGE"]}, make_ddi_xml=make_ddi_xml
        ),
    )
    extractor.extract(collection="cps", samples=["cps2006_09s"], variables=["INCWAGE"])

    extractor.client = _ClientMustNotBeCalled()
    with pytest.raises(ValueError) as excinfo:
        extractor.extract(
            collection="cps", samples=["cps2007_09s"], variables=["QINCWAGE"]
        )

    message = str(excinfo.value)
    assert "QINCWAGE" in message
    assert "INCWAGE" in message
    assert "data_quality_flags=True" in message
    # the remedy is "request INCWAGE instead", not a bare "drop QINCWAGE" -
    # dropping it alone yields an extract with no flag column at all
    assert "Request INCWAGE instead" in message
    assert "allow_flag_variables=True" in message


def test_extract_allows_a_flag_variable_through_the_escape_hatch(
    tmp_path: Path, make_ddi_xml
) -> None:
    """The verdict is a label heuristic over whatever codebooks are on disk, so
    a reworded label or a hand-copied .xml can make a legitimate variable
    permanently unrequestable. allow_flag_variables is the way out that does
    not involve deleting files - and it must say so in the log.
    """
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(
            start_id=1, flags={"INCWAGE": ["QINCWAGE"]}, make_ddi_xml=make_ddi_xml
        ),
    )
    extractor.extract(collection="cps", samples=["cps2006_09s"], variables=["INCWAGE"])

    with structlog.testing.capture_logs() as logs:
        record = extractor.extract(
            collection="cps",
            samples=["cps2007_09s"],
            variables=["QINCWAGE"],
            allow_flag_variables=True,
        )

    assert record.metadata["variables"] == ("QINCWAGE",)
    requested = [
        entry for entry in logs if entry["event"] == "ipums_flag_variable_requested"
    ]
    assert len(requested) == 1
    assert requested[0]["allowed"] is True


def test_extract_logs_when_it_rejects_a_flag_variable(
    tmp_path: Path, make_ddi_xml
) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(
            start_id=1, flags={"INCWAGE": ["QINCWAGE"]}, make_ddi_xml=make_ddi_xml
        ),
    )
    extractor.extract(collection="cps", samples=["cps2006_09s"], variables=["INCWAGE"])

    extractor.client = _ClientMustNotBeCalled()
    with structlog.testing.capture_logs() as logs:
        with pytest.raises(ValueError):
            extractor.extract(
                collection="cps", samples=["cps2007_09s"], variables=["QINCWAGE"]
            )

    requested = [
        entry for entry in logs if entry["event"] == "ipums_flag_variable_requested"
    ]
    assert len(requested) == 1
    assert requested[0]["allowed"] is False


def test_extract_rejects_known_topcode_flag_with_its_own_message(
    tmp_path: Path, make_ddi_xml
) -> None:
    # A topcode flag arrives regardless of data_quality_flags, so telling the
    # user to switch that on would be wrong advice.
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    (collection_dir / "cps_00001.xml").write_text(
        make_ddi_xml(
            [
                ("INCFARM", "Farm income", 7),
                ("TINCFARM", "Topcode Flag for INCFARM", 1),
            ]
        ),
        encoding="utf-8",
    )
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=_ClientMustNotBeCalled()
    )

    with pytest.raises(ValueError) as excinfo:
        extractor.extract(
            collection="cps", samples=["cps2006_09s"], variables=["TINCFARM"]
        )

    message = str(excinfo.value)
    assert "topcode flag for INCFARM" in message
    assert "regardless of data_quality_flags" in message


def test_extract_never_rejects_a_regular_q_or_t_prefixed_variable(
    tmp_path: Path, make_ddi_xml
) -> None:
    """TRANWORK/TRANTIME are ordinary IPUMS variables. Rejecting a name for
    starting with T or Q would block legitimate requests, so detection reads
    the codebook label and nothing else."""
    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    (collection_dir / "cps_00001.xml").write_text(
        make_ddi_xml(
            [
                ("TRANWORK", "Means of transportation to work", 2),
                ("TRANTIME", "Travel time to work", 3),
                ("INCWAGE", "Wage income", 7),
                ("QINCWAGE", "Data quality flag for INCWAGE", 1),
            ]
        ),
        encoding="utf-8",
    )
    client = _SequentialFakeClient(start_id=1, make_ddi_xml=make_ddi_xml)
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=client
    )

    # Same collection as the seeded codebook, so the registry is populated and
    # genuinely consulted - it knows QINCWAGE is a flag and must still let
    # TRANWORK/TRANTIME through.
    extractor.extract(
        collection="cps",
        samples=["cps2006_09s"],
        variables=["TRANWORK", "TRANTIME"],
    )

    assert client.submit_calls == 1


def test_extract_lets_an_unknown_flag_name_reach_the_api(tmp_path: Path) -> None:
    # Nothing on disk describes QOINCWAGE yet, so it cannot be rejected
    # locally - it goes to the API, which is the designed fallback.
    client = _BadRequestFakeClient()
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=client
    )

    with pytest.raises(BadIpumsApiRequest):
        extractor.extract(
            collection="cps", samples=["cps1989_03s"], variables=["QOINCWAGE"]
        )


def test_extract_wraps_bad_api_request_with_flag_guidance(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=_BadRequestFakeClient()
    )

    with pytest.raises(BadIpumsApiRequest) as excinfo:
        extractor.extract(
            collection="cps", samples=["cps1989_03s"], variables=["QOINCWAGE"]
        )

    message = str(excinfo.value)
    # The server's own wording survives verbatim...
    assert _BadRequestFakeClient.detail in message
    # ...with the flag guidance appended, and the original chained.
    assert "data_quality_flags=True" in message
    assert isinstance(excinfo.value.__cause__, BadIpumsApiRequest)


def test_extract_requests_per_variable_data_quality_flags(
    tmp_path: Path, make_ddi_xml
) -> None:
    # Guards the silent-kwarg regression: passing data_quality_flags to
    # MicrodataExtract lands it in **kwargs as a top-level field while every
    # per-variable entry still says false, and swallows a misspelled keyword.
    client = _SequentialFakeClient(start_id=1, make_ddi_xml=make_ddi_xml)
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=client
    )

    extractor.extract(
        collection="cps",
        samples=["cps2006_09s"],
        variables=["AGE", "SEX"],
        data_quality_flags=True,
    )

    built = client.submitted[0]
    assert built["variables"]["AGE"]["dataQualityFlags"] is True
    assert built["variables"]["SEX"]["dataQualityFlags"] is True


def test_extract_omits_data_quality_flags_when_false(
    tmp_path: Path, make_ddi_xml
) -> None:
    client = _SequentialFakeClient(start_id=1, make_ddi_xml=make_ddi_xml)
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=client
    )

    extractor.extract(
        collection="cps",
        samples=["cps2006_09s"],
        variables=["AGE"],
        data_quality_flags=False,
    )

    assert client.submitted[0]["variables"]["AGE"]["dataQualityFlags"] is False


def _seed_cached_entry(tmp_path: Path, make_ddi_xml) -> Path:
    """One sound manifest entry for cps_00029 (cps2006_09s, AGE) with both files
    on disk - enough for extract() to resolve a cache hit.
    """
    from src.extractors.base import build_extraction_record
    from src.extractors.manifest import append_to_manifest

    collection_dir = tmp_path / "cps"
    collection_dir.mkdir()
    data_path = collection_dir / "cps_00029.dat.gz"
    data_path.write_bytes(b"data")
    ddi_path = collection_dir / "cps_00029.xml"
    ddi_path.write_text(make_ddi_xml([("AGE", "Age", 2)]), encoding="utf-8")
    append_to_manifest(
        collection_dir,
        build_extraction_record(
            source="ipums_api",
            extraction_id="cps_00029",
            file_path=data_path,
            metadata={
                "collection": "cps",
                "samples": ("cps2006_09s",),
                "variables": ("AGE",),
                "extract_id": 29,
                "ddi_path": str(ddi_path),
                "cached": False,
            },
        ),
    )
    return collection_dir


def test_find_matching_extract_skips_entry_with_mismatched_size(
    tmp_path: Path, make_ddi_xml
) -> None:
    collection_dir = _seed_cached_entry(tmp_path, make_ddi_xml)
    # Corrupt the file
    (collection_dir / "cps_00029.dat.gz").write_bytes(b"truncated")

    match = find_matching_extract(
        collection_dir,
        samples=["cps2006_09s"],
        variables=["AGE"],
        data_structure=_default_data_structure(),
        data_quality_flags=True,
    )

    assert match is None


def test_find_matching_extract_reuses_entry_without_flag_or_structure_keys(
    tmp_path: Path, make_ddi_xml
) -> None:
    # Entries predating those keys were pulled with today's defaults
    # (rectangular-on-P, flags on). Treating a missing key as "different"
    # would make every one of them a permanent cache miss and re-spend quota.
    _seed_cached_entry(tmp_path, make_ddi_xml)
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=_ClientMustNotBeCalled()
    )

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert record.metadata["cached"] is True


@pytest.mark.parametrize(
    ("bad_entry", "reason"),
    [
        pytest.param(
            {"extraction_id": "cps_00030", "metadata": "not-a-mapping"},
            "metadata_not_a_mapping",
            id="scalar_metadata",
        ),
        pytest.param(
            {
                "extraction_id": "cps_00030",
                "metadata": {"samples": ["cps2006_09s"], "variables": ["AGE"]},
            },
            "missing_metadata_keys",
            id="partial_metadata",
        ),
        pytest.param(
            {
                "metadata": {
                    "samples": ["cps2006_09s"],
                    "variables": ["AGE"],
                    "ddi_path": "/nonexistent/cps_00030.xml",
                    "extract_id": 30,
                }
            },
            "missing_entry_keys",
            id="no_file_path",
        ),
        pytest.param(
            # build_coverage requires extraction_id too. If the cache accepted
            # an entry the planner cannot see, extract_incremental would keep
            # re-planning a sample it already has and never converge.
            {
                "file_path": "/nonexistent/cps_00030.dat.gz",
                "metadata": {
                    "samples": ["cps2006_09s"],
                    "variables": ["AGE"],
                    "ddi_path": "/nonexistent/cps_00030.xml",
                    "extract_id": 30,
                },
            },
            "missing_entry_keys",
            id="no_extraction_id",
        ),
    ],
)
def test_find_matching_extract_skips_a_malformed_entry(
    tmp_path: Path, make_ddi_xml, bad_entry, reason: str
) -> None:
    """A truncated entry is skipped with a reason, and the sound entry beside it
    still produces the cache hit - a corrupted manifest must not silently turn
    into a fresh, quota-spending extract.
    """
    import yaml

    collection_dir = _seed_cached_entry(tmp_path, make_ddi_xml)
    manifest = collection_dir / "_MANIFEST.yaml"
    entries = yaml.safe_load(manifest.read_text())
    manifest.write_text(yaml.safe_dump(entries + [bad_entry], sort_keys=False))
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=_ClientMustNotBeCalled()
    )

    with structlog.testing.capture_logs() as logs:
        record = extractor.extract(
            collection="cps", samples=["cps2006_09s"], variables=["AGE"]
        )

    assert record.metadata["cached"] is True
    skipped = [entry for entry in logs if entry["event"] == "manifest_entry_skipped"]
    assert reason in [entry["reason"] for entry in skipped]


def test_find_matching_extract_returns_the_newest_of_two_matching_entries(
    tmp_path: Path,
) -> None:
    """Two entries match equally well; the later one wins. `reversed` is what
    makes that true, and nothing else in the suite fails if it is dropped.
    """
    client = _SequentialFakeClient(start_id=1)
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=client
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])
    extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"], force=True
    )
    assert len(read_manifest(tmp_path / "cps")) == 2
    extractor.client = _ClientMustNotBeCalled()

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    assert record.metadata["cached"] is True
    assert record.metadata["extract_id"] == 2


def test_find_matching_extract_skips_an_entry_whose_samples_are_not_a_list(
    tmp_path: Path, make_ddi_xml
) -> None:
    """A bare string passes iter_valid_entries - the key is present - so the
    shape check inside the match loop is the only thing rejecting it. Without
    it, set("cps2006_09s") compares characters and never matches.
    """
    collection_dir = _seed_cached_entry(tmp_path, make_ddi_xml)
    manifest = collection_dir / "_MANIFEST.yaml"
    entries = yaml.safe_load(manifest.read_text())
    entries.append(
        {
            "extraction_id": "cps_00030",
            "file_path": "/nonexistent/cps_00030.dat.gz",
            "metadata": {
                "samples": "cps2006_09s",
                "variables": ["AGE"],
                "ddi_path": "/nonexistent/cps_00030.xml",
                "extract_id": 30,
            },
        }
    )
    manifest.write_text(yaml.safe_dump(entries, sort_keys=False))
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=_ClientMustNotBeCalled()
    )

    with structlog.testing.capture_logs() as logs:
        record = extractor.extract(
            collection="cps", samples=["cps2006_09s"], variables=["AGE"]
        )

    assert record.metadata["cached"] is True
    # Still the old event name: iter_valid_entries emits "manifest_entry_skipped",
    # the shape checks left in this loop emit "ipums_manifest_entry_skipped".
    # Update this together with that rename.
    assert "samples_or_variables_not_a_list_of_names" in [
        entry.get("reason")
        for entry in logs
        if entry["event"] == "ipums_manifest_entry_skipped"
    ]


def test_extract_carries_over_the_recorded_checksum_without_rehashing(
    tmp_path: Path,
) -> None:
    """The size-matched branch trusts what the entry recorded. Pinned with a
    checksum that is wrong on purpose: a re-hash would overwrite it, so the
    bogus value surviving is the proof no re-hash happened.
    """
    bogus_sha = "0" * 64  # pragma: allowlist secret
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE"])
    # Size left alone - only a matching size reaches this branch at all.
    _corrupt_manifest_checksum(tmp_path / "cps", sha256=bogus_sha)
    extractor.client = _ClientMustNotBeCalled()

    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["AGE"]
    )

    data_path = tmp_path / "cps" / "cps_00001.dat.gz"
    assert record.metadata["cached"] is True
    assert record.sha256 == bogus_sha
    assert record.sha256 != hashlib.sha256(data_path.read_bytes()).hexdigest()


def test_find_matching_extract_misses_when_flags_explicitly_differ(
    tmp_path: Path, make_ddi_xml
) -> None:
    client = _SequentialFakeClient(start_id=1, make_ddi_xml=make_ddi_xml)
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=client
    )
    extractor.extract(
        collection="cps",
        samples=["cps2006_09s"],
        variables=["AGE"],
        data_quality_flags=True,
    )

    extractor.extract(
        collection="cps",
        samples=["cps2006_09s"],
        variables=["AGE"],
        data_quality_flags=False,
    )

    # The cached extract has flag columns the flags-off request does not want,
    # so it must not be reused.
    assert client.submit_calls == 2


def test_extract_accepts_lowercase_variable_names(tmp_path, make_ddi_xml) -> None:
    client = _SequentialFakeClient(start_id=1, make_ddi_xml=make_ddi_xml)
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY, storage_dir=tmp_path, client=client
    )
    record = extractor.extract(
        collection="cps", samples=["cps2006_09s"], variables=["age"]
    )
    assert client.submitted[0]["variables"]["AGE"]["dataQualityFlags"] is True
    assert record.metadata["variables"] == ("AGE",)


def test_extract_incremental_accepts_lowercase_variable_names(tmp_path: Path) -> None:
    extractor = IPUMSExtractor(
        api_key=_FAKE_API_KEY,
        storage_dir=tmp_path,
        client=_SequentialFakeClient(start_id=1),
    )
    _seed_manifest_entry(extractor, "cps", ["cps2006_09s"], ["AGE", "SEX"])
    client = _SequentialFakeClient(start_id=50)
    extractor.client = client

    records = extractor.extract_incremental(
        collection="cps", samples=["cps2006_09s"], variables=["AGE", "SEX", "race"]
    )

    assert client.submitted[0]["variables"]["RACE"]["dataQualityFlags"] is True
    assert records[0].metadata["variables"] == ("RACE",)


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
