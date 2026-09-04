from pathlib import Path

import pytest
import structlog.testing

from src.extractors.base import build_extraction_record
from src.extractors.manifest import (
    append_to_manifest,
    as_name_list,
    iter_valid_entries,
    read_manifest,
)


def _record(tmp_path: Path, name: str):
    file_path = tmp_path / name
    file_path.write_bytes(b"raw data")
    return build_extraction_record(
        source="bea_api",
        extraction_id=name,
        file_path=file_path,
        metadata={"table": name},
    )


def test_read_manifest_returns_empty_list_when_missing(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) == []


def test_append_to_manifest_creates_and_appends(tmp_path: Path) -> None:
    record1 = _record(tmp_path, "FAAt201.parquet")
    append_to_manifest(tmp_path, record1)

    entries = read_manifest(tmp_path)
    assert len(entries) == 1
    assert entries[0]["extraction_id"] == "FAAt201.parquet"
    assert entries[0]["sha256"] == record1.sha256

    record2 = _record(tmp_path, "T11400.parquet")
    append_to_manifest(tmp_path, record2)

    entries = read_manifest(tmp_path)
    assert len(entries) == 2
    assert [e["extraction_id"] for e in entries] == [
        "FAAt201.parquet",
        "T11400.parquet",
    ]


def test_append_to_manifest_creates_missing_parent_dir(tmp_path: Path) -> None:
    nested_dir = tmp_path / "bea" / "FixedAssets"
    record = _record(tmp_path, "FAAt201.parquet")

    manifest_path = append_to_manifest(nested_dir, record)

    assert manifest_path.exists()
    assert read_manifest(nested_dir) != []


# --- as_name_list -----------------------------------------------------------


def test_as_name_list_accepts_a_list_or_a_tuple() -> None:
    assert as_name_list(["AGE", "SEX"]) == ["AGE", "SEX"]
    # extract() holds tuples in memory; only a YAML round trip makes them lists.
    assert as_name_list(("AGE", "SEX")) == ["AGE", "SEX"]
    # Empty is a valid list of names, and must stay distinguishable from None.
    assert as_name_list([]) == []


def test_as_name_list_rejects_a_bare_string() -> None:
    # The shape a hand-edited manifest produces: list("AGE") would silently
    # become ["A", "G", "E"] rather than being rejected.
    assert as_name_list("AGE") is None


def test_as_name_list_rejects_anything_that_is_not_all_names() -> None:
    assert as_name_list(["AGE", 2006]) is None
    assert as_name_list(2006) is None
    assert as_name_list(None) is None


# --- iter_valid_entries -----------------------------------------------------

_SOUND_ENTRY = {
    "file_path": "/data/cps_00001.dat.gz",
    "extraction_id": "cps_00001",
    "metadata": {"samples": ["cps2006_09s"], "variables": ["AGE"]},
}


@pytest.mark.parametrize(
    ("bad_entry", "reason"),
    [
        pytest.param(
            {"extraction_id": "cps_00002", "metadata": None},
            "metadata_not_a_mapping",
            id="metadata_not_a_mapping",
        ),
        pytest.param(
            "just-a-string",
            "metadata_not_a_mapping",
            id="entry_not_a_mapping",
        ),
        pytest.param(
            # Also missing file_path: metadata is checked first, so this pins
            # which of the two reasons is reported.
            {"extraction_id": "cps_00002", "metadata": {"samples": ["cps2007_09s"]}},
            "missing_metadata_keys",
            id="missing_metadata_keys",
        ),
        pytest.param(
            {"metadata": {"samples": ["cps2007_09s"], "variables": ["AGE"]}},
            "missing_entry_keys",
            id="missing_entry_keys",
        ),
    ],
)
def test_iter_valid_entries_skips_a_malformed_entry_with_its_reason(
    tmp_path: Path, bad_entry, reason: str
) -> None:
    """One warning per skipped entry, and the sound entry beside it still comes
    through - a caller must not lose a whole manifest to one bad row.
    """
    with structlog.testing.capture_logs() as logs:
        yielded = list(
            iter_valid_entries(
                tmp_path,
                required_entry_keys=("file_path", "extraction_id"),
                required_metadata_keys=("samples", "variables"),
                entries=[bad_entry, _SOUND_ENTRY],
            )
        )

    assert [entry for entry, _ in yielded] == [_SOUND_ENTRY]
    skipped = [entry for entry in logs if entry["event"] == "manifest_entry_skipped"]
    assert [entry["reason"] for entry in skipped] == [reason]


def test_iter_valid_entries_yields_metadata_alongside_the_entry(
    tmp_path: Path,
) -> None:
    entry, metadata = next(iter(iter_valid_entries(tmp_path, entries=[_SOUND_ENTRY])))

    assert entry is _SOUND_ENTRY
    assert metadata is _SOUND_ENTRY["metadata"]


def test_iter_valid_entries_reads_the_manifest_when_entries_is_none(
    tmp_path: Path,
) -> None:
    append_to_manifest(tmp_path, _record(tmp_path, "FAAt201.parquet"))

    yielded = list(iter_valid_entries(tmp_path, required_entry_keys=("extraction_id",)))

    assert [entry["extraction_id"] for entry, _ in yielded] == ["FAAt201.parquet"]


def test_iter_valid_entries_prefers_supplied_entries_over_the_file(
    tmp_path: Path,
) -> None:
    """`entries=` is how a caller that already read _MANIFEST.yaml avoids a
    second read; the file must not be consulted at all.
    """
    append_to_manifest(tmp_path, _record(tmp_path, "FAAt201.parquet"))

    yielded = list(iter_valid_entries(tmp_path, entries=[_SOUND_ENTRY]))

    assert [entry for entry, _ in yielded] == [_SOUND_ENTRY]


def test_iter_valid_entries_treats_an_empty_entries_list_as_empty(
    tmp_path: Path,
) -> None:
    """An empty supplied list means "nothing to iterate", not "fall back to the
    file" - the difference between `entries is not None` and `if entries`.
    """
    append_to_manifest(tmp_path, _record(tmp_path, "FAAt201.parquet"))

    assert list(iter_valid_entries(tmp_path, entries=[])) == []
