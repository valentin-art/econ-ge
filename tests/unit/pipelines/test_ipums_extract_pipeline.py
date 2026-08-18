"""Tests for pipelines.ipums_extract_pipeline.

Covers the data_quality_flags plumbing: the setting has to be reachable from
the pipeline and from each request, without either silently overriding the
other.
"""

import pytest

from src.config.sources import IPUMSExtractRequest
from src.pipelines import ipums_extract_pipeline
from src.pipelines.ipums_extract_pipeline import extract_ipums_extracts

_FAKE_API_KEY = "fake-key"  # pragma: allowlist secret


class _RecordingExtractor:
    """Captures extract_incremental's arguments instead of hitting the API."""

    calls: list[dict] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def extract_incremental(self, **kwargs):
        type(self).calls.append(kwargs)
        return []


@pytest.fixture
def recording_extractor(monkeypatch: pytest.MonkeyPatch):
    _RecordingExtractor.calls = []
    monkeypatch.setattr(ipums_extract_pipeline, "IPUMSExtractor", _RecordingExtractor)
    return _RecordingExtractor


def _request(**overrides) -> IPUMSExtractRequest:
    return IPUMSExtractRequest(
        collection="cps",
        samples=("cps2006_09s",),
        variables=("AGE",),
        **overrides,
    )


def test_requests_default_to_flags_on(recording_extractor) -> None:
    extract_ipums_extracts(_FAKE_API_KEY, extracts=[_request()])

    assert recording_extractor.calls[0]["data_quality_flags"] is True


def test_request_can_turn_flags_off(recording_extractor) -> None:
    extract_ipums_extracts(_FAKE_API_KEY, extracts=[_request(data_quality_flags=False)])

    assert recording_extractor.calls[0]["data_quality_flags"] is False


def test_pipeline_argument_overrides_every_request(recording_extractor) -> None:
    extract_ipums_extracts(
        _FAKE_API_KEY,
        extracts=[_request(), _request(data_quality_flags=False)],
        data_quality_flags=True,
    )

    assert [call["data_quality_flags"] for call in recording_extractor.calls] == [
        True,
        True,
    ]


def test_pipeline_default_leaves_each_request_to_decide(recording_extractor) -> None:
    # The None sentinel is the point: without it, a pipeline-level default of
    # True would silently overwrite a request that deliberately turned flags
    # off.
    extract_ipums_extracts(
        _FAKE_API_KEY,
        extracts=[_request(), _request(data_quality_flags=False)],
    )

    assert [call["data_quality_flags"] for call in recording_extractor.calls] == [
        True,
        False,
    ]


def test_empty_api_key_raises(recording_extractor) -> None:
    with pytest.raises(RuntimeError, match="IPUMS_API_KEY"):
        extract_ipums_extracts("", extracts=[_request()])
