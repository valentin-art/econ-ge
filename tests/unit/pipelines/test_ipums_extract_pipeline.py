"""Tests for pipelines.ipums_extract_pipeline.

Covers the data_quality_flags plumbing: the setting has to be reachable from
the pipeline and from each request, without either silently overriding the
other. Also covers data_structure passthrough and the loop's fail-fast
contract, both of which decide what reaches the IPUMS API - and every
submission that reaches it costs account quota.
"""

from types import SimpleNamespace

import pytest

from src.config.sources import IPUMSExtractRequest
from src.pipelines import ipums_extract_pipeline
from src.pipelines.ipums_extract_pipeline import extract_ipums_extracts

_FAKE_API_KEY = "fake-key"  # pragma: allowlist secret


class _RecordingExtractor:
    def __init__(self, api_key: str, calls: list[dict]) -> None:
        self.api_key = api_key
        self.calls = calls

    def extract_incremental(self, **kwargs):
        self.calls.append(kwargs)
        return []


@pytest.fixture
def recording_extractor(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        ipums_extract_pipeline,
        "IPUMSExtractor",
        lambda api_key: _RecordingExtractor(api_key, calls),
    )
    return SimpleNamespace(calls=calls)


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


@pytest.mark.parametrize("api_key", ["", None])
def test_missing_api_key_raises(recording_extractor, api_key) -> None:
    with pytest.raises(RuntimeError, match="IPUMS_API_KEY"):
        extract_ipums_extracts(api_key, extracts=[_request()])
    assert recording_extractor.calls == []


def test_data_structure_defaults_to_none(recording_extractor) -> None:
    # None, not {}: the extractor supplies rectangular-on-P, and an explicit
    # empty dict here would be a different cache key in find_matching_extract.
    extract_ipums_extracts(_FAKE_API_KEY, extracts=[_request()])

    assert recording_extractor.calls[0]["data_structure"] is None


def test_request_can_set_its_own_data_structure(recording_extractor) -> None:
    # Symmetric with data_quality_flags: a hierarchical pull is a property of
    # what a collection is for, so it belongs on the request, not the run.
    hierarchical = {"hierarchical": {}}

    extract_ipums_extracts(
        _FAKE_API_KEY,
        extracts=[_request(data_structure=hierarchical), _request()],
    )

    assert [call["data_structure"] for call in recording_extractor.calls] == [
        hierarchical,
        None,
    ]


def test_data_structure_reaches_every_request(recording_extractor) -> None:
    hierarchical = {"hierarchical": {}}

    # the pipeline argument overrides a request that asked for something else
    extract_ipums_extracts(
        _FAKE_API_KEY,
        extracts=[_request(), _request(data_structure={"rectangular": {"on": "H"}})],
        data_structure=hierarchical,
    )

    assert [call["data_structure"] for call in recording_extractor.calls] == [
        hierarchical,
        hierarchical,
    ]


def test_failing_request_stops_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-fast is deliberate - a failure is usually auth or quota, shared by
    every remaining request, and each submission costs quota.
    """
    calls: list[dict] = []

    class _FailingExtractor(_RecordingExtractor):
        def extract_incremental(self, **kwargs):
            super().extract_incremental(**kwargs)
            raise RuntimeError("boom")

    monkeypatch.setattr(
        ipums_extract_pipeline,
        "IPUMSExtractor",
        lambda api_key: _FailingExtractor(api_key, calls),
    )

    with pytest.raises(RuntimeError, match="boom"):
        extract_ipums_extracts(
            _FAKE_API_KEY, extracts=[_request(), _request(description="second")]
        )

    assert len(calls) == 1
