"""External stage: IPUMS API -> raw .dat.gz + DDI .xml on disk (data/external/ipums/).

Companion to `pipelines.ipums_parse_pipeline` (external -> bronze). Mirrors
`pipelines.bea_extract_pipeline`'s per-item extraction loop over a source list
in `config.sources`.
"""

from collections.abc import Sequence

import structlog

from extractors.ipums.ipums_api import IPUMSExtractor
from src.config import sources
from src.extractors.base import ExtractionRecord

log = structlog.get_logger(__name__)


def extract_ipums_extracts(
    api_key: str | None,
    extracts: Sequence[sources.IPUMSExtractRequest] | None = None,
    force: bool = False,
    data_quality_flags: bool | None = None,
    data_structure: dict[str, dict[str, str]] | None = None,
) -> list[ExtractionRecord]:
    """Download every IPUMS extract request via the IPUMS API.

    Saves each request's raw .dat.gz + DDI .xml codebook as-is to
    data/external/ipums/{collection}/ (via extractors.ipums_api.IPUMSExtractor)
    and appends a record to that directory's _MANIFEST.yaml. Each request is
    compared against its collection's existing coverage
    (extractors.ipums_api.IPUMSExtractor.extract_incremental) and only the
    missing samples/variables are actually submitted - unlike BEA, which
    always re-downloads, every IPUMS submission counts against the user's
    account extract-request quota, and a request can be partially covered by
    several prior extracts.

    Args:
        api_key (str | None):
            IPUMS API registration key (settings.ipums_api_key). Empty or None
            raises rather than falling back to the environment - a silent
            fallback would spend account quota unasked.
        extracts(list[IPUMSExtractRequest] | None):
            Requests to pull; defaults to sources.IPUMS_EXTRACTS.
        force (bool):
            Re-submit and re-download every request in full, bypassing
            coverage-checking entirely.
        data_quality_flags (bool | None):
            Override every request's own `data_quality_flags`
            field. None (the default) leaves each request to decide, which
            is itself True unless the request says otherwise. IPUMS attaches
            a flag column to each requested variable that has one; the flag
            column is never listed in a request's `variables`.
        data_structure (dict[str, dict[str, str]] | None):
            Override every request's own `data_structure` field, e.g.
            {"hierarchical": {}}. None (the default) leaves each request to
            decide, which is itself the extractor's rectangular-on-P default
            unless the request says otherwise - the same None-sentinel shape
            as `data_quality_flags`. It is a cache dimension in
            extractors.ipums_api.find_matching_extract, so changing it makes
            prior extracts of the same samples/variables non-matching.

    Returns:
        list[ExtractionRecord]

    Raises:
        RuntimeError
            `api_key` is empty; nothing is submitted.
        Exception
            Whatever a request raises is re-raised immediately, discarding the
            records collected so far - see the fail-fast note in the loop.
    """
    if not api_key:
        raise RuntimeError("IPUMS_API_KEY is empty - cannot extract IPUMS data")
    extracts = extracts if extracts is not None else sources.IPUMS_EXTRACTS

    log.info("ipums_extract_pipeline_start", n_requests=len(extracts), force=force)
    extractor = IPUMSExtractor(api_key=api_key)
    records: list[ExtractionRecord] = []
    for req in extracts:
        try:
            records.extend(
                extractor.extract_incremental(
                    collection=req.collection,
                    samples=req.samples,
                    variables=req.variables,
                    description=req.description,
                    force=force,
                    data_quality_flags=(
                        req.data_quality_flags
                        if data_quality_flags is None
                        else data_quality_flags
                    ),
                    data_structure=(
                        req.data_structure if data_structure is None else data_structure
                    ),
                )
            )
        except Exception:
            # Deliberate fail-fast: a failure here is usually auth, quota
            # exhaustion, or a malformed request shared by the remaining
            # entries, and every further submission costs quota. Anything
            # already downloaded is on disk and in _MANIFEST.yaml, so the loss
            # is the returned list, not the data - log what it held.
            log.error(
                "ipums_extract_pipeline_failed",
                collection=req.collection,
                samples=list(req.samples),
                n_records_discarded=len(records),
                exc_info=True,
            )
            raise
    log.info(
        "ipums_extract_pipeline_complete",
        n_requests=len(extracts),
        n_records=len(records),
    )
    return records
