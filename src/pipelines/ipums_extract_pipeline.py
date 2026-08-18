"""External stage: IPUMS API -> raw .dat.gz + DDI .xml on disk (data/external/ipums/).

Companion to `pipelines.ipums_parse_pipeline` (external -> bronze). Mirrors
`pipelines.bea_extract_pipeline`'s per-item extraction loop over a source list
in `config.sources`.
"""

import structlog

from src.config import sources
from src.extractors.base import ExtractionRecord
from src.extractors.ipums_api import IPUMSExtractor

log = structlog.get_logger(__name__)


def extract_ipums_extracts(
    api_key: str,
    extracts: list[sources.IPUMSExtractRequest] | None = None,
    force: bool = False,
    data_quality_flags: bool | None = None,
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

    Parameters
    ----------
    api_key  : IPUMS API registration key (settings.ipums_api_key)
    extracts : requests to pull; defaults to sources.IPUMS_EXTRACTS
    force    : re-submit and re-download every request in full, bypassing
               coverage-checking entirely
    data_quality_flags : override every request's own `data_quality_flags`
               field. None (the default) leaves each request to decide, which
               is itself True unless the request says otherwise. IPUMS attaches
               a flag column to each requested variable that has one; the flag
               column is never listed in a request's `variables`.
    """
    if api_key == "":
        raise RuntimeError("IPUMS_API_KEY is empty - cannot extract IPUMS data")
    extracts = extracts if extracts is not None else sources.IPUMS_EXTRACTS

    log.info("ipums_extract_pipeline_start", n_extracts=len(extracts), force=force)
    extractor = IPUMSExtractor(api_key=api_key)
    records = [
        record
        for req in extracts
        for record in extractor.extract_incremental(
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
        )
    ]
    log.info("ipums_extract_pipeline_complete", n_extracts=len(records))
    return records
