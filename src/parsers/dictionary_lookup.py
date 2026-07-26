"""Generic lookup over the per-source cleaned CPS variable dictionaries
persisted under data/reference/cps/{source}/ (e.g. data/reference/cps/mw/1964.json),
each mapping {variable_name: {"start": int, "end": int, "numeric": bool,
"Description": str, "Values": {code: label}}}.

Deliberately source-agnostic: this module only knows the on-disk convention
(settings.paths.cps_clean_dictionaries_dir(source)/*.json), not how any
particular source's dictionary was built — that parsing logic lives with the
source (e.g. parsers.cps.build_variable_dictionary).
"""

import json
from pathlib import Path

from src.config.settings import settings


def get_variable_info(
    source: str, variable: str, dictionaries_root: Path | None = None
) -> dict[str, object]:
    """Look up one variable's Description + Values for a data source.

    Scans dictionaries_root/{source}/*.json (defaults to
    settings.paths.reference / "cps") — there's one file per extracted
    period (e.g. 1964.json) — and returns the entry from the first file that
    defines `variable`. Different periods under the same source are expected
    to agree when a variable appears in both (same underlying SPS dictionary
    vintage); this doesn't attempt to reconcile disagreements.

    Raises KeyError if `source` has no dictionaries on disk, or none of them
    define `variable`.
    """
    root = (
        dictionaries_root
        if dictionaries_root is not None
        else settings.paths.reference / "cps"
    )
    source_dir = root / source
    json_paths = sorted(source_dir.glob("*.json"))
    if not json_paths:
        raise KeyError(f"No dictionaries found for source {source!r} in {source_dir}")

    for json_path in json_paths:
        variable_dictionary = json.loads(json_path.read_text(encoding="utf-8"))
        if variable in variable_dictionary:
            result: dict[str, object] = variable_dictionary[variable]
            return result

    raise KeyError(f"Variable {variable!r} not found in any {source!r} dictionary")
