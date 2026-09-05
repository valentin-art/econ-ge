"""Shared parsing of BEA API JSON responses, used by both the extractor (to pull
metadata for the manifest) and the bronze parser (to read Data back off disk).
"""

import json
from pathlib import Path


def read_bea_results(json_path: Path) -> dict:
    """Unwrap a raw BEA API JSON response file down to its 'Results' node.

    Raises RuntimeError if BEA reports an API-level error instead of data.
    """
    payload = json.loads(json_path.read_text())
    api = payload["BEAAPI"]
    if "Error" in api:
        raise RuntimeError(f"BEA API error in {json_path}: {api['Error']}")
    results = api["Results"]
    if isinstance(results, list):  # BEA returns a 1-item list for a few datasets
        results = results[0]
    if "Error" in results:
        raise RuntimeError(f"BEA API error in {json_path}: {results['Error']}")
    return results
