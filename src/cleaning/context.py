"""Define context CleaningContext: the immutable, versioned methodology
configuration for one cleaning-pipeline run.

Classes:
    SourceProfile:
        Identifies which source is used to load a context.
    RunMetadata:
        Saves general info about given run (for future logs).
    CleaningContext:
        Builds a full context from a pipeline coonfig file (source_profile.yaml)
        and subcontext files (*.yaml).

Functions:
    _deep_merge(...):
        Used for force update of existing context.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

SourceKind = Literal["ipums_cps_asec", "nber_mw", "raw_asec_march", "psid"]


class SourceProfile(BaseModel):
    """Identifies which source a CleaningContext was built for, and what it
    can provide. Steps use instsnces of that class to branch on source-specific
    behavior.

    Attributes:
        model_config (ConfigDict):
            Class (pydantic model) configuration.
        kind (SourceKind):
            Identifies data source ("psid", "raw_asec_march", etc.).
        notes (str):
            Any other notes for logs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: SourceKind
    available_flags: frozenset[str] = frozenset()
    notes: str = ""


class RunMetadata(BaseModel):
    """Per-run bookkeeping for logs."""

    # Configure as immutable
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class CleaningContext(BaseModel):
    """Immutable, versioned methodology configuration for one cleaning run.

    Attributes:
        model_config (ConfigDict):
            Class (pydantic model) configuration. Frozen - immutable for
            external functions.
        source_profile (SourceProfile):
            Contains general info about data source.
        run_metadata (RunMetadata):
            To collect info for logs.

    Methods:
        from_config(...):
            Builds a context from yaml-config file.
        compute_hash(...):
            Needs to identify a verstion of the context used.
    """

    # Configure as immutable
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_profile: SourceProfile
    run_metadata: RunMetadata = Field(default_factory=RunMetadata)

    @classmethod
    def from_config(
        cls,
        config_dir: Path,
        source: SourceKind,
        overrides: dict | None = None,
    ) -> CleaningContext:
        """Build a CleaningContext for given data source from YAML config.

        Args:
            config_dir (Path):
                A path to yaml-configuration.
            source (SourceKind):
                A data source that is used.
            overrides:
                Overrides the instance with new information.

        Raises:
            ValueError if `config_dir/source_profile.yaml` is missing or
                doesn't parse into a valid SourceProfile matching `source`.
        """
        # Load source profile
        source_profile_path = config_dir / "source_profile.yaml"
        if not source_profile_path.exists():
            raise ValueError(
                f"CleaningContext.from_config: missing {source_profile_path} - "
                "every source needs a source_profile.yaml before a context "
                "can be built for it"
            )
        raw = yaml.safe_load(source_profile_path.read_text()) or {}

        # Override if needed
        if overrides:
            raw = _deep_merge(raw, overrides)
        raw.setdefault("kind", source)
        if raw["kind"] != source:
            raise ValueError(
                f"CleaningContext.from_config: {source_profile_path} declares "
                f"kind={raw['kind']!r} but source={source!r} was requested"
            )
        try:
            source_profile = SourceProfile.model_validate(raw)
        except Exception as exc:
            raise ValueError(
                f"CleaningContext.from_config: {source_profile_path} failed to "
                f"validate as SourceProfile: {exc}"
            ) from exc

        return cls(
            source_profile=source_profile,
        )

    def compute_hash(self) -> str:
        """A stable hash of this context's methodology.

        - Excludes `run_id` (a run identifier, not a methodology).
        """
        payload = self.model_dump(mode="json")
        payload["run_metadata"].pop("run_id", None)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        )
        return digest.hexdigest()


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` onto a copy of `base`."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
