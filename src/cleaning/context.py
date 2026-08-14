"""Define context CleaningContext: the immutable, versioned methodology
configuration for one cleaning-pipeline run.

Classes:
    SourceProfile:
        Identifies which source is used to load a context.
    RunMetadata:
        Saves general info about given run (for future logs).
    YearBandThreshold:
        One year-banded topcode threshold entry.
    TopcodeConfig:
        Multiplier + year-banded thresholds for one income column's topcode
        adjustment (see TopcodeAdjuster).
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
from types import MappingProxyType
from typing import Literal

import polars as pl
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

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


class YearBandThreshold(BaseModel):
    """One year-banded topcode threshold: applies to `[start_year, end_year]`
    inclusive, matched either by exact equality (aa_clean's pre-1988 style)
    or `>=` (aa_clean's 1988+ style, `match_mode="gte"`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    start_year: int
    end_year: int
    threshold: float
    match_mode: Literal["exact", "gte"]


class TopcodeConfig(BaseModel):
    """Multiplier + year-banded thresholds for one income column's topcode
    adjustment (see `TopcodeAdjuster`).

    Authoring shorthand: raw YAML may supply either (or both) of `bands` (a
    list of `{start_year, end_year, threshold}` blocks, folded in as
    `match_mode="exact"`) and `per_year` (a compact `{year: value}` mapping,
    folded in one entry per year as `match_mode="gte"`) - see
    `_fold_threshold_blocks`. Both fold into the single `thresholds` list;
    neither `bands` nor `per_year` survives as a field on the built model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    multiplier: float = 1.0
    thresholds: list[YearBandThreshold] = Field(default_factory=list)
    uncovered_years: Literal["error", "skip"]

    @model_validator(mode="before")
    @classmethod
    def _fold_threshold_blocks(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        bands = data.pop("bands", None)
        per_year = data.pop("per_year", None)
        if bands is None and per_year is None:
            return data

        thresholds = list(data.get("thresholds", []))
        for band in bands or []:
            if "match_mode" in band:
                raise ValueError(
                    f"TopcodeConfig: bands entries always fold as match_mode='exact'; "
                    f"got an explicit match_mode={band['match_mode']!r} in {band!r} - "
                    "remove it or use the top-level `thresholds:` list directly"
                )
            thresholds.append({**band, "match_mode": "exact"})
        for year, value in (per_year or {}).items():
            thresholds.append(
                {
                    "start_year": year,
                    "end_year": year,
                    "threshold": value,
                    "match_mode": "gte",
                }
            )
        data["thresholds"] = thresholds
        return data

    @model_validator(mode="after")
    def _check_bands(self) -> TopcodeConfig:
        if not self.thresholds:
            raise ValueError(
                "TopcodeConfig: `thresholds` is empty - the adjuster would silently "
                "no-op on every row. Provide `bands:` and/or `per_year:`."
            )
        spans = sorted((t.start_year, t.end_year) for t in self.thresholds)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s1 > e1:
                raise ValueError(
                    f"TopcodeConfig: band {s1}-{e1} has start_year > end_year"
                )
            if s2 <= e1:
                raise ValueError(
                    f"TopcodeConfig: overlapping bands {s1}-{e1} and {s2}-{e2}; "
                    "band precedence would be silently order-dependent"
                )
        return self


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
        topcode (dict[str, TopcodeConfig]):
            Named topcode sub-contexts (one per income column), loaded from
            `config_dir/topcode/*.yaml`, keyed by filename stem. Empty when
            no `topcode/` directory exists for this source.
        crosswalks (dict[str, pl.DataFrame]):
            Named crosswalk tables, loaded from `crosswalks_dir/*.csv`,
            keyed by filename stem. Empty when `crosswalks_dir` is not
            given or doesn't exist.

    Methods:
        from_config(...):
            Builds a context from yaml-config file.
        compute_hash(...):
            Needs to identify a verstion of the context used.
    """

    # Configure as immutable
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source_profile: SourceProfile
    run_metadata: RunMetadata = Field(default_factory=RunMetadata)
    topcode: dict[str, TopcodeConfig] = Field(default_factory=dict)
    crosswalks: dict[str, pl.DataFrame] = Field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config_dir: Path,
        source: SourceKind,
        crosswalks_dir: Path | None = None,
        overrides: dict | None = None,
    ) -> CleaningContext:
        """Build a CleaningContext for given data source from YAML config.

        Args:
            config_dir (Path):
                A path to yaml-configuration.
            source (SourceKind):
                A data source that is used.
            crosswalks_dir (Path | None):
                A path to a directory of crosswalk `*.csv` files, each
                loaded into `context.crosswalks[<file stem>]`. Optional -
                skipped entirely (crosswalks stays `{}`) when not given or
                when the directory doesn't exist.
            overrides:
                Overrides the instance with new information.

        Raises:
            ValueError if `config_dir/source_profile.yaml` is missing or
                doesn't parse into a valid SourceProfile matching `source`,
                or if any `config_dir/topcode/*.yaml` fails to validate as
                a TopcodeConfig.
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
        except ValidationError as exc:
            raise ValueError(
                f"CleaningContext.from_config: {source_profile_path} failed to "
                f"validate as SourceProfile: {exc}"
            ) from exc

        # Load optional per-income-column topcode sub-contexts
        topcode: dict[str, TopcodeConfig] = {}
        topcode_dir = config_dir / "topcode"
        if topcode_dir.exists():
            for topcode_path in sorted(topcode_dir.glob("*.yaml")):
                raw_topcode = yaml.safe_load(topcode_path.read_text()) or {}
                try:
                    topcode[topcode_path.stem] = TopcodeConfig.model_validate(
                        raw_topcode
                    )
                except ValidationError as exc:
                    raise ValueError(
                        f"CleaningContext.from_config: {topcode_path} failed to "
                        f"validate as TopcodeConfig: {exc}"
                    ) from exc

        # Load optional crosswalk tables
        crosswalks: dict[str, pl.DataFrame] = {}
        if crosswalks_dir is not None and crosswalks_dir.exists():
            for crosswalk_path in sorted(crosswalks_dir.glob("*.csv")):
                crosswalks[crosswalk_path.stem] = pl.read_csv(crosswalk_path)

        return cls(
            source_profile=source_profile,
            topcode=topcode,
            crosswalks=crosswalks,
        )

    def compute_hash(self) -> str:
        """A stable hash of this context's methodology.

        - Excludes `run_id` (a run identifier, not a methodology).
        - `topcode`/`crosswalks` are dumped explicitly rather than via the
          whole-model `model_dump()`: both are frozen into `MappingProxyType`
          by `_freeze_mappings`, which pydantic's JSON serializer can't walk.
        """
        run_metadata = self.run_metadata.model_dump(mode="json")
        run_metadata.pop("run_id", None)
        payload = {
            "source_profile": self.source_profile.model_dump(mode="json"),
            "run_metadata": run_metadata,
            "topcode": {
                name: cfg.model_dump(mode="json")
                for name, cfg in sorted(self.topcode.items())
            },
            "crosswalks": {
                name: self.crosswalks[name].to_dicts()
                for name in sorted(self.crosswalks)
            },
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        )
        return digest.hexdigest()

    @model_validator(mode="after")
    def _freeze_mappings(self) -> CleaningContext:
        object.__setattr__(self, "topcode", MappingProxyType(dict(self.topcode)))
        object.__setattr__(self, "crosswalks", MappingProxyType(dict(self.crosswalks)))
        return self


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` onto a copy of `base`."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
