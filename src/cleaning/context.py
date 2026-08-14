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
    DeflatorTableConfig:
        A year-keyed deflator table (see DeflatorMergeStep).
    CleaningContext:
        Builds a full context from a pipeline config file (source_profile.yaml)
        and subcontext files (*.yaml).

Functions:
    _deep_merge(...):
        Used for force update of existing context.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from itertools import pairwise
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
    can provide. Steps use instances of that class to branch on source-specific
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
    thresholds: tuple[YearBandThreshold, ...] = Field(default_factory=tuple)
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
        for start, end in spans:
            if start > end:
                raise ValueError(
                    f"TopcodeConfig: band {start}-{end} has start_year > end_year; "
                    "is_between() would match nothing and the band would silently "
                    "no-op"
                )
        for (s1, e1), (s2, e2) in pairwise(spans):
            if s2 <= e1:
                raise ValueError(
                    f"TopcodeConfig: overlapping bands {s1}-{e1} and {s2}-{e2}; "
                    "band precedence would be silently order-dependent"
                )
        return self


class DeflatorTableConfig(BaseModel):
    """One year-keyed deflator table (income year -> multiplier), e.g. CPI
    or GDP-PCE (see `DeflatorMergeStep`). `uncovered_years` mirrors
    `TopcodeConfig`'s policy: an income year absent from `values` is either
    a raised error or a per-row warning, never a silent null.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    values_: dict[int, float] = Field(alias="values")
    uncovered_years: Literal["error", "skip"]

    @model_validator(mode="after")
    def _check_values(self) -> DeflatorTableConfig:
        if not self.values_:
            raise ValueError(
                "DeflatorTableConfig: `values` is empty - every income year "
                "would be uncovered."
            )
        return self

    @property
    def values(self) -> Mapping[int, float]:
        """Read-only view; the model itself stays pydantic-serializable."""
        return MappingProxyType(self.values_)


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
        deflators (dict[str, DeflatorTableConfig]):
            Named deflator tables, loaded from `config_dir/deflators/*.yaml`,
            keyed by filename stem. Empty when no `deflators/` directory
            exists for this source.
        crosswalks (dict[str, pl.DataFrame]):
            Named crosswalk tables, loaded from `crosswalks_dir/*.csv`,
            keyed by filename stem. Empty when `crosswalks_dir` is not
            given or doesn't exist.

    Methods:
        from_config(...):
            Builds a context from yaml-config file.
        compute_hash(...):
            Needs to identify a version of the context used.
    """

    # Configure as immutable
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source_profile: SourceProfile
    run_metadata: RunMetadata = Field(default_factory=RunMetadata)
    topcode_: dict[str, TopcodeConfig] = Field(default_factory=dict, alias="topcode")
    deflators_: dict[str, DeflatorTableConfig] = Field(
        default_factory=dict, alias="deflators"
    )
    crosswalks_: dict[str, pl.DataFrame] = Field(
        default_factory=dict, alias="crosswalks"
    )

    @classmethod
    def from_config(
        cls,
        config_dir: Path,
        source: SourceKind,
        crosswalks_dir: Path | None = None,
        overrides: dict[str, object] | None = None,
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
            overrides (dict[str, object] | None):
                Deep-merged onto `source_profile.yaml` before validation -
                reaches only `source_profile`, not `topcode` or `crosswalks`.

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

        # Load optional named deflator tables
        deflators: dict[str, DeflatorTableConfig] = {}
        deflators_dir = config_dir / "deflators"
        if deflators_dir.exists():
            for deflator_path in sorted(deflators_dir.glob("*.yaml")):
                raw_deflator = yaml.safe_load(deflator_path.read_text()) or {}
                try:
                    deflators[deflator_path.stem] = DeflatorTableConfig.model_validate(
                        raw_deflator
                    )
                except ValidationError as exc:
                    raise ValueError(
                        f"CleaningContext.from_config: {deflator_path} failed to "
                        f"validate as DeflatorTableConfig: {exc}"
                    ) from exc

        # Load optional crosswalk tables
        crosswalks: dict[str, pl.DataFrame] = {}
        if crosswalks_dir is not None and crosswalks_dir.exists():
            for crosswalk_path in sorted(crosswalks_dir.glob("*.csv")):
                crosswalks[crosswalk_path.stem] = pl.read_csv(crosswalk_path)

        return cls(
            source_profile=source_profile,
            topcode=topcode,
            deflators=deflators,
            crosswalks=crosswalks,
        )

    def compute_hash(self) -> str:
        """A stable hash of this context's methodology.

        - Excludes `run_id` (a run identifier, not a methodology).
        - `crosswalks` (arbitrary-type `pl.DataFrame` values) is dumped via
          each table's own row contents rather than the automatic JSON dump.
        - `available_flags` is a `frozenset`; its `model_dump(mode="json")`
          ordering follows `PYTHONHASHSEED` and is re-sorted here so the
          hash identifies the methodology, not the interpreter instance.
        """
        run_metadata = self.run_metadata.model_dump(mode="json")
        run_metadata.pop("run_id", None)
        source_profile = self.source_profile.model_dump(mode="json")
        # frozenset -> list ordering follows PYTHONHASHSEED; canonicalise so the
        # hash identifies the methodology, not the interpreter instance.
        source_profile["available_flags"] = sorted(source_profile["available_flags"])
        payload = {
            "source_profile": source_profile,
            "run_metadata": run_metadata,
            "topcode": {
                name: cfg.model_dump(mode="json")
                for name, cfg in sorted(self.topcode.items())
            },
            "deflators": {
                name: cfg.model_dump(mode="json", by_alias=True)
                for name, cfg in sorted(self.deflators.items())
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

    @property
    def topcode(self) -> Mapping[str, TopcodeConfig]:
        """Read-only view; the model itself stays fully pydantic-serializable."""
        return MappingProxyType(self.topcode_)

    @property
    def deflators(self) -> Mapping[str, DeflatorTableConfig]:
        """Read-only view; the model itself stays fully pydantic-serializable."""
        return MappingProxyType(self.deflators_)

    @property
    def crosswalks(self) -> Mapping[str, pl.DataFrame]:
        return MappingProxyType(self.crosswalks_)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` onto a copy of `base`."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
