"""A registry of valid cleaning steps.

An explicit mapping from cleaner types (used in pipeline config, e.g.,
pipeline.yaml) into actual Steps used by the Pipeline:

STEP_BUILDERS: name -> callable(name=..., **kwargs) -> Step, used by
`Pipeline.from_config()` to build a Step list from a YAML `steps:` block.
"""

from collections.abc import Callable

from src.cleaning.base import Step
from src.cleaning.steps.band_filter import BandFilter
from src.cleaning.steps.deflator_merge import DeflatorMergeStep
from src.cleaning.steps.derived_weights import DerivedWeightsStep
from src.cleaning.steps.function_step import _build_function_step
from src.cleaning.steps.membership_filter import MembershipFilter
from src.cleaning.steps.topcode_adjuster import TopcodeAdjuster
from src.cleaning.steps.topcode_cap import TopcodeCapFilter

STEP_BUILDERS: dict[str, Callable[..., Step]] = {
    "BandFilter": BandFilter,
    "MembershipFilter": MembershipFilter,
    "TopcodeCapStep": TopcodeCapFilter,
    "TopcodeAdjuster": TopcodeAdjuster,
    "DeflatorMergeStep": DeflatorMergeStep,
    "DerivedWeightsStep": DerivedWeightsStep,
    "FunctionStep": _build_function_step,
}
