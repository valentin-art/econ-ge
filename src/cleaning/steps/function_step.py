"""FunctionStep: wraps a plain Python function as a Step."""

import functools
import importlib
from collections.abc import Callable, Mapping

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext

_ALLOWED_FUNCTION_PREFIX = "src.cleaning.custom_functions."


class FunctionStep(Step):
    def __init__(
        self,
        name: str,
        fn: Callable[[pl.DataFrame, CleaningContext], pl.DataFrame],
        required_columns: frozenset[str],
        produced_columns: frozenset[str],
    ) -> None:
        super().__init__(name)
        self.fn = fn
        self.required_columns = required_columns
        self.produced_columns = produced_columns

    def apply(
        self, df: pl.DataFrame, context: CleaningContext
    ) -> tuple[pl.DataFrame, StepReport]:
        n_in = len(df)
        result = self.fn(df, context)
        n_out = len(result)

        dropped_reason_counts = {"function_step": n_in - n_out} if n_out < n_in else {}

        return result, StepReport(
            step_name=self.name,
            n_in=n_in,
            n_out=n_out,
            dropped_reason_counts=dropped_reason_counts,
        )


def _build_function_step(
    name: str,
    function: str,
    required_columns: list[str],
    produced_columns: list[str],
    params: Mapping[str, object] | None = None,
    allowed_prefixes: tuple[str, ...] = (_ALLOWED_FUNCTION_PREFIX,),
) -> FunctionStep:
    """Function Step."""
    module_path, _, attr_name = function.rpartition(".")
    if not module_path:
        raise ValueError(
            f"_build_function_step: {function!r} is not a valid dotted path "
            "(expected 'module.submodule.function_name')"
        )
    if not function.startswith(allowed_prefixes):
        raise ValueError(
            f"FunctionStep {name!r}: function={function!r} must live under "
            f"{_ALLOWED_FUNCTION_PREFIX}; arbitrary dotted paths are not "
            "resolvable from YAML"
        )
    try:
        module = importlib.import_module(module_path)
        fn = getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"_build_function_step: could not resolve {function!r}: {exc}"
        ) from exc

    if params:
        fn = functools.partial(fn, **params)

    return FunctionStep(
        name=name,
        fn=fn,
        required_columns=frozenset(required_columns),
        produced_columns=frozenset(produced_columns),
    )
