"""FunctionStep: wraps a plain Python function as a Step."""

import functools
import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence

import polars as pl

from src.cleaning.base import Step, StepReport
from src.cleaning.context import CleaningContext

_ALLOWED_FUNCTION_PREFIX = "src.cleaning.custom_functions."

# A wrapped function may return either a bare DataFrame, or (df, warnings) so
# it can surface a silent-failure risk (e.g. an empty fit population) through
# the same StepReport.warnings channel every other step uses.
FunctionStepResult = pl.DataFrame | tuple[pl.DataFrame, Sequence[str]]


class FunctionStep(Step):
    """Wraps a plain `(df, context) -> df` (or `-> (df, warnings)`) function
    as a `Step`, so a one-off transformation doesn't need its own `Step`
    subclass. `dropped_reason_counts`/`warnings` are inferred generically
    from the row-count delta and any warnings the function returns - a
    function that needs a more specific report should still get its own
    `Step` subclass instead.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[pl.DataFrame, CleaningContext], FunctionStepResult],
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
        returned = self.fn(df, context)
        result, fn_warnings = (
            returned if isinstance(returned, tuple) else (returned, [])
        )
        if not isinstance(result, pl.DataFrame):
            raise TypeError(
                f"FunctionStep {self.name!r}: {self.fn} returned "
                f"{type(result).__name__}, expected a pl.DataFrame (or "
                "(pl.DataFrame, warnings))"
            )
        n_out = len(result)

        dropped_reason_counts = {"function_step": n_in - n_out} if n_out < n_in else {}
        warnings = list(fn_warnings)
        if n_out > n_in:
            warnings.append(
                f"{self.name} added {n_out - n_in} rows (n_in={n_in}, n_out={n_out}); "
                "check for join fan-out"
            )

        return result, StepReport(
            step_name=self.name,
            n_in=n_in,
            n_out=n_out,
            dropped_reason_counts=dropped_reason_counts,
            warnings=warnings,
        )


def _resolve_function_step(
    name: str,
    function: str,
    required_columns: list[str],
    produced_columns: list[str],
    params: Mapping[str, object] | None = None,
    *,
    allowed_prefixes: tuple[str, ...],
) -> FunctionStep:
    """Shared implementation behind `_build_function_step` (the
    STEP_BUILDERS-registered, YAML-facing builder, fixed to
    `_ALLOWED_FUNCTION_PREFIX`) and tests, which bind a wider allowlist via
    `functools.partial` to exercise dotted-path resolution against a
    fixtures module without weakening what a pipeline YAML can reach.
    """
    module_path, _, attr_name = function.rpartition(".")
    if not module_path:
        raise ValueError(
            f"FunctionStep {name!r}: {function!r} is not a valid dotted path "
            "(expected 'module.submodule.function_name')"
        )
    if not function.startswith(allowed_prefixes):
        raise ValueError(
            f"FunctionStep {name!r}: function={function!r} must live under "
            f"one of {allowed_prefixes}; arbitrary dotted paths are not "
            "resolvable from YAML"
        )
    try:
        module = importlib.import_module(module_path)
        fn = getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"FunctionStep {name!r}: could not resolve {function!r}: {exc}"
        ) from exc

    if not callable(fn):
        raise TypeError(
            f"FunctionStep {name!r}: {function!r} resolved to {type(fn).__name__}, which is not callable"
        )
    signature = inspect.signature(fn)
    if params:
        try:
            signature.bind_partial(**params)
        except TypeError as exc:
            raise ValueError(
                f"FunctionStep {name!r}: params {sorted(params)} do not match "
                f"{function}{signature}: {exc}"
            ) from exc
        fn = functools.partial(fn, **params)
    remaining = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(remaining) != 2:
        raise ValueError(
            f"FunctionStep {name!r}: {function} must accept (df, context) after "
            f"params are bound; got {signature}"
        )

    return FunctionStep(
        name=name,
        fn=fn,
        required_columns=frozenset(required_columns),
        produced_columns=frozenset(produced_columns),
    )


def _build_function_step(
    name: str,
    function: str,
    required_columns: list[str],
    produced_columns: list[str],
    params: Mapping[str, object] | None = None,
) -> FunctionStep:
    """Registered under `STEP_BUILDERS["FunctionStep"]`. Unlike every other
    constructor kwarg here, the allowlist is intentionally not a parameter
    of this function: `Pipeline.from_config` calls builders as
    `builder(**block)` with `block` taken straight from YAML, so if
    `allowed_prefixes` were settable here a pipeline config could widen its
    own import sandbox. A YAML block that includes an `allowed_prefixes`
    key fails loudly with a `TypeError` (wrapped as a `ValueError` by
    `Pipeline.from_config`) rather than silently taking effect.
    """
    return _resolve_function_step(
        name=name,
        function=function,
        required_columns=required_columns,
        produced_columns=produced_columns,
        params=params,
        allowed_prefixes=(_ALLOWED_FUNCTION_PREFIX,),
    )
