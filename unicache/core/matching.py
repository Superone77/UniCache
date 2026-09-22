"""Runtime matching for compiled UniCache execution steps."""

from __future__ import annotations

from typing import Any

from .schema import ExecutionStep, KVSegment, RuntimeContext, WILDCARD


BRANCH_ALIASES = {
    "gen": "generation",
    "generation": "generation",
    "und": "understanding",
    "understanding": "understanding",
}


def normalize_branch(value: str | None) -> str | None:
    if value is None:
        return None
    return BRANCH_ALIASES.get(str(value), str(value))


def _choice_matches(spec: Any, value: Any) -> bool:
    if spec is None or spec == WILDCARD or spec == [WILDCARD]:
        return True
    if isinstance(spec, (list, tuple, set)):
        return value in spec or WILDCARD in spec
    return value == spec


def _range_matches(spec: Any, value: int | None) -> bool:
    if spec is None or spec == WILDCARD:
        return True
    if value is None:
        return False
    if isinstance(spec, int):
        return int(value) == int(spec)
    if isinstance(spec, (list, tuple, set)):
        return int(value) in {int(item) for item in spec}
    if not isinstance(spec, dict):
        return False
    if "include" in spec and int(value) not in {int(item) for item in spec["include"]}:
        return False
    if "exclude" in spec and int(value) in {int(item) for item in spec["exclude"]}:
        return False
    lower = spec.get("min", spec.get("start"))
    upper = spec.get("max", spec.get("end"))
    if lower is not None and int(value) < int(lower):
        return False
    if upper is not None and int(value) > int(upper):
        return False
    return True


def runtime_step_matches(
    step: ExecutionStep,
    segment: KVSegment,
    ctx: RuntimeContext,
    *,
    preflight: bool = False,
) -> bool:
    """Match a compiled step against current runtime context.

    ``segment.selector.branches`` describes cache ownership and is resolved by
    planning. ``runtime_branches`` describes the branch issuing the current
    attention query and is evaluated here.
    """

    runtime_match = dict(step.match.get("runtime", {}) or {})
    for key in (
        "task_type",
        "task_types",
        "layers",
        "phases",
        "runtime_branches",
        "cfg_branch",
        "batch",
        "step_range",
        "decode_range",
    ):
        if key in step.match:
            runtime_match[key] = step.match[key]

    task_spec = runtime_match.get("task_types", runtime_match.get("task_type", WILDCARD))
    if not _choice_matches(task_spec, ctx.task_type):
        return False
    if not _choice_matches(segment.selector.phases, ctx.phase):
        return False
    if not _choice_matches(runtime_match.get("phases", WILDCARD), ctx.phase):
        return False

    branch = normalize_branch(ctx.branch)
    selector_runtime_branches = segment.selector.runtime_branches
    normalized_selector_branches = (
        [normalize_branch(item) for item in selector_runtime_branches]
        if isinstance(selector_runtime_branches, list)
        else normalize_branch(selector_runtime_branches)
    )
    if not _choice_matches(normalized_selector_branches, branch):
        return False
    match_branches = runtime_match.get("runtime_branches", WILDCARD)
    normalized_match_branches = (
        [normalize_branch(item) for item in match_branches]
        if isinstance(match_branches, list)
        else normalize_branch(match_branches)
    )
    if not _choice_matches(normalized_match_branches, branch):
        return False

    if not _range_matches(segment.selector.layers, ctx.layer_idx):
        return False
    if not _range_matches(runtime_match.get("layers", WILDCARD), ctx.layer_idx):
        return False
    if not _range_matches(segment.selector.step_range, ctx.step_index):
        return False
    if not _range_matches(runtime_match.get("step_range", WILDCARD), ctx.step_index):
        return False
    if not _range_matches(runtime_match.get("decode_range", WILDCARD), ctx.decode_index):
        return False

    if not _choice_matches(segment.selector.cfg_branch, ctx.cfg_branch):
        return False
    if not _choice_matches(runtime_match.get("cfg_branch", WILDCARD), ctx.cfg_branch):
        return False
    if not preflight:
        if not _choice_matches(segment.selector.batch, ctx.batch_idx):
            return False
        if not _choice_matches(runtime_match.get("batch", WILDCARD), ctx.batch_idx):
            return False
    return True
