"""Compile UniCache assignments into validated execution steps."""

from __future__ import annotations

from collections import defaultdict

from ..core.registry import CurrentStateOperator, KVRegistry
from ..core.schema import (
    AssignmentPlan,
    ExecutionPlan,
    ExecutionStep,
    KVSegment,
    RuntimeContext,
    SchedulerDryRun,
)


STAGE_ORDER = {
    "cache_update": 0,
    "pre_attention": 10,
    "attention": 20,
    "post_attention": 30,
    "denoise_step": 40,
    "decode_step": 50,
}


FAMILY_ORDER = {
    "protection": 0,
    "identity": 10,
    "eviction": 10,
    "quantization": 20,
    "reuse": 30,
    "observation": 90,
}


RUNTIME_RESOLVED = [
    "run_id",
    "task_type",
    "layer_idx",
    "phase",
    "branch",
    "cfg_branch",
    "batch_idx",
    "step_index",
    "total_steps",
    "decode_index",
    "key_type_ids",
    "protected_mask",
    "qkv_shape",
    "persistent_storage_handle",
    "physical_packed_layout",
]

RUNTIME_UNRESOLVED: list[str] = []


def compile_execution_plan(
    assignment_plan: AssignmentPlan,
    registry: KVRegistry,
    *,
    dry_run_contexts: list[RuntimeContext] | None = None,
) -> ExecutionPlan:
    segments = {segment.id: segment for segment in assignment_plan.segment_plan.segments}
    steps: list[ExecutionStep] = []
    warnings: list[str] = list(assignment_plan.warnings)
    for assignment in assignment_plan.assignments:
        segment = segments[assignment.segment_id]
        operator = registry.require_any_operator(assignment.operator)
        is_current_state_segment = segment.pattern_id == "current_vae_state"
        if is_current_state_segment and not isinstance(operator, CurrentStateOperator):
            raise ValueError(
                f"Persistent-KV operator {assignment.operator} cannot process current_vae_state"
            )
        if isinstance(operator, CurrentStateOperator) and not is_current_state_segment:
            raise ValueError(
                f"Current-state operator {assignment.operator} requires current_vae_state"
            )
        if operator.family not in FAMILY_ORDER:
            raise ValueError(
                f"Operator {assignment.operator} has unsupported pipeline family {operator.family!r}; "
                f"expected one of {sorted(FAMILY_ORDER)}"
            )
        capabilities = operator.capability_descriptor()
        capabilities.validate_static(operator=assignment.operator, stage=assignment.stage)
        params = operator.build_params(assignment.params)
        if assignment.scheduler is not None:
            registry.require_scheduler(assignment.scheduler.name).validate_params(assignment.scheduler.params)
        steps.append(
            ExecutionStep(
                id=f"{assignment.rule_id}:{assignment.segment_id}",
                segment_id=assignment.segment_id,
                operator=assignment.operator,
                operator_family=operator.family,
                pipeline_order=FAMILY_ORDER[operator.family],
                stage=assignment.stage,
                params=params,
                scheduler=assignment.scheduler,
                priority=assignment.priority,
                mode=assignment.mode,
                order=assignment.order,
                match=dict(assignment.match),
                hook_status="executable" if operator.executable else "plan_only",
                runtime_resolved=list(RUNTIME_RESOLVED),
                runtime_unresolved=list(RUNTIME_UNRESOLVED),
                capabilities=capabilities.to_dict(),
            )
        )
    _validate_pipeline_stage_order(steps)
    # Stage determines the hook point. Within one hook point, the pipeline
    # dependency is authoritative; priority only orders operators in the same
    # family. This prevents a high-priority quantizer from running before a
    # protection operator.
    steps.sort(
        key=lambda step: (
            STAGE_ORDER.get(step.stage, 999),
            step.pipeline_order,
            -step.priority,
            step.order,
        )
    )
    dry_runs = build_scheduler_dry_runs(steps, segments, registry, dry_run_contexts or [])
    hook_points = sorted({step.stage for step in steps}, key=lambda stage: STAGE_ORDER.get(stage, 999))
    return ExecutionPlan(
        steps=steps,
        hook_points=hook_points,
        scheduler_states={step.id: {} for step in steps if step.scheduler is not None},
        dry_runs=dry_runs,
        summaries={
            "num_segments": len(segments),
            "num_assignments": len(assignment_plan.assignments),
            "num_steps": len(steps),
            "num_conflicts": len(assignment_plan.conflicts),
        },
        warnings=warnings,
    )


def _validate_pipeline_stage_order(steps: list[ExecutionStep]) -> None:
    attention_stages = {"pre_attention", "attention", "post_attention"}
    grouped: dict[str, list[ExecutionStep]] = defaultdict(list)
    for step in steps:
        if step.stage in attention_stages:
            grouped[step.segment_id].append(step)
    for segment_id, rows in grouped.items():
        for earlier in rows:
            for later in rows:
                if STAGE_ORDER.get(earlier.stage, 999) >= STAGE_ORDER.get(later.stage, 999):
                    continue
                if earlier.pipeline_order > later.pipeline_order:
                    raise ValueError(
                        "Invalid pipeline order for segment "
                        f"{segment_id}: {earlier.operator}@{earlier.stage} would run before "
                        f"{later.operator}@{later.stage}"
                    )


def build_scheduler_dry_runs(
    steps: list[ExecutionStep],
    segments: dict[str, KVSegment],
    registry: KVRegistry,
    contexts: list[RuntimeContext],
) -> list[SchedulerDryRun]:
    rows: list[SchedulerDryRun] = []
    for step in steps:
        if step.scheduler is None:
            continue
        scheduler = registry.require_scheduler(step.scheduler.name)
        segment = segments[step.segment_id]
        for ctx in contexts:
            rows.append(
                SchedulerDryRun(
                    rule_id=step.id.split(":", 1)[0],
                    segment_id=step.segment_id,
                    scheduler=step.scheduler.name,
                    context=ctx,
                    resolved=scheduler.resolve(step.scheduler.params, ctx, segment),
                )
            )
    return rows
