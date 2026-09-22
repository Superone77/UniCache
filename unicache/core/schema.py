"""Dataclasses and JSON helpers for the UniCache planning and execution IR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


WILDCARD = "*"


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    return value


def as_list(value: Any, default: list[Any] | None = None) -> list[Any] | str:
    if value is None:
        return [] if default is None else list(default)
    if value == WILDCARD:
        return WILDCARD
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class ReferenceInfo:
    name: str
    url: str
    commit: str
    path: str
    key_files: list[str] = field(default_factory=list)


@dataclass
class TaskRequest:
    task_type: str
    prompt: str = ""
    source_images: list[str] = field(default_factory=list)
    output_modality: str = "image"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TaskRequest":
        raw = config.get("task", config)
        return cls(
            task_type=str(raw.get("task_type", "auto")),
            prompt=str(raw.get("prompt", "")),
            source_images=list(raw.get("source_images", []) or []),
            output_modality=str(raw.get("output_modality", "image")),
            metadata=dict(raw.get("metadata", {}) or {}),
        )


@dataclass
class KVSelector:
    kv_types: list[str] | str = WILDCARD
    layers: dict[str, Any] | list[int] | str = WILDCARD
    # branches identifies the branch that produced/owns the cached KV. It is
    # segment metadata, not the branch issuing the current attention query.
    branches: list[str] | str = WILDCARD
    runtime_branches: list[str] | str = WILDCARD
    phases: list[str] | str = WILDCARD
    token_ranges: list[dict[str, int]] | str = WILDCARD
    step_range: dict[str, int] | str | None = None
    cfg_branch: str = WILDCARD
    batch: int | str = WILDCARD


@dataclass
class KVSegment:
    id: str
    pattern_id: str
    label: str
    selector: KVSelector
    tags: list[str] = field(default_factory=list)
    lifecycle: str | None = None
    role: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    materialization: str = "runtime_resolved"
    # Composite segments reference planner-produced atomic segments. Runtime
    # adapters materialize their mask as the union of active member masks.
    member_segment_ids: list[str] = field(default_factory=list)


@dataclass
class SegmentPlan:
    request: TaskRequest
    planners: list[str]
    segments: list[KVSegment]
    warnings: list[str] = field(default_factory=list)
    references: list[ReferenceInfo] = field(default_factory=list)


@dataclass
class SchedulerSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SchedulerSpec | None":
        if not raw:
            return None
        return cls(name=str(raw["name"]), params=dict(raw.get("params", {}) or {}))


@dataclass
class AssignmentRule:
    id: str
    match: dict[str, Any]
    operator: str
    params: dict[str, Any] = field(default_factory=dict)
    scheduler: SchedulerSpec | None = None
    priority: int = 0
    mode: str = "composable"
    stage: str = "attention"
    order: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any], order: int = 0) -> "AssignmentRule":
        return cls(
            id=str(raw["id"]),
            match=dict(raw.get("match", {}) or {}),
            operator=str(raw["operator"]),
            params=dict(raw.get("params", {}) or {}),
            scheduler=SchedulerSpec.from_dict(raw.get("scheduler")),
            priority=int(raw.get("priority", 0)),
            mode=str(raw.get("mode", "composable")),
            stage=str(raw.get("stage", "attention")),
            order=int(order),
        )


@dataclass
class ResolvedAssignment:
    rule_id: str
    segment_id: str
    operator: str
    stage: str
    params: dict[str, Any]
    scheduler: SchedulerSpec | None
    priority: int
    mode: str
    order: int
    match: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssignmentConflict:
    segment_id: str
    stage: str
    rule_ids: list[str]
    reason: str


@dataclass
class AssignmentPlan:
    segment_plan: SegmentPlan
    rules: list[AssignmentRule]
    assignments: list[ResolvedAssignment]
    conflicts: list[AssignmentConflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RuntimeContext:
    task_type: str
    run_id: str = "unbound"
    phase: str = "plan"
    layer_idx: int = 0
    total_layers: int = 28
    branch: str | None = None
    step_index: int | None = None
    total_steps: int | None = None
    decode_index: int | None = None
    cfg_branch: str = WILDCARD
    batch_idx: int = 0
    kv_metadata: dict[str, Any] = field(default_factory=dict)
    operator_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RuntimeContext":
        return cls(
            task_type=str(raw.get("task_type", "editing")),
            run_id=str(raw.get("run_id", "unbound")),
            phase=str(raw.get("phase", "plan")),
            layer_idx=int(raw.get("layer_idx", 0)),
            total_layers=int(raw.get("total_layers", 28)),
            branch=raw.get("branch"),
            step_index=raw.get("step_index"),
            total_steps=raw.get("total_steps"),
            decode_index=raw.get("decode_index"),
            cfg_branch=str(raw.get("cfg_branch", WILDCARD)),
            batch_idx=int(raw.get("batch_idx", 0)),
            kv_metadata=dict(raw.get("kv_metadata", {}) or {}),
            operator_state=dict(raw.get("operator_state", {}) or {}),
        )


@dataclass
class SchedulerDryRun:
    rule_id: str
    segment_id: str
    scheduler: str
    context: RuntimeContext
    resolved: dict[str, Any]


@dataclass
class ExecutionStep:
    id: str
    segment_id: str
    operator: str
    operator_family: str
    pipeline_order: int
    stage: str
    params: dict[str, Any]
    scheduler: SchedulerSpec | None = None
    priority: int = 0
    mode: str = "composable"
    order: int = 0
    match: dict[str, Any] = field(default_factory=dict)
    hook_status: str = "plan_only"
    runtime_resolved: list[str] = field(default_factory=list)
    runtime_unresolved: list[str] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    steps: list[ExecutionStep]
    hook_points: list[str]
    scheduler_states: dict[str, Any] = field(default_factory=dict)
    dry_runs: list[SchedulerDryRun] = field(default_factory=list)
    summaries: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlanBundle:
    segment_plan: SegmentPlan
    assignment_plan: AssignmentPlan
    execution_plan: ExecutionPlan

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)
