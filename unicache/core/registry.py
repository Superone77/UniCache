"""Registries and base classes for UniCache plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .capabilities import CapabilityDescriptor
from .decisions import CurrentStateDecision, PhysicalCacheRequest, SelectionDecision
from .schema import ExecutionStep, KVSegment, RuntimeContext, SegmentPlan, TaskRequest
from .storage import CacheStorageView


@dataclass
class PlanningContext:
    planner_params: dict[str, Any]
    references: list[Any]


class PatternPlanner:
    id: str = ""
    owner_plugin: str | None = None

    def plan(self, request: TaskRequest, metadata: dict[str, Any], ctx: PlanningContext) -> list[KVSegment]:
        raise NotImplementedError


class ProcessingOperator:
    name: str = ""
    family: str = "generic"
    stages: set[str] = {"attention"}
    executable: bool = False
    capabilities: CapabilityDescriptor | None = None
    uses_final_attention: bool = False
    preserves_attention_dispatch: bool = False

    def capability_descriptor(self) -> CapabilityDescriptor:
        if self.capabilities is not None:
            return self.capabilities
        return CapabilityDescriptor(stages=frozenset(self.stages))

    def supports(self, segment: KVSegment, ctx: RuntimeContext) -> bool:
        return True

    def build_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)

    def execute(self, ctx: "OperatorExecutionContext") -> "OperatorExecutionContext":
        raise NotImplementedError(f"Operator {self.name} has no executable runtime implementation")

    def update_after_attention(
        self,
        ctx: "OperatorExecutionContext",
        attention_probs: Any,
    ) -> "OperatorExecutionContext":
        del attention_probs
        return ctx


class CurrentStateOperator:
    """Registered policy for feature reuse across denoising timesteps."""

    name: str = ""
    family: str = "reuse"
    stages: set[str] = {"denoise_step"}
    executable: bool = True
    capabilities: CapabilityDescriptor | None = None

    def capability_descriptor(self) -> CapabilityDescriptor:
        if self.capabilities is not None:
            return self.capabilities
        return CapabilityDescriptor(stages=frozenset(self.stages), stateful=True)

    def build_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)

    def begin_run(
        self,
        params: dict[str, Any],
        ctx: RuntimeContext,
        state: dict[str, Any],
    ) -> CurrentStateDecision:
        raise NotImplementedError

    def generation_kwargs(self, decision: CurrentStateDecision) -> dict[str, Any]:
        raise NotImplementedError

    def on_step(
        self,
        params: dict[str, Any],
        ctx: RuntimeContext,
        state: dict[str, Any],
    ) -> CurrentStateDecision:
        """Resolve a layer/step/CFG-specific current-state decision."""

        return self.begin_run(params, ctx, state)


@dataclass
class OperatorExecutionContext:
    """Tensor and runtime state passed to a registered processing operator.

    Operators may replace ``k``, ``v``, and ``protected_mask``. They may also
    contribute a per-head key keep mask or a complete attention-probability
    override. Query tensors, model attention masks/probabilities, and attention
    output are read-only inputs.
    """

    runtime: RuntimeContext
    step: ExecutionStep
    params: dict[str, Any]
    q: Any
    k: Any
    v: Any
    attn_mask: Any
    key_type_ids: Any
    segment_mask: Any
    protected_mask: Any
    state: dict[Any, Any]
    record: Callable[[dict[str, float]], None]
    set_storage_snapshot: Callable[[dict[str, float]], None]
    metadata_state: dict[Any, Any] | None = None
    storage_num_kv_heads: int | None = None
    attention_probs: Any = None
    attention_prob_sources: Any = None
    attention_probs_override: Any = None
    attention_output: Any = None
    storage: CacheStorageView | None = None
    decision: SelectionDecision | None = None
    physical_request: PhysicalCacheRequest | None = None
    attention_keep_mask_override: Any = None


class BudgetScheduler:
    name: str = ""

    def validate_params(self, params: dict[str, Any]) -> None:
        return None

    def resolve(self, params: dict[str, Any], ctx: RuntimeContext, segment: KVSegment | None = None) -> dict[str, Any]:
        raise NotImplementedError


class OperatorRouter:
    """Expand a task-level routing rule into ordinary assignment rules."""

    name: str = ""

    def expand(
        self,
        rule: dict[str, Any],
        segment_plan: SegmentPlan,
        metric_profiles: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class KVRuntimePlugin:
    name: str = ""
    provides: set[str] = set()

    def register(self, registry: "KVRegistry") -> None:
        raise NotImplementedError


class KVRegistry:
    def __init__(self) -> None:
        self.pattern_planners: dict[str, PatternPlanner] = {}
        self.processing_operators: dict[str, ProcessingOperator] = {}
        self.current_state_operators: dict[str, CurrentStateOperator] = {}
        self.budget_schedulers: dict[str, BudgetScheduler] = {}
        self.operator_routers: dict[str, OperatorRouter] = {}
        self.plugins: dict[str, KVRuntimePlugin] = {}

    def add_plugin(self, plugin: KVRuntimePlugin) -> None:
        capabilities = {plugin.name, *set(plugin.provides)}
        for capability in capabilities:
            existing = self.plugins.get(capability)
            if existing is not None and existing is not plugin:
                raise ValueError(
                    f"Plugin capability {capability!r} is already provided by {existing.name!r}"
                )
            self.plugins[capability] = plugin
        plugin.register(self)

    def add_pattern_planner(self, planner: PatternPlanner) -> None:
        if not planner.id:
            raise ValueError("PatternPlanner.id must be non-empty")
        self.pattern_planners[planner.id] = planner

    def add_processing_operator(self, operator: ProcessingOperator) -> None:
        if not operator.name:
            raise ValueError("ProcessingOperator.name must be non-empty")
        self.processing_operators[operator.name] = operator

    def add_current_state_operator(self, operator: CurrentStateOperator) -> None:
        if not operator.name:
            raise ValueError("CurrentStateOperator.name must be non-empty")
        if operator.name in self.processing_operators:
            raise ValueError(f"Operator {operator.name!r} is already registered as a processing operator")
        self.current_state_operators[operator.name] = operator

    def add_budget_scheduler(self, scheduler: BudgetScheduler) -> None:
        if not scheduler.name:
            raise ValueError("BudgetScheduler.name must be non-empty")
        self.budget_schedulers[scheduler.name] = scheduler

    def add_operator_router(self, router: OperatorRouter) -> None:
        if not router.name:
            raise ValueError("OperatorRouter.name must be non-empty")
        self.operator_routers[router.name] = router

    def require_planner(self, planner_id: str) -> PatternPlanner:
        if planner_id not in self.pattern_planners:
            raise KeyError(f"Unknown pattern planner: {planner_id}")
        return self.pattern_planners[planner_id]

    def require_operator(self, operator_name: str) -> ProcessingOperator:
        if operator_name not in self.processing_operators:
            raise KeyError(f"Unknown processing operator: {operator_name}")
        return self.processing_operators[operator_name]

    def require_current_state_operator(self, operator_name: str) -> CurrentStateOperator:
        if operator_name not in self.current_state_operators:
            raise KeyError(f"Unknown current-state operator: {operator_name}")
        return self.current_state_operators[operator_name]

    def require_any_operator(self, operator_name: str) -> ProcessingOperator | CurrentStateOperator:
        if operator_name in self.processing_operators:
            return self.processing_operators[operator_name]
        return self.require_current_state_operator(operator_name)

    def require_scheduler(self, scheduler_name: str) -> BudgetScheduler:
        if scheduler_name not in self.budget_schedulers:
            raise KeyError(f"Unknown budget scheduler: {scheduler_name}")
        return self.budget_schedulers[scheduler_name]

    def require_router(self, router_name: str) -> OperatorRouter:
        if router_name not in self.operator_routers:
            raise KeyError(f"Unknown operator router: {router_name}")
        return self.operator_routers[router_name]
