"""Built-in UniCache plugin registration."""

from __future__ import annotations

from ..core.registry import KVRegistry, KVRuntimePlugin
from ..operators import built_in_operators
from ..current_state import built_in_current_state_operators
from ..planning import BagelTypedSegmentsPlanner, CurrentVAEStatePlanner, WholeConditioningKVPlanner
from ..schedulers import built_in_schedulers
from ..routing import K90ThresholdRouter


class BuiltInPlugin(KVRuntimePlugin):
    name = "builtin"
    provides = {
        "bagel_typed",
        "duca",
        "full_kv",
        "global_kv",
        "h2o",
        "kivi",
        "observer",
        "pyramidkv",
        "snapkv",
        "streamingllm",
        "lifetime",
        "k90_routing",
    }

    def register(self, registry: KVRegistry) -> None:
        registry.add_pattern_planner(BagelTypedSegmentsPlanner())
        registry.add_pattern_planner(WholeConditioningKVPlanner())
        registry.add_pattern_planner(CurrentVAEStatePlanner())
        for operator in built_in_operators():
            registry.add_processing_operator(operator)
        for operator in built_in_current_state_operators():
            registry.add_current_state_operator(operator)
        for scheduler in built_in_schedulers():
            registry.add_budget_scheduler(scheduler)
        registry.add_operator_router(K90ThresholdRouter())


def build_default_registry() -> KVRegistry:
    registry = KVRegistry()
    registry.add_plugin(BuiltInPlugin())
    return registry
