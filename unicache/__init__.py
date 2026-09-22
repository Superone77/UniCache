"""Stable public API for the UniCache runtime framework."""

from .adapters import PlanOnlyHookPolicy, UniCacheHookPolicy, compile_plan_only_hook_policy, compile_unicache_hook_policy
from .core import (
    CacheStorageView,
    CapabilityDescriptor,
    PhysicalCacheLayout,
    PlanBundle,
    RuntimeContext,
    SelectionDecision,
    TaskRequest,
)
from .runtime import build_plan_from_config, load_config, write_plan_bundle

__all__ = [
    "PlanBundle",
    "PlanOnlyHookPolicy",
    "CacheStorageView",
    "CapabilityDescriptor",
    "PhysicalCacheLayout",
    "RuntimeContext",
    "SelectionDecision",
    "TaskRequest",
    "UniCacheHookPolicy",
    "build_plan_from_config",
    "compile_plan_only_hook_policy",
    "compile_unicache_hook_policy",
    "load_config",
    "write_plan_bundle",
]
