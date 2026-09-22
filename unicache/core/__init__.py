"""Core IR, runtime context, matching, and plugin interfaces."""

from .capabilities import CapabilityDescriptor
from .decisions import CurrentStateDecision, SelectionDecision
from .matching import normalize_branch, runtime_step_matches
from .constants import SUPPORTED_FAKE_QUANT_BITS
from .registry import (
    BudgetScheduler,
    CurrentStateOperator,
    KVRegistry,
    KVRuntimePlugin,
    OperatorExecutionContext,
    OperatorRouter,
    PatternPlanner,
    PlanningContext,
    ProcessingOperator,
)
from .schema import *
from .storage import CacheStorageView, PhysicalCacheLayout

__all__ = [
    "BudgetScheduler",
    "CacheStorageView",
    "CapabilityDescriptor",
    "CurrentStateDecision",
    "CurrentStateOperator",
    "KVRegistry",
    "KVRuntimePlugin",
    "OperatorExecutionContext",
    "OperatorRouter",
    "PatternPlanner",
    "PlanningContext",
    "PhysicalCacheLayout",
    "ProcessingOperator",
    "SelectionDecision",
    "SUPPORTED_FAKE_QUANT_BITS",
    "normalize_branch",
    "runtime_step_matches",
]
