"""Pattern planners, assignment resolution, and execution compilation."""

from .assignment import detect_conflicts, resolve_assignments, segment_matches
from .execution import FAMILY_ORDER, STAGE_ORDER, build_scheduler_dry_runs, compile_execution_plan
from .planners import BagelTypedSegmentsPlanner, CurrentVAEStatePlanner, WholeConditioningKVPlanner, infer_task_type

__all__ = [
    "BagelTypedSegmentsPlanner",
    "CurrentVAEStatePlanner",
    "FAMILY_ORDER",
    "STAGE_ORDER",
    "WholeConditioningKVPlanner",
    "build_scheduler_dry_runs",
    "compile_execution_plan",
    "detect_conflicts",
    "infer_task_type",
    "resolve_assignments",
    "segment_matches",
]
