"""Static task-level operator routing from calibrated K90 values."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..core.registry import OperatorRouter
from ..core.schema import SegmentPlan


class K90ThresholdRouter(OperatorRouter):
    """Route sparse segments to H2O and dense segments to KIVI.

    Routing runs when a task plan is built. It therefore remains fixed within
    one task; a later task may build a new plan from a different K90 profile.
    """

    name = "k90_threshold"

    def expand(
        self,
        rule: dict[str, Any],
        segment_plan: SegmentPlan,
        metric_profiles: dict[str, Any],
    ) -> list[dict[str, Any]]:
        profile_name = str(rule.get("metric_profile", "default"))
        profile = dict(metric_profiles.get(profile_name, {}) or {})
        k90_values = dict(profile.get("k90", {}) or {})
        threshold = float(rule.get("threshold", 0.3))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("k90_threshold threshold must be in [0, 1]")
        known_segments = {segment.id for segment in segment_plan.segments}
        segment_ids = [str(item) for item in rule.get("segments", [])]
        if not segment_ids:
            raise ValueError(f"K90 routing rule {rule.get('id', '<unnamed>')} has no segments")

        common_match = dict(rule.get("match", {}) or {})
        budget_params = dict(rule.get("budget", {}) or {})
        budget_params.setdefault("budget_segments", list(segment_ids))
        observer_params = {
            key: value
            for key, value in budget_params.items()
            if key in {"step_chunk_size", "layer_chunk_size", "first_layer_separate"}
        }
        observer_params.update(dict(rule.get("observer", {}) or {}))
        expanded: list[dict[str, Any]] = []
        for segment_id in segment_ids:
            if segment_id not in known_segments:
                raise ValueError(f"K90 routing references unknown segment {segment_id}")
            if segment_id not in k90_values:
                raise ValueError(
                    f"K90 routing profile {profile_name!r} has no value for segment {segment_id}"
                )
            k90 = float(k90_values[segment_id])
            if not 0.0 <= k90 <= 1.0:
                raise ValueError(f"K90 for segment {segment_id} must be in [0, 1], got {k90}")
            branch = deepcopy(rule["sparse"] if k90 <= threshold else rule["dense"])
            match = dict(common_match)
            match["segment_id"] = segment_id
            scheduler = deepcopy(branch.get("scheduler"))
            if scheduler is not None:
                scheduler_params = dict(budget_params)
                scheduler_params.update(dict(scheduler.get("params", {}) or {}))
                scheduler_params["segment_id"] = segment_id
                scheduler["params"] = scheduler_params
            base_id = str(rule["id"])
            operator_params = dict(branch.get("params", {}) or {})
            operator_params.update(
                {
                    "routing_metric": "k90",
                    "routing_k90": k90,
                    "routing_threshold": threshold,
                    "routing_profile": profile_name,
                }
            )
            expanded.append(
                {
                    "id": f"{base_id}:{segment_id}:policy",
                    "match": match,
                    "operator": str(branch["operator"]),
                    "params": operator_params,
                    "scheduler": scheduler,
                    "priority": int(branch.get("priority", rule.get("priority", 0))),
                    "mode": str(branch.get("mode", rule.get("mode", "composable"))),
                    "stage": str(branch.get("stage", "attention")),
                }
            )
            expanded.append(
                {
                    "id": f"{base_id}:{segment_id}:metrics",
                    "match": match,
                    "operator": "block_attention_metrics",
                    "params": {
                        **observer_params,
                        "calibrated_k90": k90,
                        "routing_threshold": threshold,
                        "metric_profile": profile_name,
                    },
                    "priority": -100,
                    "mode": "composable",
                    "stage": "post_attention",
                }
            )
        return expanded
