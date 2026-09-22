"""Assignment matching and conflict handling."""

from __future__ import annotations

from typing import Any

from ..core.schema import (
    AssignmentConflict,
    AssignmentPlan,
    AssignmentRule,
    KVSegment,
    ResolvedAssignment,
    SegmentPlan,
    WILDCARD,
)


def _list_contains(values: list[str] | str, wanted: str) -> bool:
    return values == WILDCARD or wanted in values


def _tags_match(segment_tags: list[str], wanted: list[str] | str) -> bool:
    if wanted == WILDCARD or wanted == [WILDCARD]:
        return True
    return all(tag in segment_tags for tag in wanted)


def _layers_match(match_layers: Any) -> bool:
    # Layer filtering is resolved at runtime. Plan-level matching only rejects
    # explicit impossible specs; valid layer filters are kept symbolic.
    return match_layers is None or isinstance(match_layers, (dict, list, str, int))


def segment_matches(segment: KVSegment, match: dict[str, Any]) -> bool:
    if not match:
        return not segment.member_segment_ids
    # Feature-state segments are a different execution domain from persistent
    # K/V. Require an exact target so broad cache rules cannot cross domains.
    if segment.pattern_id == "current_vae_state" and match.get("segment_id") != segment.id:
        return False
    # A logical group is an explicit assignment target. Broad tag/type rules
    # continue to apply to planner-produced atomic segments without also
    # processing the same tokens a second time through every declared group.
    if segment.member_segment_ids and match.get("segment_id") != segment.id:
        return False
    if "segment_id" in match and match["segment_id"] not in {segment.id, WILDCARD}:
        return False
    if "label" in match and match["label"] not in {segment.label, WILDCARD}:
        return False
    if "pattern_id" in match and match["pattern_id"] not in {segment.pattern_id, WILDCARD}:
        return False
    if "tags" in match and not _tags_match(segment.tags, match["tags"]):
        return False
    if "lifecycle" in match and match["lifecycle"] not in {segment.lifecycle, WILDCARD}:
        return False
    if "role" in match and match["role"] not in {segment.role, WILDCARD}:
        return False
    if "kv_types" in match:
        wanted = match["kv_types"]
        segment_types = segment.selector.kv_types
        if wanted != WILDCARD and segment_types != WILDCARD:
            wanted_list = wanted if isinstance(wanted, list) else [wanted]
            if not any(_list_contains(segment_types, item) for item in wanted_list):
                return False
    if "branches" in match:
        wanted = match["branches"]
        segment_branches = segment.selector.branches
        if wanted != WILDCARD and segment_branches != WILDCARD:
            wanted_list = wanted if isinstance(wanted, list) else [wanted]
            if not any(_list_contains(segment_branches, item) for item in wanted_list):
                return False
    if "layers" in match and not _layers_match(match["layers"]):
        return False
    return True


def resolve_assignments(
    segment_plan: SegmentPlan,
    raw_rules: list[dict[str, Any]],
    *,
    fail_on_conflict: bool = True,
) -> AssignmentPlan:
    rules = [AssignmentRule.from_dict(raw, order=i) for i, raw in enumerate(raw_rules)]
    assignments: list[ResolvedAssignment] = []
    warnings: list[str] = []
    for rule in rules:
        matched = [segment for segment in segment_plan.segments if segment_matches(segment, rule.match)]
        if not matched:
            warnings.append(f"Rule {rule.id} matched no segments")
            continue
        for segment in matched:
            assignments.append(
                ResolvedAssignment(
                    rule_id=rule.id,
                    segment_id=segment.id,
                    operator=rule.operator,
                    stage=rule.stage,
                    params=dict(rule.params),
                    scheduler=rule.scheduler,
                    priority=rule.priority,
                    mode=rule.mode,
                    order=rule.order,
                    match=dict(rule.match),
                )
            )
    conflicts = detect_conflicts(assignments, segment_plan)
    if fail_on_conflict and conflicts:
        conflict_text = "; ".join(
            f"{c.segment_id}/{c.stage}: {c.rule_ids} ({c.reason})" for c in conflicts
        )
        raise ValueError(f"Assignment conflicts: {conflict_text}")
    return AssignmentPlan(
        segment_plan=segment_plan,
        rules=rules,
        assignments=assignments,
        conflicts=conflicts,
        warnings=warnings,
    )


def _segment_token_domains(segment: KVSegment) -> tuple[set[str], set[str] | None]:
    """Return explicit member ids and KV types used for overlap detection."""

    if segment.member_segment_ids:
        return set(segment.member_segment_ids), None
    kv_types = segment.selector.kv_types
    if kv_types == WILDCARD:
        return {segment.id}, None
    return {segment.id}, set(kv_types)


def _segments_overlap(left: KVSegment, right: KVSegment) -> bool:
    if left.pattern_id == "current_vae_state" or right.pattern_id == "current_vae_state":
        return left.id == right.id
    left_is_broad = not left.member_segment_ids and left.selector.kv_types == WILDCARD
    right_is_broad = not right.member_segment_ids and right.selector.kv_types == WILDCARD
    if left_is_broad or right_is_broad:
        return True
    left_ids, left_types = _segment_token_domains(left)
    right_ids, right_types = _segment_token_domains(right)
    if left_ids & right_ids:
        return True
    if left.member_segment_ids and right.id in left_ids:
        return True
    if right.member_segment_ids and left.id in right_ids:
        return True
    if left_types is None or right_types is None:
        # A wildcard selector can cover any persistent KV token domain.
        return not left.member_segment_ids and not right.member_segment_ids
    return bool(left_types & right_types)


def _runtime_choice_set(match: dict[str, Any], key: str) -> set[str] | None:
    runtime = dict(match.get("runtime", {}) or {})
    value = match.get(key, runtime.get(key, WILDCARD))
    if key == "task_types" and value == WILDCARD:
        value = match.get("task_type", runtime.get("task_type", WILDCARD))
    if value is None or value == WILDCARD or value == [WILDCARD]:
        return None
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def _runtime_scopes_overlap(left: ResolvedAssignment, right: ResolvedAssignment) -> bool:
    for key in ("task_types", "phases", "runtime_branches", "cfg_branch"):
        left_values = _runtime_choice_set(left.match, key)
        right_values = _runtime_choice_set(right.match, key)
        if left_values is not None and right_values is not None and not (left_values & right_values):
            return False
    return True


def detect_conflicts(
    assignments: list[ResolvedAssignment],
    segment_plan: SegmentPlan,
) -> list[AssignmentConflict]:
    segments = {segment.id: segment for segment in segment_plan.segments}
    conflicts: list[AssignmentConflict] = []
    for index, left in enumerate(assignments):
        for right in assignments[index + 1 :]:
            if left.stage != right.stage:
                continue
            if left.mode != "exclusive" and right.mode != "exclusive":
                continue
            if not _runtime_scopes_overlap(left, right):
                continue
            left_segment = segments[left.segment_id]
            right_segment = segments[right.segment_id]
            if not _segments_overlap(left_segment, right_segment):
                continue
            same_segment = left.segment_id == right.segment_id
            reason = (
                "exclusive assignment overlaps another assignment on same segment and stage"
                if same_segment
                else "exclusive assignments target overlapping segment token domains"
            )
            conflicts.append(
                AssignmentConflict(
                    segment_id=(
                        left.segment_id
                        if same_segment
                        else f"{left.segment_id}<->{right.segment_id}"
                    ),
                    stage=left.stage,
                    rule_ids=[left.rule_id, right.rule_id],
                    reason=reason,
                )
            )
    return conflicts
