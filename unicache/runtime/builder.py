"""Build UniCache plans for dry-run or executable runtime adapters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ..core.registry import KVRegistry, PlanningContext
from ..core.schema import (
    KVSegment,
    KVSelector,
    PlanBundle,
    ReferenceInfo,
    RuntimeContext,
    SegmentPlan,
    TaskRequest,
    jsonable,
)
from ..planning import compile_execution_plan, infer_task_type, resolve_assignments
from .defaults import build_default_registry


CLEAN_REF_ROOT = Path(__file__).resolve().parents[2] / "third_party"
REFERENCE_SPECS = {
    "bagel": {
        "url": "https://github.com/ByteDance-Seed/BAGEL.git",
        "key_files": ["modeling/bagel/bagel.py", "modeling/bagel/qwen2_navit.py", "inferencer.py"],
    },
    "h2o": {
        "url": "https://github.com/FMInference/H2O.git",
        "key_files": ["h2o_hf/utils_hh/modify_llama.py", "h2o_hf/README.md"],
    },
    "kvcache_factory": {
        "url": "https://github.com/Zefan-Cai/KVCache-Factory.git",
        "key_files": ["pyramidkv/pyramidkv_utils.py", "pyramidkv/cache_utils_think.py", "README.md"],
    },
    "pyramidkv": {
        "url": "https://github.com/IsaacRe/PyramidKV.git",
        "key_files": ["pyramidkv/pyramidkv_utils.py", "README.md"],
    },
    "kivi": {
        "url": "https://github.com/jy-yuan/KIVI.git",
        "key_files": ["quant/new_pack.py", "models/mistral_kivi.py", "README.md"],
    },
    "streamingllm": {
        "url": "https://github.com/mit-han-lab/streaming-llm.git",
        "key_files": ["streaming_llm/pos_shift/modify_llama.py", "README.md"],
    },
    "snapkv": {
        "url": "https://github.com/FasterDecoding/SnapKV.git",
        "key_files": ["snapkv/monkeypatch/snapkv_utils.py", "README.md"],
    },
    "duca": {
        "url": "https://github.com/Shenyi-Z/DuCa.git",
        "key_files": [
            "DuCa-DiT/cache_functions/cache_cutfresh.py",
            "DuCa-DiT/cache_functions/force_scheduler.py",
            "DuCa-DiT/cache_functions/scores.py",
        ],
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_references(root: Path = CLEAN_REF_ROOT) -> list[ReferenceInfo]:
    references: list[ReferenceInfo] = []
    for name, spec in REFERENCE_SPECS.items():
        repo = root / name
        commit = ""
        if (repo / ".git").exists():
            try:
                commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                commit = ""
        references.append(
            ReferenceInfo(
                name=name,
                url=spec["url"],
                commit=commit,
                path=str(repo),
                key_files=list(spec["key_files"]),
            )
        )
    return references


def _string_list(value: Any, *, field_name: str, group_id: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Segment group {group_id} {field_name} must be a string or list of strings")
    return list(value)


def _build_segment_groups(raw_groups: list[dict[str, Any]], atomic_segments: list[KVSegment]) -> list[KVSegment]:
    atomic_by_id = {segment.id: segment for segment in atomic_segments}
    atomic_ids = list(atomic_by_id)
    groups: list[KVSegment] = []
    seen_group_ids: set[str] = set()
    for raw in raw_groups:
        group_id = str(raw.get("id", "")).strip()
        if not group_id:
            raise ValueError("Segment group id must be non-empty")
        if group_id in atomic_by_id or group_id in seen_group_ids:
            raise ValueError(f"Duplicate segment id: {group_id}")
        seen_group_ids.add(group_id)

        include = _string_list(raw.get("include", ["*"]), field_name="include", group_id=group_id)
        exclude = set(_string_list(raw.get("exclude", []), field_name="exclude", group_id=group_id))
        explicit = [item for item in include if item != "*"]
        unknown = [item for item in explicit if item not in atomic_by_id]
        if unknown:
            raise ValueError(f"Segment group {group_id} references unknown segment(s): {', '.join(unknown)}")

        candidates = atomic_ids if "*" in include else explicit
        member_ids = []
        for segment_id in candidates:
            if segment_id not in exclude and segment_id not in member_ids:
                member_ids.append(segment_id)
        if not member_ids:
            raise ValueError(f"Segment group {group_id} has no members after exclusions")

        tags = ["segment_group"]
        for member_id in member_ids:
            for tag in atomic_by_id[member_id].tags:
                if tag not in tags:
                    tags.append(tag)
        groups.append(
            KVSegment(
                id=group_id,
                pattern_id="segment_group",
                label=str(raw.get("label", group_id)),
                selector=KVSelector(),
                tags=tags,
                lifecycle="composite",
                role="combined_operator_target",
                provenance={"member_segment_ids": list(member_ids)},
                materialization="logical_union",
                member_segment_ids=member_ids,
            )
        )
    return groups


def build_segment_plan(config: dict[str, Any], *, registry: KVRegistry | None = None) -> SegmentPlan:
    registry = registry or build_default_registry()
    missing_plugins = [name for name in config.get("plugins", []) if name not in registry.plugins]
    if missing_plugins:
        raise KeyError(
            "UniCache config requires unregistered plugin capabilities: " + ", ".join(sorted(missing_plugins))
        )
    request = TaskRequest.from_config(config)
    if request.task_type == "auto":
        request.task_type = infer_task_type(request)
    metadata = dict(config.get("metadata", {}) or {})
    references = collect_references()
    planners = []
    segments = []
    warnings = []
    for raw in config.get("planners", []):
        if not raw.get("enabled", True):
            continue
        planner_id = str(raw["id"])
        planner = registry.require_planner(planner_id)
        planner_segments = planner.plan(
            request,
            metadata,
            PlanningContext(planner_params=dict(raw.get("params", {}) or {}), references=references),
        )
        planners.append(planner_id)
        segments.extend(planner_segments)
        if not planner_segments:
            warnings.append(f"Planner {planner_id} produced no segments")
    segments.extend(_build_segment_groups(list(config.get("segment_groups", []) or []), segments))
    seen = set()
    for segment in segments:
        if segment.id in seen:
            raise ValueError(
                f"Duplicate segment id: {segment.id}. Segment ids must be unique across all enabled planners."
            )
        seen.add(segment.id)
    return SegmentPlan(request=request, planners=planners, segments=segments, warnings=warnings, references=references)


def build_plan_from_config(config: dict[str, Any], *, registry: KVRegistry | None = None) -> PlanBundle:
    registry = registry or build_default_registry()
    runtime_config = dict(config.get("runtime", {}) or {})
    segment_plan = build_segment_plan(config, registry=registry)
    raw_rules = list(config.get("rules", []) or [])
    metric_profiles = dict(config.get("metric_profiles", {}) or {})
    for routing_rule in list(config.get("routing_rules", []) or []):
        router = registry.require_router(str(routing_rule["router"]))
        raw_rules.extend(router.expand(routing_rule, segment_plan, metric_profiles))
    assignment_plan = resolve_assignments(
        segment_plan,
        raw_rules,
        fail_on_conflict=bool(runtime_config.get("fail_on_conflict", True)),
    )
    dry_contexts = [RuntimeContext.from_dict(raw) for raw in runtime_config.get("dry_run_contexts", [])]
    execution_plan = compile_execution_plan(assignment_plan, registry, dry_run_contexts=dry_contexts)
    execution_plan.summaries["routing_decisions"] = [
        {
            "segment_id": step.segment_id,
            "metric": step.params["routing_metric"],
            "k90": step.params["routing_k90"],
            "threshold": step.params["routing_threshold"],
            "profile": step.params["routing_profile"],
            "operator": step.operator,
        }
        for step in execution_plan.steps
        if step.params.get("routing_metric") == "k90"
    ]
    return PlanBundle(segment_plan=segment_plan, assignment_plan=assignment_plan, execution_plan=execution_plan)


def write_plan_bundle(bundle: PlanBundle, path: str | Path) -> None:
    Path(path).write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def print_plan_bundle(bundle: PlanBundle) -> None:
    print(json.dumps(jsonable(bundle), ensure_ascii=False, indent=2))
