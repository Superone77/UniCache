"""Built-in pattern planners for UniCache."""

from __future__ import annotations

from typing import Any

from ..core.registry import PatternPlanner, PlanningContext
from ..core.schema import KVSegment, KVSelector, TaskRequest, WILDCARD


def _segment(
    *,
    segment_id: str,
    pattern_id: str,
    label: str,
    kv_types: list[str] | str,
    tags: list[str],
    lifecycle: str,
    role: str,
    branches: list[str] | str = WILDCARD,
    phases: list[str] | str = WILDCARD,
    provenance: dict[str, Any] | None = None,
) -> KVSegment:
    return KVSegment(
        id=segment_id,
        pattern_id=pattern_id,
        label=label,
        selector=KVSelector(kv_types=kv_types, branches=branches, phases=phases),
        tags=tags,
        lifecycle=lifecycle,
        role=role,
        provenance=provenance or {},
    )


class BagelTypedSegmentsPlanner(PatternPlanner):
    id = "bagel_typed_segments"
    owner_plugin = "bagel_typed"

    def plan(self, request: TaskRequest, metadata: dict[str, Any], ctx: PlanningContext) -> list[KVSegment]:
        task_type = request.task_type
        if task_type == "auto":
            task_type = infer_task_type(request)
        segments: list[KVSegment] = []
        segments.append(
            _segment(
                segment_id="instruction",
                pattern_id=self.id,
                label="instruction",
                kv_types=["instruction"],
                tags=["instruction", "conditioning", "text"],
                lifecycle="conditioning",
                role="task_instruction",
                branches=["understanding"],
                phases=["prefill", "text_decode", "denoise"],
                provenance={"task_type": task_type},
            )
        )
        segments.append(
            _segment(
                segment_id="boundary",
                pattern_id=self.id,
                label="boundary",
                kv_types=["boundary"],
                tags=["boundary", "protect_default"],
                lifecycle="boundary",
                role="separator_or_boundary",
                provenance={"task_type": task_type},
            )
        )
        if task_type in {"understanding", "editing"} and request.source_images:
            segments.append(
                _segment(
                    segment_id="source_vit",
                    pattern_id=self.id,
                    label="source_vit",
                    kv_types=["source_vit"],
                    tags=["source_vit", "conditioning", "image", "understanding_branch"],
                    lifecycle="conditioning",
                    role="source_image_visual_condition",
                    branches=["understanding"],
                    phases=["prefill", "text_decode", "denoise"],
                    provenance={"source_images": request.source_images},
                )
            )
        if task_type == "editing" and request.source_images:
            segments.append(
                _segment(
                    segment_id="source_vae",
                    pattern_id=self.id,
                    label="source_vae",
                    kv_types=["source_vae"],
                    tags=["source_vae", "conditioning", "image", "generation_branch"],
                    lifecycle="conditioning",
                    role="source_image_latent_condition",
                    branches=["generation"],
                    phases=["prefill", "denoise"],
                    provenance={"source_images": request.source_images},
                )
            )
        if task_type in {"text_to_image", "generation", "editing"}:
            segments.append(
                _segment(
                    segment_id="current_vae",
                    pattern_id=self.id,
                    label="current_vae",
                    kv_types=["current_vae"],
                    tags=["current_vae", "active_state", "generation_branch", "protect_default"],
                    lifecycle="active_state",
                    role="current_denoising_latent_state",
                    branches=["generation"],
                    phases=["denoise"],
                    provenance={"task_type": task_type},
                )
            )
        if task_type == "understanding":
            segments.append(
                _segment(
                    segment_id="decoded_text",
                    pattern_id=self.id,
                    label="decoded_text",
                    kv_types=["decoded_text"],
                    tags=["decoded_text", "output_state", "protect_recent"],
                    lifecycle="output_state",
                    role="generated_text_tokens",
                    branches=["understanding"],
                    phases=["text_decode"],
                    provenance={"task_type": task_type},
                )
            )
        return segments


class WholeConditioningKVPlanner(PatternPlanner):
    id = "whole_conditioning_kv"
    owner_plugin = "global_kv"

    def plan(self, request: TaskRequest, metadata: dict[str, Any], ctx: PlanningContext) -> list[KVSegment]:
        params = ctx.planner_params
        selector = KVSelector(
            kv_types=params.get("kv_types", WILDCARD),
            layers=params.get("layers", WILDCARD),
            branches=params.get("branches", WILDCARD),
            runtime_branches=params.get("runtime_branches", WILDCARD),
            phases=params.get("phases", ["prefill", "denoise", "text_decode"]),
            token_ranges=params.get("token_ranges", WILDCARD),
            step_range=params.get("step_range"),
            cfg_branch=params.get("cfg_branch", WILDCARD),
        )
        return [
            KVSegment(
                id=params.get("segment_id", self.id),
                pattern_id=self.id,
                label="whole_conditioning_kv",
                selector=selector,
                tags=["whole_kv", "conditioning"],
                lifecycle="conditioning",
                role="broad_conditioning_or_prompt_kv",
                provenance={"planner_params": params},
            )
        ]


class CurrentVAEStatePlanner(PatternPlanner):
    id = "current_vae_state"
    owner_plugin = "duca"

    def plan(self, request: TaskRequest, metadata: dict[str, Any], ctx: PlanningContext) -> list[KVSegment]:
        task_type = request.task_type if request.task_type != "auto" else infer_task_type(request)
        if task_type not in {"text_to_image", "generation", "editing"}:
            return []
        return [
            KVSegment(
                id="current_vae_state",
                pattern_id=self.id,
                label="current_vae_state",
                selector=KVSelector(
                    kv_types=["current_vae"],
                    branches=["generation"],
                    phases=["denoise"],
                    step_range=WILDCARD,
                ),
                tags=["current_vae_state", "active_state", "current_state_reuse"],
                lifecycle="active_state",
                role="current_denoising_state_for_reuse",
                provenance={"task_type": task_type},
            )
        ]


def infer_task_type(request: TaskRequest) -> str:
    if request.output_modality == "text":
        return "understanding"
    prompt = request.prompt.lower()
    edit_terms = ["edit", "change", "replace", "modify", "transform", "改", "换", "替换", "编辑", "修改"]
    if request.source_images and any(term in prompt for term in edit_terms):
        return "editing"
    if request.source_images and request.output_modality == "image":
        return "editing"
    return "text_to_image"
