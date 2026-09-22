"""BAGEL hook adapters for UniCache execution plans."""

from __future__ import annotations

import math
import os
import uuid
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

import torch

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:  # pragma: no cover - CUDA-only optional dependency
    flash_attn_func = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None

from ..core.matching import normalize_branch, runtime_step_matches
from ..core.registry import CurrentStateOperator, KVRegistry, OperatorExecutionContext, ProcessingOperator
from ..core.schema import ExecutionPlan, ExecutionStep, PlanBundle, RuntimeContext, WILDCARD, jsonable
from ..core.storage import CacheStorageView
from ..operators import (
    asymmetric_fake_quant as _asymmetric_fake_quant,
    kivi_fake_quant_key as _kivi_fake_quant_key,
    kivi_fake_quant_value as _kivi_fake_quant_value,
)
from ..runtime.defaults import build_default_registry
from ..storage import (
    CacheMutationResult,
    GQAH2OCache,
    PackedKIVICache,
    TensorCacheStorageBackend,
    gqa_probability_value,
    gqa_query_key_logits,
)


KV_TYPE_NAME_TO_ID = {
    "unknown": 0,
    "instruction": 1,
    "source_vit": 2,
    "source_vae": 3,
    "current_vae": 4,
    "boundary": 5,
    "decoded_text": 6,
}
KV_TYPE_ID_TO_NAME = {value: key for key, value in KV_TYPE_NAME_TO_ID.items()}

ATTENTION_HOOK_STAGES = {"pre_attention", "attention", "post_attention"}


def merge_attention_summaries(
    summaries: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized segment outputs using their per-row log-sum-exp."""

    if not summaries:
        raise ValueError("At least one attention summary is required")
    total_lse = torch.logsumexp(
        torch.stack([logsumexp.float() for _, logsumexp in summaries], dim=0),
        dim=0,
    )
    output = torch.zeros_like(summaries[0][0], dtype=torch.float32)
    for segment_output, segment_lse in summaries:
        weight = torch.exp(segment_lse.float() - total_lse).transpose(0, 1).unsqueeze(-1)
        output = output + segment_output.float() * weight
    return output.to(summaries[0][0].dtype), total_lse


def flash_attention_summary(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlashAttention while returning LSE without materializing S_dmask."""

    if _flash_attn_forward is None:
        raise RuntimeError("FlashAttention low-level forward is unavailable")
    output, logsumexp, _, _ = _flash_attn_forward(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        0.0,
        query.shape[-1] ** -0.5,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
    )
    return output.squeeze(0), logsumexp.squeeze(0)


@dataclass
class PlanOnlyHookPolicy:
    execution_plan: ExecutionPlan
    enabled: bool = False
    phase: str = "idle"
    step_index: int = 0
    total_steps: int = 1
    context_history: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = "unbound"
    efficiency_recorder: Any = None

    def set_context(self, *, phase=None, step_index=None, total_steps=None, **kwargs: Any) -> None:
        if phase is not None:
            self.phase = str(phase)
        if step_index is not None:
            self.step_index = int(step_index)
        if total_steps is not None:
            self.total_steps = max(1, int(total_steps))
        self.context_history.append(
            {"phase": self.phase, "step_index": self.step_index, "total_steps": self.total_steps}
        )
        if self.efficiency_recorder is not None and (
            phase is not None or step_index is not None
        ):
            self.efficiency_recorder.mark_step(
                phase=self.phase,
                step_index=self.step_index,
                run_id=self.run_id,
                branch=kwargs.get("branch"),
            )

    def begin_run(self, run_id: str | None = None) -> str:
        self.run_id = run_id or f"plan-only-{uuid.uuid4().hex}"
        return self.run_id

    def attach_efficiency_recorder(
        self, recorder: Any, *, instrument_attention: bool = False
    ) -> None:
        del instrument_attention
        self.efficiency_recorder = recorder

    def finalize_efficiency_step(self) -> None:
        if self.efficiency_recorder is not None:
            self.efficiency_recorder.finish_step()

    def should_apply(self, layer_idx: int, mode: str | None) -> bool:
        return False

    def current_state_generation_kwargs(self) -> dict[str, Any]:
        return {}

    def current_state_runtime_params(self, *, layer_idx: int) -> dict[str, Any]:
        del layer_idx
        return {}

    def capture_current_state_model_stats(self, model: Any) -> None:
        del model

    def summary(self) -> dict[str, Any]:
        return {
            "adapter": "PlanOnlyHookPolicy",
            "enabled": self.enabled,
            "phase": self.phase,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "hook_points": list(self.execution_plan.hook_points),
            "num_steps": len(self.execution_plan.steps),
            "runtime_resolved": sorted({item for step in self.execution_plan.steps for item in step.runtime_resolved}),
            "runtime_unresolved": sorted({item for step in self.execution_plan.steps for item in step.runtime_unresolved}),
            "context_history": list(self.context_history),
        }


@dataclass
class UniCacheHookPolicy:
    """Execute a compiled UniCache attention plan on BAGEL tensors.

    Operators run through the registry. Selection operators may update the
    current attention view or submit a decision to the persistent tensor cache
    backend. Physical H2O compaction and packed KIVI storage use explicit
    layouts supplied by that backend.
    """

    bundle: PlanBundle
    enabled: bool = True
    registry: KVRegistry | None = None
    phase: str = "idle"
    step_index: int = 0
    total_steps: int = 1
    total_layers: int = 28
    branch: str | None = None
    cfg_branch: str = "main"
    cfg_branches: tuple[str, ...] | None = None
    decode_index: int | None = None
    run_id: str = "unbound"
    context_history: list[dict[str, Any]] = field(default_factory=list)
    hh_scores: dict[tuple[Any, ...], torch.Tensor] = field(default_factory=dict)
    h2o_states: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    quantization_states: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    topk_state: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    classic_states: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    lifetime_states: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    block_metric_states: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    block_summary_states: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    current_state_states: dict[str, Any] = field(default_factory=dict)
    current_state_attention_scores: dict[tuple[Any, ...], torch.Tensor] = field(default_factory=dict)
    current_state_decision: Any = None
    current_state_model_stats: dict[str, Any] = field(default_factory=dict)
    current_state_step_decisions: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    stats: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    storage_snapshots: dict[str, dict[str, float]] = field(default_factory=dict)
    budget_allocation_records: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    runtime_warnings: list[str] = field(default_factory=list)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    tensor_shape_trace: list[dict[str, Any]] = field(default_factory=list)
    efficiency_recorder: Any = None
    instrument_attention: bool = False
    collect_diagnostics: bool = True
    pending_storage_decisions: dict[tuple[Any, ...], list[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pending_physical_requests: dict[tuple[Any, ...], list[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )
    realized_storage_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    coverage_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    coverage_type_counts: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    coverage_unknown_ids: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    coverage_rules: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    static_fastpath_cache: dict[tuple[Any, ...], str] = field(default_factory=dict)
    should_apply_cache: dict[tuple[Any, ...], bool] = field(default_factory=dict)
    mixed_layout_cache: dict[tuple[int, ...], tuple[Any, ...]] = field(
        default_factory=dict
    )
    mixed_workspace_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    mixed_auxiliary_streams: dict[tuple[str, int | None], Any] = field(
        default_factory=dict
    )
    physical_varlen_cu_cache: dict[tuple[Any, ...], torch.Tensor] = field(
        default_factory=dict
    )
    physical_varlen_span_cache: dict[tuple[Any, ...], tuple[list[int], list[int]]] = field(
        default_factory=dict
    )
    segment_count_cache: dict[tuple[Any, ...], int] = field(default_factory=dict)
    segment_span_cache: dict[tuple[Any, ...], tuple[int, int] | None] = field(
        default_factory=dict
    )
    segment_type_ids: dict[str, int] = field(default_factory=dict, init=False)
    max_context_history: int = 512
    accepts_storage_kv_metadata: bool = field(default=True, init=False)
    accepts_physical_cache_metadata: bool = field(default=True, init=False)
    accepts_native_gqa: bool = field(default=True, init=False)

    def attach_efficiency_recorder(
        self, recorder: Any, *, instrument_attention: bool = False
    ) -> None:
        self.efficiency_recorder = recorder
        self.instrument_attention = bool(instrument_attention)

    def _measure(self, name: str, runtime: RuntimeContext):
        if self.efficiency_recorder is None or not self.instrument_attention:
            return nullcontext()
        return self.efficiency_recorder.measure(
            name,
            phase=runtime.phase,
            branch=runtime.branch,
            cfg_branch=runtime.cfg_branch,
            step_index=int(runtime.step_index or 0),
            layer_idx=int(runtime.layer_idx),
            sample_idx=int(runtime.batch_idx),
        )

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = build_default_registry()
        self.segments_by_id = {segment.id: segment for segment in self.bundle.segment_plan.segments}
        unsupported = []
        attention_steps = []
        current_state_steps = []
        for step in self.bundle.execution_plan.steps:
            operator = self.registry.require_any_operator(step.operator)
            if step.stage not in operator.stages:
                raise ValueError(
                    f"Operator {step.operator} does not support stage {step.stage}; "
                    f"supported stages are {sorted(operator.stages)}"
                )
            if isinstance(operator, CurrentStateOperator):
                if step.stage != "denoise_step":
                    raise ValueError(
                        f"Current-state operator {step.operator} must run at denoise_step, got {step.stage}"
                    )
                current_state_steps.append(step)
            elif isinstance(operator, ProcessingOperator) and step.stage in ATTENTION_HOOK_STAGES:
                attention_steps.append(step)
            else:
                raise ValueError(
                    f"BAGEL adapter does not implement hook point {step.stage} "
                    f"for operator {step.operator}"
                )
            if not operator.executable:
                unsupported.append(step.operator)
        if unsupported:
            names = ", ".join(sorted(set(unsupported)))
            raise NotImplementedError(f"Executable UniCache runtime does not implement operator(s): {names}")
        if len(current_state_steps) > 1:
            names = ", ".join(step.operator for step in current_state_steps)
            raise ValueError(f"Only one current_vae_state operator may be active per run: {names}")
        self.steps = attention_steps
        self.current_state_steps = current_state_steps
        for segment in self.bundle.segment_plan.segments:
            members = (
                [self.segments_by_id[item] for item in segment.member_segment_ids]
                if segment.member_segment_ids
                else [segment]
            )
            type_names = {
                name
                for member in members
                if member.selector.kv_types != WILDCARD
                for name in member.selector.kv_types
                if name in KV_TYPE_NAME_TO_ID
            }
            if len(type_names) == 1:
                self.segment_type_ids[segment.id] = KV_TYPE_NAME_TO_ID[
                    next(iter(type_names))
                ]
        self.attention_match_progress_invariant = (
            not self.current_state_steps
            and all(self._step_match_progress_invariant(step) for step in self.steps)
        )
        self.attention_passthrough_only = all(
            self.registry.require_operator(step.operator).preserves_attention_dispatch
            for step in self.steps
        )
        self.score_query_count = 0
        self.kernel_size = 1
        self.pooling = "none"
        self.kivi_attention_backend = os.environ.get(
            "UNICACHE_KIVI_ATTENTION_BACKEND", "chunked_flash"
        ).lower()
        if self.kivi_attention_backend not in {
            "chunked_flash",
            "chunked_flash_overlap",
            "chunked_flash_dual_stream",
            "triton_fused",
        }:
            raise ValueError(
                f"Unknown KIVI attention backend: {self.kivi_attention_backend}"
            )
        self.kivi_flash_chunk_size = max(
            128, int(os.environ.get("UNICACHE_KIVI_FLASH_CHUNK_SIZE", "8192"))
        )
        self.batch_physical_attention = (
            os.environ.get("UNICACHE_BATCH_PHYSICAL_ATTENTION", "0") == "1"
        )
        self.overlap_batch_dequant = (
            os.environ.get("UNICACHE_OVERLAP_BATCH_DEQUANT", "0") == "1"
        )
        self._audit_mask_consumers()

    def _audit_mask_consumers(self) -> None:
        for segment_id in {step.segment_id for step in self.steps if step.operator == "heavy_hitter_protect"}:
            consumers = [
                step
                for step in self.steps
                if step.segment_id == segment_id
                and (
                    self.registry.require_operator(step.operator).family == "eviction"
                    or (
                        self.registry.require_operator(step.operator).family == "quantization"
                        and bool(step.params.get("preserve_protected", True))
                    )
                )
            ]
            if not consumers:
                self.runtime_warnings.append(
                    f"heavy_hitter_protect on {segment_id} is mask-only: no downstream operator consumes protected_mask"
                )

    def begin_run(self, run_id: str | None = None) -> str:
        self.run_id = str(run_id or f"unicache-{uuid.uuid4().hex}")
        self.hh_scores.clear()
        self.h2o_states.clear()
        self.quantization_states.clear()
        self.topk_state.clear()
        self.classic_states.clear()
        self.lifetime_states.clear()
        self.block_metric_states.clear()
        self.block_summary_states.clear()
        self.current_state_states.clear()
        self.current_state_attention_scores.clear()
        self.current_state_decision = None
        self.current_state_model_stats.clear()
        self.current_state_step_decisions.clear()
        self.stats.clear()
        self.storage_snapshots.clear()
        self.budget_allocation_records.clear()
        self.context_history.clear()
        self.decision_history.clear()
        self.tensor_shape_trace.clear()
        self.pending_storage_decisions.clear()
        self.pending_physical_requests.clear()
        self.realized_storage_totals.clear()
        self.coverage_totals.clear()
        self.coverage_type_counts.clear()
        self.coverage_unknown_ids.clear()
        self.coverage_rules.clear()
        self.static_fastpath_cache.clear()
        self.should_apply_cache.clear()
        self.mixed_layout_cache.clear()
        self.physical_varlen_span_cache.clear()
        self.segment_count_cache.clear()
        self.segment_span_cache.clear()
        self.phase = "idle"
        self.step_index = 0
        self.total_steps = 1
        self.branch = None
        self.cfg_branch = "main"
        self.cfg_branches = None
        self.decode_index = None
        return self.run_id

    def set_context(
        self,
        *,
        phase=None,
        step_index=None,
        total_steps=None,
        branch=None,
        cfg_branch=None,
        cfg_branches=None,
        decode_index=None,
        run_id=None,
    ) -> None:
        if run_id is not None and str(run_id) != self.run_id:
            self.begin_run(str(run_id))
        if phase is not None:
            self.phase = str(phase)
        if step_index is not None:
            self.step_index = int(step_index)
        if total_steps is not None:
            self.total_steps = max(1, int(total_steps))
        if branch is not None:
            self.branch = normalize_branch(str(branch))
        if cfg_branch is not None:
            self.cfg_branch = str(cfg_branch)
            if cfg_branches is None:
                self.cfg_branches = None
        if cfg_branches is not None:
            values = tuple(str(item) for item in cfg_branches)
            self.cfg_branches = values or None
        if decode_index is not None:
            self.decode_index = int(decode_index)
        row = {
            "run_id": self.run_id,
            "phase": self.phase,
            "branch": self.branch,
            "cfg_branch": self.cfg_branch,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "decode_index": self.decode_index,
        }
        if self.collect_diagnostics:
            self.context_history.append(row)
        if self.efficiency_recorder is not None and (
            phase is not None or step_index is not None
        ):
            self.efficiency_recorder.mark_step(
                phase=self.phase,
                step_index=self.step_index,
                run_id=self.run_id,
                branch=self.branch,
            )
        if self.collect_diagnostics and len(self.context_history) > self.max_context_history:
            self.context_history = self.context_history[-self.max_context_history :]

    def finalize_efficiency_step(self) -> None:
        if self.efficiency_recorder is not None:
            self.efficiency_recorder.finish_step()

    def _runtime_context(self, *, layer_idx: int, mode: str | None, sample_idx: int) -> RuntimeContext:
        cfg_branch = self.cfg_branch
        if self.cfg_branches is not None:
            if sample_idx >= len(self.cfg_branches):
                raise ValueError("Packed CFG branch metadata is shorter than the sample batch")
            cfg_branch = self.cfg_branches[sample_idx]
        return RuntimeContext(
            task_type=self.bundle.segment_plan.request.task_type,
            run_id=self.run_id,
            phase=self.phase,
            layer_idx=int(layer_idx),
            total_layers=self.total_layers,
            branch=normalize_branch(mode) or self.branch,
            step_index=self.step_index,
            total_steps=self.total_steps,
            decode_index=self.decode_index,
            cfg_branch=cfg_branch,
            batch_idx=int(sample_idx),
            operator_state={"block_metrics": self.block_metric_states},
        )

    def should_apply(self, layer_idx: int, mode: str | None) -> bool:
        if not self.enabled:
            return False
        cache_key = None
        if self.attention_match_progress_invariant:
            cache_key = (
                self.run_id,
                self.phase,
                normalize_branch(mode) or self.branch,
                self.cfg_branch,
                int(layer_idx),
            )
            if cache_key in self.should_apply_cache:
                return self.should_apply_cache[cache_key]
        ctx = self._runtime_context(layer_idx=layer_idx, mode=mode, sample_idx=0)
        has_attention_step = any(
            self._runtime_step_matches(step, ctx, preflight=True)
            for step in self.steps
        )
        result = has_attention_step or self._needs_current_state_attention(ctx)
        if cache_key is not None:
            self.should_apply_cache[cache_key] = result
        return result

    def _step_match_progress_invariant(self, step: ExecutionStep) -> bool:
        """Return whether runtime matching cannot change across decode/denoise progress."""

        runtime_match = dict(step.match.get("runtime", {}) or {})
        for name in ("step_range", "decode_range"):
            direct = step.match.get(name, WILDCARD)
            nested = runtime_match.get(name, WILDCARD)
            if (direct is not None and direct != WILDCARD) or (
                nested is not None and nested != WILDCARD
            ):
                return False
        segment = self.segments_by_id[step.segment_id]
        members = (
            [self.segments_by_id[member_id] for member_id in segment.member_segment_ids]
            if segment.member_segment_ids
            else [segment]
        )
        return all(
            member.selector.step_range is None
            or member.selector.step_range == WILDCARD
            for member in members
        )

    def current_state_generation_kwargs(self) -> dict[str, Any]:
        """Resolve the single cross-timestep feature-reuse policy for generation."""

        if not self.current_state_steps:
            return {}
        step = self.current_state_steps[0]
        ctx = RuntimeContext(
            task_type=self.bundle.segment_plan.request.task_type,
            run_id=self.run_id,
            phase="denoise",
            layer_idx=0,
            total_layers=self.total_layers,
            branch="generation",
            step_index=0,
            total_steps=self.total_steps,
            cfg_branch="main",
            batch_idx=0,
        )
        params = self._resolve_params(step, ctx)
        operator = self.registry.require_current_state_operator(step.operator)
        state = self.current_state_states.setdefault(step.id, {})
        decision = operator.begin_run(params, ctx, state)
        self.current_state_decision = decision
        return operator.generation_kwargs(decision)

    def current_state_runtime_params(self, *, layer_idx: int) -> dict[str, Any]:
        """Resolve dynamic current-state parameters for one layer invocation."""

        if not self.current_state_steps:
            return {}
        step = self.current_state_steps[0]
        ctx = self._runtime_context(layer_idx=layer_idx, mode="gen", sample_idx=0)
        if ctx.phase != "denoise":
            return {}
        params = self._resolve_params(step, ctx)
        operator = self.registry.require_current_state_operator(step.operator)
        state_key = f"{step.id}:{ctx.cfg_branch}"
        state = self.current_state_states.setdefault(state_key, {})
        decision = operator.on_step(params, ctx, state)
        key = (
            ctx.run_id,
            ctx.cfg_branch,
            int(ctx.step_index or 0),
            int(ctx.layer_idx),
            step.id,
        )
        self.current_state_step_decisions[key] = decision
        return dict(decision.params)

    def capture_current_state_model_stats(self, model: Any) -> None:
        """Copy model-owned feature-cache statistics before the model clears flags."""

        if not self.current_state_steps:
            return
        if hasattr(model, "last_duca_stats"):
            self.current_state_model_stats["duca"] = jsonable(model.last_duca_stats)
        if hasattr(model, "last_taylorseer_stats"):
            self.current_state_model_stats["taylorseer"] = jsonable(model.last_taylorseer_stats)

    def get_current_state_attention_scores(
        self,
        *,
        layer_idx: int,
        sample_idx: int = 0,
    ) -> torch.Tensor | None:
        key = (
            self.run_id,
            self.phase,
            self.cfg_branch,
            int(self.step_index),
            int(layer_idx),
            int(sample_idx),
        )
        return self.current_state_attention_scores.get(key)

    def _needs_current_state_attention(self, runtime: RuntimeContext) -> bool:
        if not self.current_state_steps or runtime.phase != "denoise":
            return False
        step = self.current_state_steps[0]
        if step.operator not in {"duca_current_state_reuse", "duca_age_norm_experimental"}:
            return False
        params = self._resolve_params(step, runtime)
        return str(params.get("score_type", "attention")) == "attention"

    def _record_current_state_attention(
        self,
        probs: torch.Tensor,
        key_type_ids: torch.Tensor | None,
        runtime: RuntimeContext,
    ) -> None:
        if not self._needs_current_state_attention(runtime) or key_type_ids is None:
            return
        current_mask = key_type_ids.to(device=probs.device) == KV_TYPE_NAME_TO_ID["current_vae"]
        if not bool(current_mask.any()):
            return
        # Official DuCa ranks current image tokens using attention received by
        # each token. Average heads after summing over current queries.
        score = probs[:, :, current_mask].float().sum(dim=1).mean(dim=0).detach()
        key = (
            runtime.run_id,
            runtime.phase,
            runtime.cfg_branch,
            int(runtime.step_index or 0),
            int(runtime.layer_idx),
            int(runtime.batch_idx),
        )
        self.current_state_attention_scores[key] = score

    def _static_fastpath_kind(
        self,
        *,
        runtime: RuntimeContext,
        query: torch.Tensor,
        physical_segments: list[Any],
        causal: bool,
    ) -> str:
        """Classify immutable physical layouts once per run and layer.

        Physical segment objects are recreated for each inference run and stay
        stable after materialization.  The execution plan and frozen-selection
        status are therefore invariant across denoising steps for a given
        object tuple.  Caching this control-plane decision removes repeated
        rule/segment traversal without caching any tensor result.
        """

        if not physical_segments:
            return "none"
        key = (
            runtime.run_id,
            runtime.phase,
            runtime.branch,
            runtime.cfg_branch,
            int(runtime.layer_idx),
            int(runtime.batch_idx),
            bool(causal),
            int(query.shape[0]),
            tuple(id(segment) for segment in physical_segments),
        )
        cached = self.static_fastpath_cache.get(key)
        if cached is not None:
            return cached
        if any(isinstance(segment, PackedKIVICache) for segment in physical_segments):
            kind = (
                "mixed"
                if self._can_bypass_execution_for_static_mixed(
                    runtime=runtime,
                    query=query,
                    physical_segments=physical_segments,
                    causal=causal,
                )
                else "none"
            )
        else:
            kind = (
                "h2o"
                if self._can_bypass_execution_for_frozen_h2o(
                    runtime=runtime,
                    query=query,
                    physical_segments=physical_segments,
                    causal=causal,
                )
                else "none"
            )
        self.static_fastpath_cache[key] = kind
        return kind

    def apply_static_physical_attention(
        self,
        *,
        q_i: torch.Tensor,
        k_i: torch.Tensor,
        v_i: torch.Tensor,
        layer_idx: int,
        mode: str,
        sample_idx: int,
        causal: bool,
        physical_segments: list[Any] | None,
    ) -> torch.Tensor | None:
        """Dispatch a frozen physical layout before generic hook preparation."""

        segments = physical_segments or []
        if not segments:
            return None
        ctx = self._runtime_context(
            layer_idx=layer_idx, mode=mode, sample_idx=sample_idx
        )
        fastpath = self._static_fastpath_kind(
            runtime=ctx,
            query=q_i,
            physical_segments=segments,
            causal=causal,
        )
        if fastpath == "none":
            return None
        if ctx.phase == "denoise":
            self.realized_storage_totals["max_denoise_query_tokens"] = max(
                int(
                    self.realized_storage_totals.get(
                        "max_denoise_query_tokens", 0
                    )
                ),
                int(q_i.shape[0]),
            )
        self.realized_storage_totals["static_preparation_bypass_calls"] += 1
        if fastpath == "h2o":
            return self._run_fused_h2o_attention(
                query=q_i,
                dense_key=k_i,
                dense_value=v_i,
                physical_segments=segments,
                runtime=ctx,
            )
        if fastpath == "mixed":
            return self._run_fused_mixed_attention(
                query=q_i,
                dense_key=k_i,
                dense_value=v_i,
                physical_segments=segments,
                runtime=ctx,
            )
        raise RuntimeError(f"Unknown static physical fast path: {fastpath}")

    def apply_static_physical_attention_batch(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        layer_idx: int,
        mode: str,
        causal: bool,
        physical_cache_by_sample: list[list[Any]] | None,
    ) -> torch.Tensor | None:
        """Run frozen per-sample physical layouts with one varlen FA launch."""

        if (
            not self.batch_physical_attention
            or flash_attn_varlen_func is None
            or q.device.type != "cuda"
            or causal
            or not physical_cache_by_sample
            or len(physical_cache_by_sample) <= 1
        ):
            return None

        from ..storage.kivi_cuda import dequantize_kivi_chunk

        cu_q, cu_k = self._physical_varlen_sample_spans(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            q_tokens=int(q.shape[0]),
            k_tokens=int(k.shape[0]),
            layer_idx=layer_idx,
            mode=mode,
            physical_cache_by_sample=physical_cache_by_sample,
        )
        if len(cu_q) != len(physical_cache_by_sample) + 1:
            return None
        entries = []
        total_key_tokens = 0
        max_query_tokens = 0
        max_key_tokens = 0
        for sample_idx, segments in enumerate(physical_cache_by_sample):
            if not segments:
                return None
            q_i = q[cu_q[sample_idx] : cu_q[sample_idx + 1]]
            k_i = k[cu_k[sample_idx] : cu_k[sample_idx + 1]]
            v_i = v[cu_k[sample_idx] : cu_k[sample_idx + 1]]
            runtime = self._runtime_context(
                layer_idx=layer_idx, mode=mode, sample_idx=sample_idx
            )
            fastpath = self._static_fastpath_kind(
                runtime=runtime,
                query=q_i,
                physical_segments=segments,
                causal=causal,
            )
            if fastpath == "none":
                return None
            h2o_segments = tuple(
                segment for segment in segments if isinstance(segment, GQAH2OCache)
            )
            packed_segments = tuple(
                segment for segment in segments if isinstance(segment, PackedKIVICache)
            )
            if fastpath == "h2o" and packed_segments:
                return None
            if fastpath == "mixed" and not packed_segments:
                return None
            if any(not segment.selection_frozen for segment in h2o_segments):
                return None
            bf16_keys = [k_i]
            bf16_values = [v_i]
            bf16_keys.extend(segment.key.transpose(0, 1) for segment in h2o_segments)
            bf16_values.extend(segment.value.transpose(0, 1) for segment in h2o_segments)
            bf16_keys.extend(
                segment.residual_key
                for segment in packed_segments
                if int(segment.residual_key.shape[0]) > 0
            )
            bf16_values.extend(
                segment.residual_value
                for segment in packed_segments
                if int(segment.residual_value.shape[0]) > 0
            )
            bf16_tokens = sum(int(item.shape[0]) for item in bf16_keys)
            packed_tokens = sum(
                int(segment.quantized_tokens) for segment in packed_segments
            )
            key_tokens = bf16_tokens + packed_tokens
            entries.append(
                (
                    runtime,
                    bf16_keys,
                    bf16_values,
                    packed_segments,
                    bf16_tokens,
                    key_tokens,
                )
            )
            total_key_tokens += key_tokens
            max_query_tokens = max(max_query_tokens, int(q_i.shape[0]))
            max_key_tokens = max(max_key_tokens, key_tokens)

        workspace_key, workspace_value = self._mixed_attention_workspace(
            template=k,
            token_capacity=total_key_tokens,
        )
        key_offsets = [0]
        for entry in entries:
            key_offsets.append(key_offsets[-1] + int(entry[5]))

        def copy_bf16_segments() -> None:
            for offset, entry in zip(key_offsets, entries):
                _, bf16_keys, bf16_values, _, bf16_tokens, _ = entry
                torch.cat(
                    bf16_keys,
                    dim=0,
                    out=workspace_key[offset : offset + bf16_tokens],
                )
                torch.cat(
                    bf16_values,
                    dim=0,
                    out=workspace_value[offset : offset + bf16_tokens],
                )

        def dequantize_packed_segments() -> None:
            for offset, entry in zip(key_offsets, entries):
                _, _, _, packed_segments, bf16_tokens, key_tokens = entry
                packed_offset = offset + bf16_tokens
                for segment in packed_segments:
                    packed_stop = packed_offset + int(segment.quantized_tokens)
                    dequantize_kivi_chunk(
                        segment,
                        start=0,
                        stop=int(segment.quantized_tokens),
                        dtype=q.dtype,
                        output_key=workspace_key,
                        output_value=workspace_value,
                        output_offset=packed_offset,
                    )
                    packed_offset = packed_stop
                if packed_offset != offset + key_tokens:
                    raise RuntimeError(
                        "Batched physical attention workspace layout mismatch"
                    )

        if self.overlap_batch_dequant and any(entry[3] for entry in entries):
            current_stream = torch.cuda.current_stream(q.device)
            auxiliary_stream = self._mixed_auxiliary_stream(q.device)
            auxiliary_stream.wait_stream(current_stream)
            with torch.cuda.stream(auxiliary_stream):
                dequantize_packed_segments()
            copy_bf16_segments()
            current_stream.wait_stream(auxiliary_stream)
            self.realized_storage_totals[
                "physical_varlen_overlap_dequant_calls"
            ] += 1
        else:
            copy_bf16_segments()
            dequantize_packed_segments()

        for runtime, _, _, _, _, _ in entries:
            if runtime.phase == "denoise":
                self.realized_storage_totals["max_denoise_query_tokens"] = max(
                    int(
                        self.realized_storage_totals.get(
                            "max_denoise_query_tokens", 0
                        )
                    ),
                    max_query_tokens,
                )

        runtime = entries[0][0]
        cu_cache_key = (
            q.device.type,
            q.device.index,
            tuple(key_offsets),
        )
        packed_cu_seqlens_k = self.physical_varlen_cu_cache.get(cu_cache_key)
        if packed_cu_seqlens_k is None:
            packed_cu_seqlens_k = torch.tensor(
                key_offsets, device=q.device, dtype=torch.int32
            )
            self.physical_varlen_cu_cache[cu_cache_key] = packed_cu_seqlens_k
            self.realized_storage_totals["physical_varlen_cu_builds"] += 1
        else:
            self.realized_storage_totals["physical_varlen_cu_reuses"] += 1
        with self._measure("attention.physical_varlen_flash", runtime):
            output = flash_attn_varlen_func(
                q=q,
                k=workspace_key[:total_key_tokens],
                v=workspace_value[:total_key_tokens],
                cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                cu_seqlens_k=packed_cu_seqlens_k,
                max_seqlen_q=max_query_tokens,
                max_seqlen_k=max_key_tokens,
                dropout_p=0.0,
                causal=False,
            )
        self.realized_storage_totals["physical_varlen_attention_calls"] += 1
        self.realized_storage_totals["physical_varlen_attention_samples"] += len(
            entries
        )
        self.realized_storage_totals["static_preparation_bypass_calls"] += len(
            entries
        )
        return output

    def _physical_varlen_sample_spans(
        self,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        q_tokens: int,
        k_tokens: int,
        layer_idx: int,
        mode: str,
        physical_cache_by_sample: list[list[Any]],
    ) -> tuple[list[int], list[int]]:
        """Cache stable packed-sample boundaries without repeated CUDA syncs."""

        layout_identity = tuple(
            tuple(id(segment) for segment in segments)
            for segments in physical_cache_by_sample
        )
        cache_key = (
            self.run_id,
            self.phase,
            normalize_branch(mode) or self.branch,
            self.cfg_branch,
            tuple(self.cfg_branches or ()),
            int(layer_idx),
            int(q_tokens),
            int(k_tokens),
            layout_identity,
        )
        cached = self.physical_varlen_span_cache.get(cache_key)
        if cached is not None:
            self.realized_storage_totals["physical_varlen_span_reuses"] += 1
            return cached
        spans = (
            cu_seqlens_q.detach().to("cpu").tolist(),
            cu_seqlens_k.detach().to("cpu").tolist(),
        )
        self.physical_varlen_span_cache[cache_key] = spans
        self.realized_storage_totals["physical_varlen_span_builds"] += 1
        return spans

    def apply_to_attention(
        self,
        *,
        q_i: torch.Tensor,
        k_i: torch.Tensor,
        v_i: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_type_ids: torch.Tensor | None,
        protected_mask: torch.Tensor,
        layer_idx: int,
        mode: str,
        sample_idx: int,
        causal: bool,
        storage_num_kv_heads: int | None = None,
        physical_segments: list[Any] | None = None,
    ) -> torch.Tensor:
        ctx = self._runtime_context(layer_idx=layer_idx, mode=mode, sample_idx=sample_idx)
        physical_segments = physical_segments or []
        if ctx.phase == "denoise":
            self.realized_storage_totals["max_denoise_query_tokens"] = max(
                int(self.realized_storage_totals.get("max_denoise_query_tokens", 0)),
                int(q_i.shape[0]),
            )
        fastpath = self._static_fastpath_kind(
            runtime=ctx,
            query=q_i,
            physical_segments=physical_segments,
            causal=causal,
        )
        if fastpath == "h2o":
            return self._run_fused_h2o_attention(
                query=q_i,
                dense_key=k_i,
                dense_value=v_i,
                physical_segments=physical_segments,
                runtime=ctx,
            )
        if fastpath == "mixed":
            return self._run_fused_mixed_attention(
                query=q_i,
                dense_key=k_i,
                dense_value=v_i,
                physical_segments=physical_segments,
                runtime=ctx,
            )

        if self.collect_diagnostics:
            self._record_attention_coverage(key_type_ids, int(k_i.shape[0]))
        protected = protected_mask.clone()
        k_work = k_i
        v_work = v_i
        deferred_attention_updates: list[tuple[ProcessingOperator, OperatorExecutionContext]] = []

        k_work, v_work, protected, probs_override, keep_mask_override = self._execute_stages(
            {"pre_attention", "attention"},
            runtime=ctx,
            q=q_i,
            k=k_work,
            v=v_work,
            attn_mask=attn_mask,
            key_type_ids=key_type_ids,
            protected_mask=protected,
            storage_num_kv_heads=storage_num_kv_heads,
            deferred_attention_updates=deferred_attention_updates,
        )

        if self.collect_diagnostics:
            self.tensor_shape_trace.append(
                {
                    "phase": ctx.phase,
                    "branch": ctx.branch,
                    "cfg_branch": ctx.cfg_branch,
                    "step_index": int(ctx.step_index or 0),
                    "layer_idx": int(ctx.layer_idx),
                    "sample_idx": int(ctx.batch_idx),
                    "query_shape": list(q_i.shape),
                    "dense_key_shape": list(k_i.shape),
                    "physical_segments": [
                        {
                            "layout": type(segment).__name__,
                            "tokens": int(
                                getattr(
                                    segment,
                                    "token_count",
                                    getattr(segment, "retained_tokens", 0),
                                )
                            ),
                            "original_tokens": int(
                                getattr(
                                    segment,
                                    "original_tokens",
                                    getattr(segment, "token_count", 0),
                                )
                            ),
                            "resident_bytes": int(segment.resident_bytes),
                            "full_precision_bytes": int(segment.full_precision_bytes),
                            "token_type_id": int(segment.token_type_id),
                            "byte_breakdown": dict(segment.byte_breakdown),
                        }
                        for segment in physical_segments
                    ],
                }
            )
            if len(self.tensor_shape_trace) > 8192:
                self.tensor_shape_trace = self.tensor_shape_trace[-8192:]
        if self._can_use_fused_h2o_attention(
            runtime=ctx,
            query=q_i,
            physical_segments=physical_segments,
            causal=causal,
            probs_override=probs_override,
            keep_mask_override=keep_mask_override,
            deferred_attention_updates=deferred_attention_updates,
        ):
            return self._run_fused_h2o_attention(
                query=q_i,
                dense_key=k_work,
                dense_value=v_work,
                physical_segments=physical_segments,
                runtime=ctx,
            )
        if self._can_use_fused_mixed_attention(
            runtime=ctx,
            query=q_i,
            physical_segments=physical_segments,
            causal=causal,
            probs_override=probs_override,
            keep_mask_override=keep_mask_override,
            deferred_attention_updates=deferred_attention_updates,
        ):
            return self._run_fused_mixed_attention(
                query=q_i,
                dense_key=k_work,
                dense_value=v_work,
                physical_segments=physical_segments,
                runtime=ctx,
            )
        attention_unmodified = (
            not physical_segments
            and k_work is k_i
            and v_work is v_i
            and probs_override is None
            and keep_mask_override is None
            and not deferred_attention_updates
            and not self._needs_current_state_attention(ctx)
            and not any(
                step.stage == "post_attention"
                and self._runtime_step_matches(step, ctx)
                for step in self.steps
            )
        )
        if attention_unmodified:
            if flash_attn_func is not None and q_i.device.type == "cuda":
                return flash_attn_func(
                    q_i.unsqueeze(0),
                    k_i.unsqueeze(0),
                    v_i.unsqueeze(0),
                    dropout_p=0.0,
                    causal=causal,
                ).squeeze(0)
            return torch.nn.functional.scaled_dot_product_attention(
                q_i.transpose(0, 1).unsqueeze(0),
                k_i.transpose(0, 1).unsqueeze(0),
                v_i.transpose(0, 1).unsqueeze(0),
                attn_mask=attn_mask,
                enable_gqa=int(q_i.shape[1]) != int(k_i.shape[1]),
            ).squeeze(0).transpose(0, 1)

        reference_probs = None
        if any(
            str(invocation.params.get("score_source", "effective_attention"))
            == "original_attention"
            for _, invocation in deferred_attention_updates
        ):
            reference_scores = self._attention_scores(q_i, k_i, attn_mask)
            reference_probs = torch.softmax(reference_scores, dim=-1).to(v_i.dtype)

        pre_eviction_probs = None
        physical_parts: list[tuple[Any, torch.Tensor, Any]] = []
        for segment in physical_segments:
            with self._measure(f"attention.qk.{type(segment).__name__}", ctx):
                logits, value_application = segment.attention_parts(q_i)
            physical_parts.append((segment, logits, value_application))
        if probs_override is None:
            with self._measure("attention.qk.dense", ctx):
                scores = self._attention_scores(q_i, k_work, attn_mask)
            if physical_parts:
                combined_scores = torch.cat(
                    [scores, *[item[1] for item in physical_parts]], dim=-1
                )
                with self._measure("attention.softmax", ctx):
                    pre_eviction_combined = torch.softmax(combined_scores, dim=-1).to(
                        v_work.dtype
                    )
                pre_eviction_probs = pre_eviction_combined[..., : scores.shape[-1]]
            else:
                combined_scores = scores
                pre_eviction_combined = None
                with self._measure("attention.softmax", ctx):
                    pre_eviction_probs = torch.softmax(scores, dim=-1).to(v_work.dtype)
            if keep_mask_override is not None:
                dense_final_scores = scores.masked_fill(
                    ~keep_mask_override[:, None, :],
                    torch.finfo(scores.dtype).min,
                )
                final_scores = (
                    torch.cat(
                        [dense_final_scores, *[item[1] for item in physical_parts]],
                        dim=-1,
                    )
                    if physical_parts
                    else dense_final_scores
                )
                with self._measure("attention.softmax.masked", ctx):
                    combined_probs = torch.softmax(final_scores, dim=-1).to(v_work.dtype)
            else:
                combined_probs = (
                    pre_eviction_combined
                    if pre_eviction_combined is not None
                    else pre_eviction_probs
                )
            probs = combined_probs[..., : scores.shape[-1]]
        else:
            if physical_parts:
                raise RuntimeError(
                    "Attention-probability overrides cannot be combined with physical cache segments"
                )
            probs = probs_override.to(v_work.dtype)
            combined_probs = probs
        for operator, invocation in deferred_attention_updates:
            score_source = str(
                invocation.params.get("score_source", "effective_attention")
            )
            if score_source not in {
                "effective_attention",
                "original_attention",
                "pre_eviction_attention",
            }:
                raise ValueError(
                    f"Operator {invocation.step.operator} has unsupported "
                    f"score_source={score_source!r}"
                )
            update_probs = {
                "effective_attention": probs,
                "original_attention": reference_probs,
                "pre_eviction_attention": pre_eviction_probs,
            }[score_source]
            if update_probs is None:
                raise RuntimeError(
                    f"{score_source} probabilities were not computed for "
                    f"{invocation.step.operator}"
                )
            with self._measure(f"operator.{operator.name}.update", ctx):
                result = operator.update_after_attention(invocation, update_probs)
            if result.decision is not None:
                self._record_decision(result.step, result.runtime, result.decision)
            if result.physical_request is not None:
                pending_key = self._pending_decision_key(
                    result.runtime, result.step.segment_id
                )
                self.pending_physical_requests[pending_key].append(
                    result.physical_request
                )
        with self._measure("attention.av.dense", ctx):
            dense_output = gqa_probability_value(probs, v_work)
        offset = int(probs.shape[-1])
        for segment, logits, value_application in physical_parts:
            width = int(logits.shape[-1])
            segment_probs = combined_probs[..., offset : offset + width]
            with self._measure(f"attention.av.{type(segment).__name__}", ctx):
                segment_output = value_application(segment_probs)
            dense_output = dense_output + segment_output.to(dense_output.dtype)
            if hasattr(segment, "update_scores"):
                segment.update_scores(segment_probs)
                segment.enforce_budget()
            offset += width
        out_i = dense_output.transpose(0, 1)
        self._record_current_state_attention(probs, key_type_ids, ctx)

        ctx.kv_metadata["effective_attention_keep_mask"] = (
            keep_mask_override.detach()
            if keep_mask_override is not None
            else torch.ones(
                (int(q_i.shape[1]), int(k_work.shape[0])),
                device=k_work.device,
                dtype=torch.bool,
            )
        )

        self._execute_stages(
            {"post_attention"},
            runtime=ctx,
            q=q_i,
            k=k_work,
            v=v_work,
            attn_mask=attn_mask,
            key_type_ids=key_type_ids,
            protected_mask=protected,
            storage_num_kv_heads=storage_num_kv_heads,
            attention_probs=probs,
            attention_output=out_i,
            attention_prob_sources={
                "effective_attention": probs,
                "original_attention": reference_probs,
                "pre_eviction_attention": pre_eviction_probs,
            },
        )
        return out_i

    def _can_use_fused_h2o_attention(
        self,
        *,
        runtime: RuntimeContext,
        query: torch.Tensor,
        physical_segments: list[Any],
        causal: bool,
        probs_override: torch.Tensor | None,
        keep_mask_override: torch.Tensor | None,
        deferred_attention_updates: list[
            tuple[ProcessingOperator, OperatorExecutionContext]
        ],
    ) -> bool:
        if not physical_segments or probs_override is not None or keep_mask_override is not None:
            return False
        if deferred_attention_updates or self._needs_current_state_attention(runtime):
            return False
        if causal and int(query.shape[0]) != 1:
            return False
        if not all(
            isinstance(segment, GQAH2OCache) and segment.selection_frozen
            for segment in physical_segments
        ):
            return False
        return not any(
            step.stage == "post_attention"
            and self._runtime_step_matches(step, runtime)
            for step in self.steps
        )

    def _can_bypass_execution_for_frozen_h2o(
        self,
        *,
        runtime: RuntimeContext,
        query: torch.Tensor,
        physical_segments: list[Any],
        causal: bool,
    ) -> bool:
        if self.collect_diagnostics or not self._can_use_fused_h2o_attention(
            runtime=runtime,
            query=query,
            physical_segments=physical_segments,
            causal=causal,
            probs_override=None,
            keep_mask_override=None,
            deferred_attention_updates=[],
        ):
            return False
        physical_type_ids = {int(segment.token_type_id) for segment in physical_segments}
        for step in self.steps:
            if step.stage not in {"pre_attention", "attention"}:
                continue
            if not self._runtime_step_matches(step, runtime, preflight=True):
                continue
            if step.operator in {"identity", "protect"}:
                continue
            if step.operator != "h2o_physical_gqa":
                return False
            segment = self.segments_by_id[step.segment_id]
            members = (
                [self.segments_by_id[member_id] for member_id in segment.member_segment_ids]
                if segment.member_segment_ids
                else [segment]
            )
            managed_type_ids: set[int] = set()
            for member in members:
                if not runtime_step_matches(step, member, runtime):
                    continue
                if member.selector.kv_types == WILDCARD:
                    return False
                managed_type_ids.update(
                    KV_TYPE_NAME_TO_ID[name]
                    for name in member.selector.kv_types
                    if name in KV_TYPE_NAME_TO_ID
                )
            if not managed_type_ids or not managed_type_ids.issubset(physical_type_ids):
                return False
        return True

    def _mixed_attention_workspace(
        self,
        *,
        template: torch.Tensor,
        token_capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a stream-local scratch buffer shared by all layers.

        The buffer is transient attention workspace, not a BF16 cache replica:
        every used element is overwritten before FlashAttention consumes it.
        Reusing one maximum-size buffer avoids allocator traffic without adding
        one dequantized source-VAE copy per layer.
        """

        stream_id = 0
        if template.device.type == "cuda":
            stream_id = int(torch.cuda.current_stream(template.device).cuda_stream)
        key = (
            template.device.type,
            template.device.index,
            template.dtype,
            int(template.shape[1]),
            int(template.shape[2]),
            stream_id,
        )
        cached = self.mixed_workspace_cache.get(key)
        if cached is None or int(cached[0].shape[0]) < int(token_capacity):
            shape = (int(token_capacity), *template.shape[1:])
            cached = (
                torch.empty(shape, device=template.device, dtype=template.dtype),
                torch.empty(shape, device=template.device, dtype=template.dtype),
            )
            self.mixed_workspace_cache[key] = cached
            self.realized_storage_totals["mixed_workspace_allocations"] += 1
        else:
            self.realized_storage_totals["mixed_workspace_reuses"] += 1
        self.realized_storage_totals["mixed_workspace_peak_bytes"] = max(
            self.realized_storage_totals.get("mixed_workspace_peak_bytes", 0),
            sum(tensor.numel() * tensor.element_size() for tensor in cached),
        )
        return cached[0][:token_capacity], cached[1][:token_capacity]

    def _mixed_auxiliary_stream(self, device: torch.device) -> Any:
        if device.type != "cuda":
            raise RuntimeError("Mixed-attention auxiliary streams require CUDA")
        key = (device.type, device.index)
        stream = self.mixed_auxiliary_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self.mixed_auxiliary_streams[key] = stream
        return stream

    def _can_use_fused_mixed_attention(
        self,
        *,
        runtime: RuntimeContext,
        query: torch.Tensor,
        physical_segments: list[Any],
        causal: bool,
        probs_override: torch.Tensor | None,
        keep_mask_override: torch.Tensor | None,
        deferred_attention_updates: list[
            tuple[ProcessingOperator, OperatorExecutionContext]
        ],
    ) -> bool:
        def reject(reason: str) -> bool:
            self.realized_storage_totals[f"fused_mixed_reject_{reason}"] += 1
            return False
        # A plan with no materialized physical segment is a true no-op.  Do not
        # let fast-path capability probing mutate its runtime statistics.
        if not physical_segments:
            return False
        packed = [segment for segment in physical_segments if isinstance(segment, PackedKIVICache)]
        if not packed:
            return reject("no_packed")
        if flash_attn_func is None:
            return reject("no_flash")
        if query.device.type != "cuda":
            return reject("non_cuda")
        if causal:
            return reject("causal")
        if int(query.shape[0]) <= 1:
            return reject("single_query")
        if probs_override is not None:
            return reject("probs_override")
        if keep_mask_override is not None:
            return reject("keep_mask_override")
        if deferred_attention_updates:
            return reject("deferred_updates")
        if self._needs_current_state_attention(runtime):
            return reject("current_state_attention")
        if not all(
            isinstance(segment, (GQAH2OCache, PackedKIVICache))
            for segment in physical_segments
        ):
            return reject("unsupported_segment")
        if not all(
            segment.backend == "cuda"
            and segment.kernel_native_layout
            and segment.quantized_tokens > 0
            and segment.k_bits == segment.v_bits
            and segment.k_bits in {2, 4}
            for segment in packed
        ):
            return reject("unsupported_packed_layout")
        if any(
            step.stage == "post_attention"
            and self._runtime_step_matches(step, runtime)
            for step in self.steps
        ):
            return reject("post_attention")
        return True

    def _can_bypass_execution_for_static_mixed(
        self,
        *,
        runtime: RuntimeContext,
        query: torch.Tensor,
        physical_segments: list[Any],
        causal: bool,
    ) -> bool:
        """Skip rule dispatch after fixed physical segments are materialized."""

        if (
            self.collect_diagnostics
            or not physical_segments
            or not any(isinstance(segment, PackedKIVICache) for segment in physical_segments)
        ):
            return False
        if not self._can_use_fused_mixed_attention(
            runtime=runtime,
            query=query,
            physical_segments=physical_segments,
            causal=causal,
            probs_override=None,
            keep_mask_override=None,
            deferred_attention_updates=[],
        ):
            return False
        if any(
            isinstance(segment, GQAH2OCache) and not segment.selection_frozen
            for segment in physical_segments
        ):
            return False
        physical_types = {
            "h2o_physical_gqa": {
                int(segment.token_type_id)
                for segment in physical_segments
                if isinstance(segment, GQAH2OCache)
            },
            "kivi_packed_quantization": {
                int(segment.token_type_id)
                for segment in physical_segments
                if isinstance(segment, PackedKIVICache)
            },
        }
        for step in self.steps:
            if step.stage not in {"pre_attention", "attention"}:
                continue
            if not self._runtime_step_matches(step, runtime, preflight=True):
                continue
            if step.operator in {"identity", "protect"}:
                continue
            if step.operator not in physical_types:
                return False
            segment = self.segments_by_id[step.segment_id]
            members = (
                [self.segments_by_id[member_id] for member_id in segment.member_segment_ids]
                if segment.member_segment_ids
                else [segment]
            )
            managed_type_ids: set[int] = set()
            for member in members:
                if not runtime_step_matches(step, member, runtime):
                    continue
                if member.selector.kv_types == WILDCARD:
                    return False
                managed_type_ids.update(
                    KV_TYPE_NAME_TO_ID[name]
                    for name in member.selector.kv_types
                    if name in KV_TYPE_NAME_TO_ID
                )
            if not managed_type_ids or not managed_type_ids.issubset(
                physical_types[step.operator]
            ):
                return False
        return True

    def _run_fused_mixed_attention(
        self,
        *,
        query: torch.Tensor,
        dense_key: torch.Tensor,
        dense_value: torch.Tensor,
        physical_segments: list[Any],
        runtime: RuntimeContext,
    ) -> torch.Tensor:
        from ..storage.kivi_cuda import (
            dequantize_kivi_chunk,
            packed_kivi_quantized_attention_summary,
        )

        layout_key = tuple(id(segment) for segment in physical_segments)
        cached_layout = self.mixed_layout_cache.get(layout_key)
        if cached_layout is None:
            h2o_segments = tuple(
                segment
                for segment in physical_segments
                if isinstance(segment, GQAH2OCache)
            )
            packed_segments = tuple(
                segment
                for segment in physical_segments
                if isinstance(segment, PackedKIVICache)
            )
            residual_segments = tuple(
                segment
                for segment in packed_segments
                if int(segment.residual_key.shape[0]) > 0
            )
            cached_layout = (h2o_segments, packed_segments, residual_segments)
            self.mixed_layout_cache[layout_key] = cached_layout
        else:
            h2o_segments, packed_segments, residual_segments = cached_layout

        bf16_keys = [dense_key]
        bf16_values = [dense_value]
        bf16_keys.extend(segment.key.transpose(0, 1) for segment in h2o_segments)
        bf16_values.extend(segment.value.transpose(0, 1) for segment in h2o_segments)
        bf16_keys.extend(segment.residual_key for segment in residual_segments)
        bf16_values.extend(segment.residual_value for segment in residual_segments)

        backend = self.kivi_attention_backend
        chunk_size = self.kivi_flash_chunk_size
        total_packed_chunks = sum(
            math.ceil(int(segment.quantized_tokens) / chunk_size)
            for segment in packed_segments
        )
        output_only_single_chunk = (
            backend in {"chunked_flash", "chunked_flash_overlap"}
            and total_packed_chunks == 1
            and all(segment.selection_frozen for segment in h2o_segments)
        )
        overlap_single_chunk = (
            backend == "chunked_flash_overlap"
            and output_only_single_chunk
            and query.is_cuda
        )
        dual_stream_single_chunk = (
            backend == "chunked_flash_dual_stream"
            and total_packed_chunks == 1
            and all(segment.selection_frozen for segment in h2o_segments)
        )
        bf16_token_count = sum(int(key.shape[0]) for key in bf16_keys)
        all_key = None
        all_value = None
        if not output_only_single_chunk and not dual_stream_single_chunk:
            all_key = torch.cat(bf16_keys, dim=0)
            all_value = torch.cat(bf16_values, dim=0)
        aggregate = None
        if backend == "triton_fused":
            assert all_key is not None and all_value is not None
            with self._measure("attention.fused_mixed.flash", runtime):
                aggregate = flash_attention_summary(query, all_key, all_value)
            for segment in packed_segments:
                with self._measure("attention.fused_mixed.kivi", runtime):
                    segment_summary = packed_kivi_quantized_attention_summary(
                        segment, query
                    )
                with self._measure("attention.fused_mixed.merge", runtime):
                    aggregate = merge_attention_summaries([aggregate, segment_summary])
        elif dual_stream_single_chunk:
            segment = packed_segments[0]
            packed_tokens = int(segment.quantized_tokens)
            workspace_key, workspace_value = self._mixed_attention_workspace(
                template=dense_key,
                token_capacity=bf16_token_count + packed_tokens,
            )
            current_stream = torch.cuda.current_stream(query.device)
            auxiliary_stream = self._mixed_auxiliary_stream(query.device)
            auxiliary_stream.wait_stream(current_stream)
            with torch.cuda.stream(auxiliary_stream):
                with self._measure("attention.dual_stream.dequant", runtime):
                    dequantize_kivi_chunk(
                        segment,
                        start=0,
                        stop=packed_tokens,
                        dtype=query.dtype,
                        output_key=workspace_key,
                        output_value=workspace_value,
                        output_offset=bf16_token_count,
                    )
                with self._measure("attention.dual_stream.packed_flash", runtime):
                    packed_summary = flash_attention_summary(
                        query,
                        workspace_key[bf16_token_count:],
                        workspace_value[bf16_token_count:],
                    )
            with self._measure("attention.dual_stream.workspace", runtime):
                torch.cat(bf16_keys, dim=0, out=workspace_key[:bf16_token_count])
                torch.cat(bf16_values, dim=0, out=workspace_value[:bf16_token_count])
            with self._measure("attention.dual_stream.bf16_flash", runtime):
                bf16_summary = flash_attention_summary(
                    query,
                    workspace_key[:bf16_token_count],
                    workspace_value[:bf16_token_count],
                )
            current_stream.wait_stream(auxiliary_stream)
            with self._measure("attention.dual_stream.merge", runtime):
                aggregate = merge_attention_summaries(
                    [bf16_summary, packed_summary]
                )
            self.realized_storage_totals["fused_mixed_dual_stream_calls"] += 1
        else:
            for segment in packed_segments:
                for start in range(0, int(segment.quantized_tokens), chunk_size):
                    stop = min(int(segment.quantized_tokens), start + chunk_size)
                    if aggregate is None and output_only_single_chunk:
                        chunk_tokens = stop - start
                        first_key, first_value = self._mixed_attention_workspace(
                            template=dense_key,
                            token_capacity=bf16_token_count + chunk_tokens,
                        )
                        if overlap_single_chunk:
                            current_stream = torch.cuda.current_stream(query.device)
                            auxiliary_stream = self._mixed_auxiliary_stream(query.device)
                            auxiliary_stream.wait_stream(current_stream)
                            with torch.cuda.stream(auxiliary_stream):
                                with self._measure(
                                    "attention.overlap.dequant", runtime
                                ):
                                    dequantize_kivi_chunk(
                                        segment,
                                        start=start,
                                        stop=stop,
                                        dtype=query.dtype,
                                        output_key=first_key,
                                        output_value=first_value,
                                        output_offset=bf16_token_count,
                                    )
                            with self._measure(
                                "attention.overlap.workspace", runtime
                            ):
                                torch.cat(
                                    bf16_keys,
                                    dim=0,
                                    out=first_key[:bf16_token_count],
                                )
                                torch.cat(
                                    bf16_values,
                                    dim=0,
                                    out=first_value[:bf16_token_count],
                                )
                            current_stream.wait_stream(auxiliary_stream)
                            self.realized_storage_totals[
                                "fused_mixed_overlap_calls"
                            ] += 1
                        else:
                            with self._measure(
                                "attention.fused_mixed.workspace", runtime
                            ):
                                torch.cat(
                                    bf16_keys,
                                    dim=0,
                                    out=first_key[:bf16_token_count],
                                )
                                torch.cat(
                                    bf16_values,
                                    dim=0,
                                    out=first_value[:bf16_token_count],
                                )
                            with self._measure(
                                "attention.chunked_kivi.dequant", runtime
                            ):
                                dequantize_kivi_chunk(
                                    segment,
                                    start=start,
                                    stop=stop,
                                    dtype=query.dtype,
                                    output_key=first_key,
                                    output_value=first_value,
                                    output_offset=bf16_token_count,
                                )
                        with self._measure(
                            "attention.fused_mixed.flash_output_only", runtime
                        ):
                            output = flash_attn_func(
                                query.unsqueeze(0),
                                first_key.unsqueeze(0),
                                first_value.unsqueeze(0),
                                dropout_p=0.0,
                                causal=False,
                            ).squeeze(0)
                        self.realized_storage_totals[
                            "fused_mixed_output_only_calls"
                        ] += 1
                        self.realized_storage_totals[
                            "fused_mixed_attention_calls"
                        ] += 1
                        return output
                    assert all_key is not None and all_value is not None
                    with self._measure("attention.chunked_kivi.dequant", runtime):
                        chunk_key, chunk_value = dequantize_kivi_chunk(
                            segment,
                            start=start,
                            stop=stop,
                            dtype=query.dtype,
                        )
                    if aggregate is None:
                        # Fold the dense/protected cache and the first packed
                        # chunk into one FlashAttention call.  This is exactly
                        # equivalent to a later LSE merge and removes one
                        # attention launch plus one merge per layer.
                        first_key = torch.cat([all_key, chunk_key], dim=0)
                        first_value = torch.cat([all_value, chunk_value], dim=0)
                        with self._measure("attention.fused_mixed.flash", runtime):
                            aggregate = flash_attention_summary(
                                query, first_key, first_value
                            )
                        continue
                    with self._measure("attention.chunked_kivi.flash", runtime):
                        chunk_summary = flash_attention_summary(
                            query, chunk_key, chunk_value
                        )
                    with self._measure("attention.chunked_kivi.merge", runtime):
                        aggregate = merge_attention_summaries(
                            [aggregate, chunk_summary]
                        )
        if aggregate is None:
            raise RuntimeError("Fused mixed attention requires packed KIVI tokens")
        output, global_lse = aggregate

        # H2O still receives exact global probabilities for its surviving tokens.
        # Recomputing only these compacted QK blocks is much cheaper than storing
        # the full mixed-attention matrix.
        for segment in h2o_segments:
            if segment.selection_frozen:
                continue
            with self._measure("attention.fused_mixed.h2o_scores", runtime):
                logits = gqa_query_key_logits(
                    query, segment.key.transpose(0, 1)
                ) / math.sqrt(query.shape[-1])
                probabilities = torch.exp(logits - global_lse.unsqueeze(-1)).to(
                    segment.value.dtype
                )
                segment.update_scores(probabilities)
                segment.enforce_budget()
        self.realized_storage_totals["fused_mixed_attention_calls"] += 1
        return output

    def _run_fused_h2o_attention(
        self,
        *,
        query: torch.Tensor,
        dense_key: torch.Tensor,
        dense_value: torch.Tensor,
        physical_segments: list[Any],
        runtime: RuntimeContext,
    ) -> torch.Tensor:
        all_key = torch.cat(
            [dense_key]
            + [segment.key.transpose(0, 1) for segment in physical_segments],
            dim=0,
        )
        all_value = torch.cat(
            [dense_value]
            + [segment.value.transpose(0, 1) for segment in physical_segments],
            dim=0,
        )
        if flash_attn_func is not None and query.device.type == "cuda":
            with self._measure("attention.fused_h2o_flash", runtime):
                output = flash_attn_func(
                    query.unsqueeze(0),
                    all_key.unsqueeze(0),
                    all_value.unsqueeze(0),
                    dropout_p=0.0,
                    causal=False,
                ).squeeze(0)
            self.realized_storage_totals["fused_h2o_flash_calls"] += 1
        else:
            with self._measure("attention.fused_h2o_sdpa", runtime):
                output = torch.nn.functional.scaled_dot_product_attention(
                    query.transpose(0, 1).unsqueeze(0),
                    all_key.transpose(0, 1).unsqueeze(0),
                    all_value.transpose(0, 1).unsqueeze(0),
                    attn_mask=None,
                    is_causal=False,
                    enable_gqa=int(query.shape[1]) != int(all_key.shape[1]),
                ).squeeze(0).transpose(0, 1)
            self.realized_storage_totals["fused_h2o_sdpa_calls"] += 1
        self.realized_storage_totals["fused_h2o_attention_calls"] += 1
        return output

    def _execute_stages(
        self,
        stages: set[str],
        *,
        runtime: RuntimeContext,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_type_ids: torch.Tensor | None,
        protected_mask: torch.Tensor,
        storage_num_kv_heads: int | None = None,
        attention_probs: torch.Tensor | None = None,
        attention_output: torch.Tensor | None = None,
        attention_prob_sources: dict[str, torch.Tensor | None] | None = None,
        deferred_attention_updates: list[
            tuple[ProcessingOperator, OperatorExecutionContext]
        ] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        k_work = k
        v_work = v
        protected = protected_mask
        probs_override = None
        keep_mask_override = None
        for step in self.steps:
            if step.stage not in stages:
                continue
            segment = self.segments_by_id[step.segment_id]
            if not self._runtime_step_matches(step, runtime):
                continue
            segment_mask = self._segment_mask(step, runtime, key_type_ids, k.shape[0], k.device)
            segment_tokens = self._segment_token_count(
                step=step,
                runtime=runtime,
                segment_mask=segment_mask,
                key_len=int(k.shape[0]),
            )
            has_segment_tokens = segment_tokens > 0
            if self.collect_diagnostics:
                coverage = self.coverage_rules[step.id]
                coverage["eligible_calls"] += 1
                coverage["tokens_seen"] += float(segment_tokens)
            if not has_segment_tokens:
                if self.collect_diagnostics:
                    coverage["miss_calls"] += 1
                continue
            if self.collect_diagnostics:
                coverage["hit_calls"] += 1
            runtime.kv_metadata.update(
                {
                    "key_len": int(k_work.shape[0]),
                    "current_segment_tokens": segment_tokens,
                    "attention_num_heads": int(q.shape[1]),
                    "storage_num_kv_heads": int(storage_num_kv_heads or k_work.shape[1]),
                    "head_dim": int(k_work.shape[-1]),
                    "element_size": int(k_work.element_size()),
                    "kv_type_name_to_id": dict(KV_TYPE_NAME_TO_ID),
                }
            )
            current_type_id = self.segment_type_ids.get(step.segment_id)
            if current_type_id is None:
                runtime.kv_metadata.pop("current_segment_type_id", None)
            else:
                runtime.kv_metadata["current_segment_type_id"] = current_type_id
            current_span = self._segment_contiguous_span(
                step=step,
                runtime=runtime,
                segment_mask=segment_mask,
                segment_tokens=segment_tokens,
                key_len=int(k.shape[0]),
            )
            if current_span is None:
                runtime.kv_metadata.pop("current_segment_span", None)
            else:
                runtime.kv_metadata["current_segment_span"] = current_span
            if step.scheduler is not None:
                runtime.kv_metadata["active_cache_types"] = sorted(
                    {
                        KV_TYPE_ID_TO_NAME[int(type_id)]
                        for type_id in (
                            torch.unique(key_type_ids).tolist()
                            if key_type_ids is not None
                            else []
                        )
                        if int(type_id) in KV_TYPE_ID_TO_NAME
                    }
                )
            params = self._resolve_params(step, runtime)
            budget_record_key = self._record_budget_allocation(
                step,
                runtime,
                segment_mask,
                params,
            )
            operator_attention_probs = attention_probs
            if attention_prob_sources is not None:
                score_source = str(params.get("score_source", "effective_attention"))
                if score_source not in attention_prob_sources:
                    raise ValueError(
                        f"Operator {step.operator} has unsupported "
                        f"score_source={score_source!r}"
                    )
                operator_attention_probs = attention_prob_sources[score_source]
            operator = self.registry.require_operator(step.operator)
            if not operator.supports(segment, runtime):
                raise ValueError(
                    f"Operator {step.operator} does not support segment {segment.id} "
                    f"for phase={runtime.phase}, branch={runtime.branch}"
                )
            storage = CacheStorageView.from_attention_tensors(
                k_work,
                v_work,
                num_query_heads=int(q.shape[1]),
                num_kv_heads=int(storage_num_kv_heads or k_work.shape[1]),
                key_type_ids=key_type_ids,
            )
            available_features = {
                "query",
                "key",
                "value",
                "segment_mask",
                "protected_mask",
                "cache_storage_view",
            }
            if key_type_ids is not None:
                available_features.add("key_type_ids")
            if operator_attention_probs is not None:
                available_features.add("attention_probs")
            if attention_output is not None:
                available_features.add("attention_output")
            operator.capability_descriptor().validate_runtime(
                operator=step.operator,
                stage=step.stage,
                phase=runtime.phase,
                device=k_work.device.type,
                layout=storage.layout.kind,
                available_features=available_features,
            )
            # Deferred post-attention operators must observe the metadata for
            # their own segment. Later steps reuse and mutate ``runtime``.
            invocation_runtime = replace(
                runtime,
                kv_metadata=dict(runtime.kv_metadata),
            )
            invocation = OperatorExecutionContext(
                runtime=invocation_runtime,
                step=step,
                params=params,
                q=q,
                k=k_work,
                v=v_work,
                attn_mask=attn_mask,
                key_type_ids=key_type_ids,
                segment_mask=segment_mask,
                protected_mask=protected,
                state=(
                    self.topk_state
                    if step.operator == "topk_eviction"
                    else self.classic_states
                    if step.operator in {
                        "streamingllm_eviction",
                        "snapkv_attention_mask",
                        "pyramidkv_attention_mask",
                    }
                    else self.lifetime_states
                    if step.operator in {"lifetime_retirement", "attention_mass_lifetime_observer"}
                    else self.block_metric_states
                    if step.operator in {
                        "block_attention_metrics",
                        "parallel_block_attention_metrics",
                    }
                    else self.h2o_states
                    if step.operator in {"h2o_attention_mask", "h2o_segment_attention_mask"}
                    else self.quantization_states
                    if step.operator == "kivi_quantization"
                    else self.hh_scores
                ),
                metadata_state=(
                    self.block_summary_states
                    if step.operator in {
                        "block_attention_metrics",
                        "parallel_block_attention_metrics",
                    }
                    else None
                ),
                record=lambda values, current_step=step, current_mask=segment_mask, current_budget_key=budget_record_key: (
                    self._record(current_step, runtime, current_mask, values)
                    if self.collect_diagnostics
                    else None,
                    self._record_budget_usage(current_budget_key, values),
                ),
                set_storage_snapshot=lambda values, current_step=step, current_budget_key=budget_record_key: (
                    self._set_storage_snapshot(current_step, runtime, values)
                    if self.collect_diagnostics
                    else None,
                    self._record_budget_storage(current_budget_key, values),
                ),
                storage_num_kv_heads=storage_num_kv_heads,
                attention_probs=operator_attention_probs,
                attention_prob_sources=attention_prob_sources,
                attention_output=attention_output,
                storage=storage,
            )
            probability_override_already_set = probs_override is not None
            result = operator.execute(invocation)
            if operator.uses_final_attention:
                if deferred_attention_updates is None:
                    raise ValueError(
                        f"Operator {step.operator} requires final attention probabilities "
                        "but no deferred update collector is available"
                    )
                deferred_attention_updates.append((operator, result))
            for field_name, original in (
                ("q", q),
                ("attn_mask", attn_mask),
                ("key_type_ids", key_type_ids),
                ("segment_mask", segment_mask),
                ("attention_probs", operator_attention_probs),
                ("attention_prob_sources", attention_prob_sources),
                ("attention_output", attention_output),
            ):
                if getattr(result, field_name) is not original:
                    raise ValueError(
                        f"Operator {step.operator} replaced read-only field {field_name}; "
                        "only k, v, and protected_mask are writable"
                    )
            if probability_override_already_set and (result.k is not k_work or result.v is not v_work):
                raise ValueError(
                    f"Operator {step.operator} modified K/V after an earlier operator replaced "
                    "final attention probabilities"
                )
            k_work = result.k
            v_work = result.v
            protected = result.protected_mask
            if result.attention_probs_override is not None:
                if probs_override is not None or keep_mask_override is not None:
                    raise ValueError("Multiple operators attempted incompatible attention overrides")
                probs_override = result.attention_probs_override
            if result.attention_keep_mask_override is not None:
                if probs_override is not None:
                    raise ValueError("Multiple operators attempted incompatible attention overrides")
                current_keep = result.attention_keep_mask_override.to(device=k_work.device, dtype=torch.bool)
                expected_shape = (int(q.shape[1]), int(k_work.shape[0]))
                if tuple(current_keep.shape) != expected_shape:
                    raise ValueError(
                        f"Operator {step.operator} returned attention keep mask shape "
                        f"{tuple(current_keep.shape)}, expected {expected_shape}"
                    )
                keep_mask_override = (
                    current_keep
                    if keep_mask_override is None
                    else keep_mask_override & current_keep
                )
            if result.decision is not None:
                self._record_decision(step, runtime, result.decision)
                if result.decision.keep_mask is not None:
                    pending_key = self._pending_decision_key(runtime, step.segment_id)
                    self.pending_storage_decisions[pending_key].append(result.decision)
            if result.physical_request is not None:
                pending_key = self._pending_decision_key(runtime, step.segment_id)
                self.pending_physical_requests[pending_key].append(
                    result.physical_request
                )
        return k_work, v_work, protected, probs_override, keep_mask_override

    def has_pending_cache_update(self, *, layer_idx: int, mode: str) -> bool:
        del layer_idx, mode
        return bool(self.pending_storage_decisions or self.pending_physical_requests)

    def apply_to_cache_update(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        key_type_ids: torch.Tensor | None,
        sample_lens: torch.Tensor,
        layer_idx: int,
        mode: str,
    ) -> CacheMutationResult | None:
        """Apply pending selection decisions to the persistent cache tensors.

        Packed BAGEL samples place existing cache tokens before the current
        query tokens. When the target is the existing cache, a decision made on
        a longer merged attention view is therefore projected by prefix.
        """

        if not self.pending_storage_decisions and not self.pending_physical_requests:
            return None
        lengths = [int(item) for item in sample_lens.to("cpu").tolist()]
        keep_masks: list[torch.Tensor] = []
        physical_requests: list[list[Any]] = []
        has_storage_decision = False
        has_physical_request = False
        runtime_branch = normalize_branch(mode) or self.branch
        for sample_idx, length in enumerate(lengths):
            keep = torch.ones(length, device=k.device, dtype=torch.bool)
            prefix = (
                self.run_id,
                self.phase,
                runtime_branch,
                self.cfg_branch,
                int(layer_idx),
                int(sample_idx),
            )
            matching_keys = [key for key in self.pending_storage_decisions if key[:6] == prefix]
            for key in matching_keys:
                for decision in self.pending_storage_decisions.pop(key):
                    if decision.keep_mask is None:
                        continue
                    has_storage_decision = True
                    if decision.token_count < length:
                        raise ValueError(
                            "SelectionDecision is shorter than the persistent cache sample; "
                            "packed cache layout is inconsistent"
                        )
                    keep &= decision.keep_mask[:length].to(device=k.device, dtype=torch.bool)
            requests_for_sample: list[Any] = []
            matching_physical = [
                key for key in self.pending_physical_requests if key[:6] == prefix
            ]
            for key in matching_physical:
                requests = self.pending_physical_requests.pop(key)
                has_physical_request = has_physical_request or bool(requests)
                requests_for_sample.extend(requests)
            keep_masks.append(keep)
            physical_requests.append(requests_for_sample)

        if not has_storage_decision and not has_physical_request:
            return None

        if has_physical_request:
            runtime = self._runtime_context(
                layer_idx=layer_idx, mode=mode, sample_idx=0
            )
            with self._measure("cache_update.physical_materialize", runtime):
                mutation = TensorCacheStorageBackend.compact_and_materialize(
                    key=k,
                    value=v,
                    key_type_ids=key_type_ids,
                    sample_lens=sample_lens,
                    keep_masks=keep_masks,
                    physical_requests=physical_requests,
                    sample_lengths=lengths,
                )
        else:
            mutation = TensorCacheStorageBackend.compact(
                key=k,
                value=v,
                key_type_ids=key_type_ids,
                sample_lens=sample_lens,
                keep_masks=keep_masks,
                sample_lengths=lengths,
            )
        self.realized_storage_totals["cache_update_calls"] += 1
        self.realized_storage_totals["evicted_tokens"] += int(k.shape[0] - mutation.key.shape[0])
        self.realized_storage_totals["physical_kv_bytes_retired"] += mutation.bytes_saved
        self.realized_storage_totals["latest_cache_bytes"] = mutation.compacted_bytes
        if mutation.physical_segments_by_sample is not None:
            physical_bytes = sum(
                segment.resident_bytes
                for segments in mutation.physical_segments_by_sample
                for segment in segments
            )
            full_bytes = sum(
                segment.full_precision_bytes
                for segments in mutation.physical_segments_by_sample
                for segment in segments
            )
            self.realized_storage_totals["physical_segment_resident_bytes"] += physical_bytes
            self.realized_storage_totals["physical_segment_full_bf16_bytes"] += full_bytes
            for segments in mutation.physical_segments_by_sample:
                for segment in segments:
                    type_name = KV_TYPE_ID_TO_NAME.get(
                        int(segment.token_type_id), f"type_{int(segment.token_type_id)}"
                    )
                    original_tokens = int(
                        segment.original_tokens
                        if hasattr(segment, "original_tokens")
                        else segment.token_count
                    )
                    retained_tokens = int(
                        segment.retained_tokens
                        if hasattr(segment, "retained_tokens")
                        else segment.token_count
                    )
                    self.realized_storage_totals[
                        f"physical_type_{type_name}_segments"
                    ] += 1
                    self.realized_storage_totals[
                        f"physical_type_{type_name}_original_tokens"
                    ] += original_tokens
                    self.realized_storage_totals[
                        f"physical_type_{type_name}_retained_tokens"
                    ] += retained_tokens
                    self.realized_storage_totals[
                        f"physical_type_{type_name}_resident_bytes"
                    ] += int(segment.resident_bytes)
                    self.realized_storage_totals[
                        f"physical_type_{type_name}_full_bf16_bytes"
                    ] += int(segment.full_precision_bytes)
                    for name, value in segment.byte_breakdown.items():
                        self.realized_storage_totals[
                            f"physical_segment_{name}"
                        ] += int(value)
        return mutation

    def summary(self) -> dict[str, Any]:
        processing_totals: dict[str, float] = defaultdict(float)
        for row in self.stats.values():
            for key, value in row.items():
                processing_totals[key] += float(value)
        logical_totals: dict[str, float] = defaultdict(float)
        for row in self.storage_snapshots.values():
            for key, value in row.items():
                logical_totals[key] += float(value)
        executable_operators = {
            step.operator for step in [*self.steps, *self.current_state_steps]
        }
        retired_bytes = int(self.realized_storage_totals.get("physical_kv_bytes_retired", 0))
        physical_segment_bytes = int(
            self.realized_storage_totals.get("physical_segment_resident_bytes", 0)
        )
        physical_segment_full_bytes = int(
            self.realized_storage_totals.get("physical_segment_full_bf16_bytes", 0)
        )
        physical_cache_types = {}
        for type_name in KV_TYPE_NAME_TO_ID:
            original_tokens = int(
                self.realized_storage_totals.get(
                    f"physical_type_{type_name}_original_tokens", 0
                )
            )
            if not original_tokens:
                continue
            resident = int(
                self.realized_storage_totals.get(
                    f"physical_type_{type_name}_resident_bytes", 0
                )
            )
            full = int(
                self.realized_storage_totals.get(
                    f"physical_type_{type_name}_full_bf16_bytes", 0
                )
            )
            physical_cache_types[type_name] = {
                "segments": int(
                    self.realized_storage_totals.get(
                        f"physical_type_{type_name}_segments", 0
                    )
                ),
                "original_tokens": original_tokens,
                "retained_tokens": int(
                    self.realized_storage_totals.get(
                        f"physical_type_{type_name}_retained_tokens", 0
                    )
                ),
                "resident_bytes": resident,
                "full_bf16_bytes": full,
                "retained_byte_ratio": float(resident / max(full, 1)),
            }
        realized_saved_bytes = max(0, retired_bytes - physical_segment_bytes)
        if physical_segment_bytes > 0:
            storage_mode = "physical_mixed_cache"
            memory_note = (
                "Dense BF16 managed segments were replaced by compacted H2O or packed KIVI "
                "tensors. Byte counts include payload, scales, minima, logical indices, "
                "H2O scores, and residual BF16 windows. Allocator peak is reported separately."
            )
        elif retired_bytes > 0:
            storage_mode = "physical_compaction"
            memory_note = (
                "Physical TopK shortens persistent KV tensors. Retired KV bytes are tensor payload "
                "accounting, not a measured reduction in allocator peak memory."
            )
        elif "kivi_quantization" in executable_operators or any(
            name.startswith("fake_quant") for name in executable_operators
        ):
            storage_mode = "fake_quant_dequant"
            memory_note = "Fake quantization tensors remain full precision; no realized memory saving is claimed."
        elif executable_operators & {
            "h2o_attention_mask",
            "h2o_segment_attention_mask",
            "snapkv_attention_mask",
            "pyramidkv_attention_mask",
        }:
            storage_mode = "attention_mask_only"
            memory_note = (
                "Mask-only operators such as H2O, SnapKV, and PyramidKV keep full K/V tensors; "
                "no realized memory saving is claimed."
            )
        else:
            storage_mode = "unmodified"
            memory_note = "No operator modified persistent cache storage."
        def aggregate_working_sets(summaries: list[dict[str, Any]]) -> dict[str, Any]:
            resident_byte_invocations = int(
                sum(item["resident_byte_invocations"] for item in summaries)
            )
            archive_byte_invocations = int(
                sum(item["archive_byte_invocations"] for item in summaries)
            )
            cumulative_h2d_bytes = int(
                sum(item["cumulative_h2d_bytes"] for item in summaries)
            )
            cumulative_full_rebuild_h2d_bytes = int(
                sum(
                    item.get("cumulative_full_rebuild_h2d_bytes", 0)
                    for item in summaries
                )
            )
            return {
                "trackers": len(summaries),
                "archive_bytes": int(sum(item["archive_bytes"] for item in summaries)),
                "current_resident_bytes": int(
                    sum(item["current_resident_bytes"] for item in summaries)
                ),
                "resident_byte_invocations": resident_byte_invocations,
                "archive_byte_invocations": archive_byte_invocations,
                "average_resident_ratio": float(
                    resident_byte_invocations / max(archive_byte_invocations, 1)
                ),
                "peak_resident_ratio": float(
                    max(
                        (item["peak_resident_ratio"] for item in summaries),
                        default=0.0,
                    )
                ),
                "cumulative_logical_restored_tokens": int(
                    sum(
                        item["cumulative_logical_restored_tokens"]
                        for item in summaries
                    )
                ),
                "cumulative_logical_evicted_tokens": int(
                    sum(
                        item["cumulative_logical_evicted_tokens"]
                        for item in summaries
                    )
                ),
                "cumulative_kv_head_restored_tokens": int(
                    sum(
                        item["cumulative_kv_head_restored_tokens"]
                        for item in summaries
                    )
                ),
                "cumulative_kv_head_evicted_tokens": int(
                    sum(
                        item["cumulative_kv_head_evicted_tokens"]
                        for item in summaries
                    )
                ),
                "cumulative_h2d_bytes": cumulative_h2d_bytes,
                "cumulative_full_rebuild_h2d_bytes": (
                    cumulative_full_rebuild_h2d_bytes
                ),
                "differential_transfer_savings_ratio": float(
                    1.0
                    - cumulative_h2d_bytes
                    / max(cumulative_full_rebuild_h2d_bytes, 1)
                )
                if cumulative_full_rebuild_h2d_bytes
                else 0.0,
            }

        tracker_states = [
            state
            for state in self.h2o_states.values()
            if state.get("working_set_tracker") is not None
        ]
        tracker_summaries = [state["working_set_tracker"].summary() for state in tracker_states]
        quantized_tracker_states = [
            state
            for state in self.quantization_states.values()
            if state.get("quantized_host_archive") is not None
        ]
        quantized_tracker_summaries = [
            state["quantized_host_archive"].summary()
            for state in quantized_tracker_states
        ]
        all_tracker_summaries = tracker_summaries + quantized_tracker_summaries
        working_set_accounting = aggregate_working_sets(all_tracker_summaries)
        working_set_accounting["mode_counts"] = {
            mode: sum(1 for state in tracker_states if state.get("working_set_mode") == mode)
            for mode in ("irreversible", "host_backed")
        }
        working_set_accounting["mode_counts"]["quantized_host_backed"] = len(
            quantized_tracker_states
        )
        working_set_accounting["note"] = (
            "Logical working-set accounting; full tensors remain materialized in this prototype."
            if all_tracker_summaries
            else "No tracked working-set mode is active."
        )

        archive_summaries = [
            state["working_set_tracker"].summary()
            for state in tracker_states
            if state.get("working_set_mode") == "host_backed"
        ]
        archive_summaries += quantized_tracker_summaries
        hierarchical_cache = aggregate_working_sets(archive_summaries)
        hierarchical_cache.update({
            "mode": "logical_host_archive" if archive_summaries else "disabled",
            "archives": len(archive_summaries),
            "token_selective_archives": sum(
                1 for state in tracker_states if state.get("working_set_mode") == "host_backed"
            ),
            "quantized_archives": len(quantized_tracker_summaries),
            "quantized_precision_transitions": int(
                sum(
                    item.get("precision_transitions", 0)
                    for item in quantized_tracker_summaries
                )
            ),
            "quantized_host_replica_catalog_bytes": int(
                sum(
                    item.get("host_replica_catalog_bytes", 0)
                    for item in quantized_tracker_summaries
                )
            ),
            "note": (
                "Algorithmic simulation only: full K/V tensors remain materialized; "
                "Host archive and H2D traffic are logical accounting."
                if archive_summaries
                else "No host-backed working-set rule is active."
            ),
        })
        return {
            "adapter": "UniCacheHookPolicy",
            "enabled": self.enabled,
            "run_id": self.run_id,
            "phase": self.phase,
            "branch": self.branch,
            "cfg_branch": self.cfg_branch,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "num_executable_steps": len(self.steps) + len(self.current_state_steps),
            "executable_operators": sorted(executable_operators),
            "hh_state_entries": len(self.hh_scores),
            "h2o_state_entries": len(self.h2o_states),
            "quantization_state_entries": len(self.quantization_states),
            "topk_state_entries": len(self.topk_state),
            "classic_state_entries": len(self.classic_states),
            "lifetime_state_entries": len(self.lifetime_states),
            "block_metric_state_entries": len(self.block_metric_states),
            "block_summary_metadata": {
                "entries": len(self.block_summary_states),
                "builds": int(
                    sum(
                        int(value.get("builds", 0))
                        for value in self.block_summary_states.values()
                    )
                ),
                "bytes": int(
                    sum(
                        int(value.get("metadata", {}).get("metadata_bytes", 0))
                        for value in self.block_summary_states.values()
                    )
                ),
                "note": (
                    "Logical phase-1 metadata remains device-resident; physical Host "
                    "placement is not implemented."
                ),
            },
            "block_metrics": [
                {"scope": list(key), **jsonable(value)}
                for key, value in sorted(self.block_metric_states.items(), key=lambda item: str(item[0]))
            ],
            "current_state": {
                "operator": self.current_state_decision.operator
                if self.current_state_decision is not None
                else None,
                "decision": jsonable(self.current_state_decision),
                "attention_score_entries": len(self.current_state_attention_scores),
                "step_decision_entries": len(self.current_state_step_decisions),
                "step_decisions_tail": [
                    jsonable(value)
                    for _, value in list(self.current_state_step_decisions.items())[-16:]
                ],
                "model_stats": jsonable(self.current_state_model_stats),
            },
            "processing_totals": dict(processing_totals),
            "budget_accounting": self._budget_accounting_summary(),
            "logical_storage": {
                "latest_unique_snapshots": len(self.storage_snapshots),
                "totals": dict(logical_totals),
                "snapshots": dict(self.storage_snapshots),
            },
            "working_set_accounting": working_set_accounting,
            "hierarchical_cache": hierarchical_cache,
            "realized_memory": {
                "storage_mode": storage_mode,
                "physical_kv_bytes_retired": retired_bytes,
                "physical_segment_full_bf16_bytes": physical_segment_full_bytes,
                "physical_segment_resident_bytes": physical_segment_bytes,
                "actual_tensor_bytes_saved": realized_saved_bytes,
                "actual_physical_retained_ratio": float(
                    physical_segment_bytes / max(physical_segment_full_bytes, 1)
                )
                if physical_segment_full_bytes
                else None,
                "physical_segment_byte_breakdown": {
                    name: int(
                        self.realized_storage_totals.get(
                            f"physical_segment_{name}", 0
                        )
                    )
                    for name in (
                        "payload_bytes",
                        "scale_bytes",
                        "minimum_bytes",
                        "residual_bytes",
                        "indices_bytes",
                        "score_bytes",
                    )
                },
                "physical_cache_types": physical_cache_types,
                "max_denoise_query_tokens": int(
                    self.realized_storage_totals.get("max_denoise_query_tokens", 0)
                ),
                "fused_mixed_attention_calls": int(
                    self.realized_storage_totals.get("fused_mixed_attention_calls", 0)
                ),
                "fused_mixed_output_only_calls": int(
                    self.realized_storage_totals.get(
                        "fused_mixed_output_only_calls", 0
                    )
                ),
                "fused_mixed_dual_stream_calls": int(
                    self.realized_storage_totals.get(
                        "fused_mixed_dual_stream_calls", 0
                    )
                ),
                "fused_mixed_overlap_calls": int(
                    self.realized_storage_totals.get("fused_mixed_overlap_calls", 0)
                ),
                "mixed_workspace_allocations": int(
                    self.realized_storage_totals.get(
                        "mixed_workspace_allocations", 0
                    )
                ),
                "mixed_workspace_reuses": int(
                    self.realized_storage_totals.get("mixed_workspace_reuses", 0)
                ),
                "mixed_workspace_peak_bytes": int(
                    self.realized_storage_totals.get(
                        "mixed_workspace_peak_bytes", 0
                    )
                ),
                "static_preparation_bypass_calls": int(
                    self.realized_storage_totals.get(
                        "static_preparation_bypass_calls", 0
                    )
                ),
                "fused_mixed_rejections": {
                    key.removeprefix("fused_mixed_reject_"): int(value)
                    for key, value in self.realized_storage_totals.items()
                    if key.startswith("fused_mixed_reject_")
                },
                "allocated_bytes_saved": 0,
                "evicted_tokens": int(self.realized_storage_totals.get("evicted_tokens", 0)),
                "latest_cache_bytes": int(self.realized_storage_totals.get("latest_cache_bytes", 0)),
                "note": memory_note,
            },
            "stats": {key: dict(value) for key, value in self.stats.items()},
            "coverage": self._coverage_summary(),
            "runtime_warnings": list(self.runtime_warnings),
            "selection_decisions": list(self.decision_history[-64:]),
            "tensor_shape_trace": list(self.tensor_shape_trace),
            "context_history_tail": list(self.context_history[-32:]),
            "plan_summary": jsonable(self.bundle.execution_plan.summaries),
        }

    def _record_attention_coverage(
        self,
        key_type_ids: torch.Tensor | None,
        key_len: int,
    ) -> None:
        self.coverage_totals["total_attention_calls"] += 1
        self.coverage_totals["total_tokens_seen"] += key_len
        if key_type_ids is None:
            self.coverage_totals["untyped_calls"] += 1
            self.coverage_totals["untyped_tokens"] += key_len
            return
        reverse_types = {value: name for name, value in KV_TYPE_NAME_TO_ID.items()}
        values, counts = torch.unique(key_type_ids.detach().to("cpu"), return_counts=True)
        for raw_value, raw_count in zip(values.tolist(), counts.tolist()):
            type_id = int(raw_value)
            count = int(raw_count)
            name = reverse_types.get(type_id)
            if name is None or name == "unknown":
                self.coverage_totals["unknown_tokens"] += count
                self.coverage_unknown_ids[str(type_id)] += count
            else:
                self.coverage_type_counts[name] += count

    def _coverage_summary(self) -> dict[str, Any]:
        total_tokens = int(self.coverage_totals.get("total_tokens_seen", 0))
        unknown_tokens = int(self.coverage_totals.get("unknown_tokens", 0))
        warnings = []
        if unknown_tokens:
            warnings.append(
                f"Observed {unknown_tokens} unknown KV type token(s) across attention calls"
            )
        for step_id, values in self.coverage_rules.items():
            if values.get("eligible_calls", 0) and not values.get("hit_calls", 0):
                warnings.append(f"Execution step {step_id} never matched a non-empty segment")
        return {
            "total_attention_calls": int(self.coverage_totals.get("total_attention_calls", 0)),
            "total_tokens_seen": total_tokens,
            "unknown_tokens": unknown_tokens,
            "unknown_ratio": (float(unknown_tokens) / total_tokens) if total_tokens else 0.0,
            "untyped_calls": int(self.coverage_totals.get("untyped_calls", 0)),
            "untyped_tokens": int(self.coverage_totals.get("untyped_tokens", 0)),
            "token_type_counts": {
                key: int(value) for key, value in sorted(self.coverage_type_counts.items())
            },
            "unknown_type_ids": {
                key: int(value) for key, value in sorted(self.coverage_unknown_ids.items())
            },
            "rules": {
                key: {name: int(value) for name, value in values.items()}
                for key, values in sorted(self.coverage_rules.items())
            },
            "warnings": warnings,
        }

    def _record_decision(self, step: ExecutionStep, runtime: RuntimeContext, decision: Any) -> None:
        if not self.collect_diagnostics:
            return
        row = {
            "run_id": runtime.run_id,
            "phase": runtime.phase,
            "branch": runtime.branch,
            "cfg_branch": runtime.cfg_branch,
            "layer_idx": int(runtime.layer_idx),
            "batch_idx": int(runtime.batch_idx),
            "step_id": step.id,
            "segment_id": step.segment_id,
            "action": decision.action,
            "reason": decision.reason,
            "budget": decision.budget,
            "protected_tokens": int(decision.protected_mask.sum().item())
            if decision.protected_mask is not None
            else 0,
            "kept_tokens": int(decision.keep_mask.sum().item()) if decision.keep_mask is not None else None,
            "metadata": jsonable(decision.metadata),
        }
        self.decision_history.append(row)
        if len(self.decision_history) > self.max_context_history:
            self.decision_history = self.decision_history[-self.max_context_history :]

    @staticmethod
    def _pending_decision_key(runtime: RuntimeContext, segment_id: str) -> tuple[Any, ...]:
        return (
            runtime.run_id,
            runtime.phase,
            runtime.branch,
            runtime.cfg_branch,
            int(runtime.layer_idx),
            int(runtime.batch_idx),
            segment_id,
        )

    @staticmethod
    def _attention_scores(q_i: torch.Tensor, k_i: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        scores = gqa_query_key_logits(q_i, k_i) / math.sqrt(q_i.shape[-1])
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        return scores

    def _runtime_step_matches(
        self,
        step: ExecutionStep,
        runtime: RuntimeContext,
        *,
        preflight: bool = False,
    ) -> bool:
        segment = self.segments_by_id[step.segment_id]
        if not segment.member_segment_ids:
            return runtime_step_matches(step, segment, runtime, preflight=preflight)
        return any(
            runtime_step_matches(step, self.segments_by_id[member_id], runtime, preflight=preflight)
            for member_id in segment.member_segment_ids
        )

    def _atomic_segment_mask(
        self,
        segment_id: str,
        key_type_ids: torch.Tensor | None,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        segment = self.segments_by_id[segment_id]
        kv_types = segment.selector.kv_types
        if kv_types == WILDCARD:
            mask = torch.ones(key_len, device=device, dtype=torch.bool)
        elif key_type_ids is None:
            mask = torch.zeros(key_len, device=device, dtype=torch.bool)
        else:
            type_ids = [KV_TYPE_NAME_TO_ID[name] for name in kv_types if name in KV_TYPE_NAME_TO_ID]
            if not type_ids:
                return torch.zeros(key_len, device=device, dtype=torch.bool)
            wanted = torch.tensor(type_ids, device=device, dtype=key_type_ids.dtype)
            mask = torch.isin(key_type_ids.to(device=device), wanted)

        token_ranges = segment.selector.token_ranges
        if token_ranges != WILDCARD:
            range_mask = torch.zeros(key_len, device=device, dtype=torch.bool)
            for item in token_ranges:
                start = max(0, int(item.get("start", 0)))
                end = min(key_len, int(item.get("end", key_len)))
                range_mask[start:end] = True
            mask &= range_mask
        return mask

    def _segment_mask(
        self,
        step: ExecutionStep,
        runtime: RuntimeContext,
        key_type_ids: torch.Tensor | None,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        segment = self.segments_by_id[step.segment_id]
        if not segment.member_segment_ids:
            return self._atomic_segment_mask(segment.id, key_type_ids, key_len, device)

        mask = torch.zeros(key_len, device=device, dtype=torch.bool)
        for member_id in segment.member_segment_ids:
            member = self.segments_by_id[member_id]
            if runtime_step_matches(step, member, runtime):
                mask |= self._atomic_segment_mask(member_id, key_type_ids, key_len, device)
        return mask

    def _segment_token_count(
        self,
        *,
        step: ExecutionStep,
        runtime: RuntimeContext,
        segment_mask: torch.Tensor,
        key_len: int,
    ) -> int:
        if runtime.phase != "denoise":
            return int(segment_mask.sum().item())
        key = (
            runtime.run_id,
            runtime.phase,
            runtime.branch,
            runtime.cfg_branch,
            int(runtime.batch_idx),
            step.segment_id,
            int(key_len),
        )
        cached = self.segment_count_cache.get(key)
        if cached is not None:
            self.realized_storage_totals["segment_count_reuses"] += 1
            return cached
        count = int(segment_mask.sum().item())
        self.segment_count_cache[key] = count
        self.realized_storage_totals["segment_count_builds"] += 1
        return count

    def _segment_contiguous_span(
        self,
        *,
        step: ExecutionStep,
        runtime: RuntimeContext,
        segment_mask: torch.Tensor,
        segment_tokens: int,
        key_len: int,
    ) -> tuple[int, int] | None:
        if runtime.phase != "denoise" or segment_tokens <= 0:
            return None
        key = (
            runtime.run_id,
            runtime.phase,
            runtime.branch,
            runtime.cfg_branch,
            int(runtime.batch_idx),
            step.segment_id,
            int(key_len),
        )
        if key in self.segment_span_cache:
            self.realized_storage_totals["segment_span_reuses"] += 1
            return self.segment_span_cache[key]
        indexes = torch.nonzero(segment_mask, as_tuple=False).flatten()
        start, last = (
            int(item)
            for item in indexes[[0, int(indexes.numel()) - 1]].to("cpu").tolist()
        )
        span = (start, last + 1) if last - start + 1 == segment_tokens else None
        self.segment_span_cache[key] = span
        self.realized_storage_totals["segment_span_builds"] += 1
        return span

    def _resolve_params(self, step: ExecutionStep, ctx: RuntimeContext) -> dict[str, Any]:
        params = dict(step.params)
        if step.scheduler is not None:
            scheduler = self.registry.require_scheduler(step.scheduler.name)
            segment = self.segments_by_id.get(step.segment_id)
            params.update(scheduler.resolve(step.scheduler.params, ctx, segment))
        return params

    def _record_budget_allocation(
        self,
        step: ExecutionStep,
        runtime: RuntimeContext,
        segment_mask: torch.Tensor,
        params: dict[str, Any],
    ) -> tuple[Any, ...] | None:
        if not bool(params.get("dynamic_budget", False)):
            return None
        segment_tokens = int(segment_mask.sum().item())
        target_ratio = float(params.get("target_budget_ratio", 1.0))
        requested_tokens = min(
            segment_tokens,
            max(0, int(round(target_ratio * segment_tokens))),
        )
        layer_chunk = int(params.get("current_layer_chunk", params.get("metric_layer_chunk", -1)))
        step_chunk = int(params.get("current_step_chunk", -1))
        key = (
            runtime.run_id,
            runtime.phase,
            runtime.branch,
            runtime.cfg_branch,
            int(runtime.batch_idx),
            step.id,
            step.segment_id,
            layer_chunk,
            step_chunk,
        )
        row = self.budget_allocation_records.get(key)
        if row is None:
            row = {
                "run_id": runtime.run_id,
                "task_type": runtime.task_type,
                "phase": runtime.phase,
                "branch": runtime.branch,
                "cfg_branch": runtime.cfg_branch,
                "batch_idx": int(runtime.batch_idx),
                "step_id": step.id,
                "segment_id": step.segment_id,
                "operator": step.operator,
                "scheduler": step.scheduler.name if step.scheduler is not None else None,
                "step_chunk": step_chunk,
                "layer_chunk": layer_chunk,
                "metric_step_chunk": params.get("metric_step_chunk"),
                "metric_layer_chunk": params.get("metric_layer_chunk"),
                "allocation_source": params.get("budget_source"),
                "pool_segments": list(params.get("budget_segments", [step.segment_id])),
                "configured_pool_segments": list(
                    params.get("configured_budget_segments", params.get("budget_segments", [step.segment_id]))
                ),
                "inactive_pool_segments": list(params.get("inactive_budget_segments", [])),
                "pool_retained_ratio": float(params.get("scheduled_total_budget_ratio", target_ratio)),
                "total_budget_source": params.get("total_budget_source"),
                "full_kv_calibration": bool(params.get("full_kv_calibration", False)),
                "reference_conditional_attention_mass": params.get(
                    "reference_conditional_attention_mass"
                ),
                "previous_conditional_attention_mass": params.get(
                    "previous_conditional_attention_mass"
                ),
                "conditional_attention_mass_ratio": params.get(
                    "conditional_attention_mass_ratio"
                ),
                "pool_budget_token_equivalent": params.get("total_budget_token_equivalent"),
                "budget_weight_by_segment": dict(
                    params.get("budget_weight_by_segment", {}) or {}
                ),
                "budget_active_ema_mass_by_segment": dict(
                    params.get("budget_active_ema_mass_by_segment", {}) or {}
                ),
                "budget_mass_estimator_by_segment": dict(
                    params.get("budget_mass_estimator_by_segment", {}) or {}
                ),
                "budget_allocation_token_equivalent_by_segment": dict(
                    params.get(
                        "budget_allocation_token_equivalent_by_segment", {}
                    )
                    or {}
                ),
                "segment_capacity_tokens": segment_tokens,
                "observed_ema_attention_mass": params.get("observed_ema_attention_mass"),
                "requested_retained_ratio": target_ratio,
                "requested_budget_tokens": requested_tokens,
                "first_step_index": int(runtime.step_index or 0),
                "last_step_index": int(runtime.step_index or 0),
                "first_decode_index": runtime.decode_index,
                "last_decode_index": runtime.decode_index,
                "invocations": 0,
                "layers_seen": [],
                "allocation_revisions": [],
                "_usage_last": {},
                "_usage_totals": defaultdict(float),
                "_storage_last": {},
            }
            self.budget_allocation_records[key] = row
        signature = {
            "requested_retained_ratio": target_ratio,
            "requested_budget_tokens": requested_tokens,
            "observed_ema_attention_mass": params.get("observed_ema_attention_mass"),
            "allocation_source": params.get("budget_source"),
        }
        previous = row["allocation_revisions"][-1] if row["allocation_revisions"] else None
        if previous != signature:
            row["allocation_revisions"].append(signature)
        row["invocations"] += 1
        row["last_step_index"] = int(runtime.step_index or 0)
        row["last_decode_index"] = runtime.decode_index
        layer_idx = int(runtime.layer_idx)
        if layer_idx not in row["layers_seen"]:
            row["layers_seen"].append(layer_idx)
            row["layers_seen"].sort()
        row["segment_capacity_tokens"] = segment_tokens
        row["observed_ema_attention_mass"] = params.get("observed_ema_attention_mass")
        row["metric_step_chunk"] = params.get("metric_step_chunk")
        row["metric_layer_chunk"] = params.get("metric_layer_chunk")
        row["allocation_source"] = params.get("budget_source")
        row["pool_budget_token_equivalent"] = params.get(
            "total_budget_token_equivalent"
        )
        row["budget_weight_by_segment"] = dict(
            params.get("budget_weight_by_segment", {}) or {}
        )
        row["budget_active_ema_mass_by_segment"] = dict(
            params.get("budget_active_ema_mass_by_segment", {}) or {}
        )
        row["budget_mass_estimator_by_segment"] = dict(
            params.get("budget_mass_estimator_by_segment", {}) or {}
        )
        row["budget_allocation_token_equivalent_by_segment"] = dict(
            params.get("budget_allocation_token_equivalent_by_segment", {}) or {}
        )
        row["requested_retained_ratio"] = target_ratio
        row["requested_budget_tokens"] = requested_tokens
        if step.operator == "kivi_quantization":
            row["_usage_last"].update(
                {
                    "interpretation": "retained_storage_precision",
                    "bits": int(params.get("bits", 4)),
                    "k_bits": int(params.get("k_bits", params.get("bits", 4))),
                    "v_bits": int(params.get("v_bits", params.get("bits", 4))),
                    "group_size": int(params.get("group_size", 32)),
                    "residual_length": int(params.get("residual_length", 32)),
                    "lookup_storage_ratio": params.get("kivi_lookup_storage_ratio"),
                    "estimated_storage_ratio": params.get("estimated_storage_ratio"),
                }
            )
        elif step.operator in {"h2o_attention_mask", "h2o_segment_attention_mask"}:
            row["_usage_last"]["interpretation"] = "per_head_token_retention"
            row["_usage_last"]["applies_on_next_call"] = True
        return key

    def _record_budget_usage(
        self,
        key: tuple[Any, ...] | None,
        values: dict[str, float],
    ) -> None:
        if key is None:
            return
        row = self.budget_allocation_records[key]
        usage = row["_usage_last"]
        mapping = {
            "h2o_requested_budget": "requested_budget_tokens",
            "h2o_effective_budget": "effective_budget_tokens",
            "h2o_budget_shortfall_tokens": "budget_shortfall_tokens",
            "h2o_heavy_budget": "heavy_budget_tokens",
            "h2o_recent_budget": "recent_budget_tokens",
            "h2o_masked_tokens": "masked_query_head_tokens",
            "h2o_next_kept_head_tokens": "next_kept_query_head_tokens",
            "quantized_tokens": "quantized_tokens",
            "protected_tokens_preserved": "protected_tokens_preserved",
            "processed_input_kv_bytes": "processed_input_kv_bytes",
        }
        for source, target in mapping.items():
            if source not in values:
                continue
            value = values[source]
            usage[target] = int(value) if float(value).is_integer() else float(value)
            row["_usage_totals"][target] += float(value)

    def _record_budget_storage(
        self,
        key: tuple[Any, ...] | None,
        values: dict[str, float],
    ) -> None:
        if key is None:
            return
        self.budget_allocation_records[key]["_storage_last"] = {
            name: int(value) if float(value).is_integer() else float(value)
            for name, value in values.items()
        }

    def _budget_accounting_summary(self) -> dict[str, Any]:
        allocations = []
        by_cache_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        overall: dict[str, float] = defaultdict(float)
        allocation_source_counts: dict[str, int] = defaultdict(int)
        for _, stored in sorted(self.budget_allocation_records.items(), key=lambda item: str(item[0])):
            row = {
                name: value
                for name, value in stored.items()
                if not name.startswith("_")
            }
            usage = dict(stored["_usage_last"])
            usage.update(stored["_storage_last"])
            original_storage_bytes = float(usage.get("original_storage_bytes", 0.0))
            ideal_storage_bytes = float(usage.get("ideal_total_storage_bytes", 0.0))
            if original_storage_bytes > 0.0:
                usage["effective_storage_ratio"] = ideal_storage_bytes / original_storage_bytes
            row["usage"] = usage
            row["usage_totals"] = dict(stored["_usage_totals"])
            allocations.append(jsonable(row))
            allocation_source_counts[str(row.get("allocation_source") or "unknown")] += 1

            segment = str(row["segment_id"])
            aggregate = by_cache_type[segment]
            invocations = int(row["invocations"])
            aggregate["allocation_chunks"] += 1
            aggregate["invocations"] += invocations
            aggregate["capacity_token_invocations"] += float(row["segment_capacity_tokens"] * invocations)
            aggregate["requested_budget_token_invocations"] += float(
                row["requested_budget_tokens"] * invocations
            )
            overall["managed_capacity_token_invocations"] += float(
                row["segment_capacity_tokens"] * invocations
            )
            overall["requested_budget_token_invocations"] += float(
                row["requested_budget_tokens"] * invocations
            )
            aggregate["effective_budget_token_invocations"] += float(
                stored["_usage_totals"].get("effective_budget_tokens", 0.0)
            )
            aggregate["budget_shortfall_token_invocations"] += float(
                stored["_usage_totals"].get("budget_shortfall_tokens", 0.0)
            )
            aggregate["heavy_budget_token_invocations"] += float(
                stored["_usage_totals"].get("heavy_budget_tokens", 0.0)
            )
            aggregate["recent_budget_token_invocations"] += float(
                stored["_usage_totals"].get("recent_budget_tokens", 0.0)
            )
            aggregate["quantized_token_invocations"] += float(
                stored["_usage_totals"].get("quantized_tokens", 0.0)
            )
            if original_storage_bytes > 0.0:
                aggregate["original_storage_byte_invocations"] += (
                    original_storage_bytes * invocations
                )
                aggregate["ideal_storage_byte_invocations"] += (
                    ideal_storage_bytes * invocations
                )
                overall["original_storage_byte_invocations"] += (
                    original_storage_bytes * invocations
                )
                overall["ideal_storage_byte_invocations"] += (
                    ideal_storage_bytes * invocations
                )
        normalized = {}
        for segment, values in sorted(by_cache_type.items()):
            row = {
                name: int(value) if float(value).is_integer() else float(value)
                for name, value in values.items()
            }
            capacity = float(row.get("capacity_token_invocations", 0.0))
            if capacity:
                row["mean_requested_retained_ratio"] = float(
                    row.get("requested_budget_token_invocations", 0.0) / capacity
                )
                if row.get("effective_budget_token_invocations", 0.0):
                    row["mean_effective_retained_ratio"] = float(
                        row["effective_budget_token_invocations"] / capacity
                    )
            original_storage = float(row.get("original_storage_byte_invocations", 0.0))
            if original_storage:
                row["mean_effective_storage_ratio"] = float(
                    row.get("ideal_storage_byte_invocations", 0.0) / original_storage
                )
            normalized[segment] = row
        average_budget = {
            name: int(value) if float(value).is_integer() else float(value)
            for name, value in overall.items()
        }
        capacity = float(average_budget.get("managed_capacity_token_invocations", 0.0))
        if capacity:
            average_budget["mean_requested_retained_ratio"] = float(
                average_budget.get("requested_budget_token_invocations", 0.0) / capacity
            )
        original_storage = float(
            average_budget.get("original_storage_byte_invocations", 0.0)
        )
        if original_storage:
            average_budget["mean_theoretical_storage_ratio"] = float(
                average_budget.get("ideal_storage_byte_invocations", 0.0)
                / original_storage
            )
            average_budget["mean_theoretical_compression_ratio"] = float(
                1.0 - average_budget["mean_theoretical_storage_ratio"]
            )
        dynamic_sources = {
            name: count
            for name, count in allocation_source_counts.items()
            if name.startswith("shared_budget_from_")
        }
        warnings = []
        if allocations and not dynamic_sources:
            warnings.append(
                "Dynamic budget allocation never activated; every allocation used "
                "calibration, fallback, or an unknown source. Check that the run "
                "crosses a step/layer chunk boundary and that bootstrap is enabled "
                "for short decoding tasks."
            )
        return {
            "schema_version": 1,
            "allocation_granularity": "task/phase/branch/cfg/sample/step_chunk/layer_chunk/cache_type",
            "allocations": allocations,
            "by_cache_type": normalized,
            "average_budget": average_budget,
            "allocation_source_counts": dict(sorted(allocation_source_counts.items())),
            "dynamic_allocation_activated": bool(dynamic_sources),
            "warnings": warnings,
        }

    def _record(
        self,
        step: ExecutionStep,
        runtime: RuntimeContext,
        segment_mask: torch.Tensor,
        values: dict[str, float],
    ) -> None:
        key = (
            f"{runtime.run_id}|{runtime.phase}|{runtime.branch}|{runtime.cfg_branch}|"
            f"B{runtime.batch_idx}|{step.operator}|{step.segment_id}|L{int(runtime.layer_idx):02d}"
        )
        row = self.stats[key]
        row["calls"] += 1
        row["segment_tokens_seen"] += float(segment_mask.sum().item())
        for name, value in values.items():
            row[name] += float(value)

    def _set_storage_snapshot(
        self,
        step: ExecutionStep,
        runtime: RuntimeContext,
        values: dict[str, float],
    ) -> None:
        segment = self.segments_by_id[step.segment_id]
        owner_branches = segment.selector.branches
        if isinstance(owner_branches, list):
            owner_key = ",".join(sorted(str(item) for item in owner_branches))
        else:
            owner_key = str(owner_branches)
        key = (
            f"{runtime.run_id}|owner={owner_key}|cfg={runtime.cfg_branch}|"
            f"B{runtime.batch_idx}|{step.segment_id}|L{int(runtime.layer_idx):02d}|{step.operator}"
        )
        self.storage_snapshots[key] = {name: float(value) for name, value in values.items()}


def compile_plan_only_hook_policy(bundle: PlanBundle) -> PlanOnlyHookPolicy:
    return PlanOnlyHookPolicy(execution_plan=bundle.execution_plan)


def compile_unicache_hook_policy(
    bundle: PlanBundle,
    *,
    enabled: bool = True,
    registry: KVRegistry | None = None,
) -> UniCacheHookPolicy:
    return UniCacheHookPolicy(bundle=bundle, enabled=enabled, registry=registry)
