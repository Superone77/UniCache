"""Representative token-retention baselines for UniCache."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..core.capabilities import CapabilityDescriptor
from ..core.decisions import SelectionDecision
from ..core.registry import OperatorExecutionContext, ProcessingOperator


def _attention_probabilities(ctx: OperatorExecutionContext, query_window: int) -> torch.Tensor:
    q = ctx.q.transpose(0, 1).float()
    k = ctx.k.transpose(0, 1).float()
    window = min(max(1, int(query_window)), int(q.shape[1]))
    q = q[:, -window:, :]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(ctx.q.shape[-1])
    if ctx.attn_mask is not None:
        scores = scores.masked_fill(
            ~ctx.attn_mask[-window:, :].unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
    return torch.softmax(scores, dim=-1)


def _pool_scores(scores: torch.Tensor, *, kernel_size: int, pooling: str) -> torch.Tensor:
    if scores.numel() == 0 or kernel_size <= 1:
        return scores
    kernel = min(int(kernel_size), int(scores.shape[-1]))
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    values = scores.unsqueeze(1)
    if pooling == "avgpool":
        return F.avg_pool1d(values, kernel, stride=1, padding=kernel // 2).squeeze(1)
    if pooling == "maxpool":
        return F.max_pool1d(values, kernel, stride=1, padding=kernel // 2).squeeze(1)
    raise ValueError(f"Unsupported observation-window pooling: {pooling}")


def _state_key(ctx: OperatorExecutionContext, method: str) -> tuple[object, ...]:
    return (
        method,
        ctx.runtime.run_id,
        ctx.runtime.phase,
        ctx.runtime.branch,
        ctx.runtime.cfg_branch,
        int(ctx.runtime.layer_idx),
        int(ctx.runtime.batch_idx),
        ctx.step.segment_id,
        ctx.step.id,
    )


def _validated_keep_ratio(params: dict) -> float | None:
    if "target_keep_ratio" not in params:
        return None
    ratio = float(params["target_keep_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"target_keep_ratio must be in (0, 1], got {ratio}")
    return ratio


def _ratio_capacity(segment_len: int, ratio: float) -> int:
    return min(segment_len, max(1, int(math.ceil(segment_len * ratio))))


def _record_mask_storage(
    ctx: OperatorExecutionContext,
    *,
    segment_len: int,
    keep: torch.Tensor,
) -> None:
    num_heads = int(keep.shape[0])
    storage_num_kv_heads = int(
        ctx.storage.layout.num_kv_heads
        if ctx.storage is not None
        else (ctx.storage_num_kv_heads or num_heads)
    )
    head_dim = int(ctx.k.shape[-1])
    elem_bytes = int(ctx.k.element_size())
    selected = keep & ctx.segment_mask.unsqueeze(0)
    kept_head_tokens = int(selected.sum().item())
    mean_keep_ratio = float(kept_head_tokens / max(num_heads * segment_len, 1))
    original_storage = float(
        segment_len * storage_num_kv_heads * head_dim * 2 * elem_bytes
    )
    ideal_payload = float(original_storage * mean_keep_ratio)
    metadata_bytes = float(kept_head_tokens * 4)

    if num_heads % storage_num_kv_heads == 0:
        repeat = num_heads // storage_num_kv_heads
        union_head_tokens = int(
            selected.reshape(storage_num_kv_heads, repeat, int(ctx.k.shape[0]))
            .any(dim=1)
            .sum()
            .item()
        )
        gqa_union_payload = float(union_head_tokens * head_dim * 2 * elem_bytes)
    else:
        repeat = 1
        union_head_tokens = kept_head_tokens
        gqa_union_payload = ideal_payload

    ctx.set_storage_snapshot(
        {
            "segment_tokens": segment_len,
            "attention_num_heads": num_heads,
            "storage_num_kv_heads": storage_num_kv_heads,
            "gqa_repeat": repeat,
            "selected_query_head_tokens": kept_head_tokens,
            "gqa_union_kv_head_tokens": union_head_tokens,
            "original_storage_bytes": original_storage,
            "ideal_packed_payload_bytes": ideal_payload,
            "ideal_index_metadata_bytes": metadata_bytes,
            "ideal_metadata_bytes": metadata_bytes,
            "ideal_total_storage_bytes": ideal_payload + metadata_bytes,
            "gqa_union_payload_bytes": gqa_union_payload,
            "gqa_union_total_storage_bytes": gqa_union_payload + metadata_bytes,
        }
    )


def _record_token_storage(
    ctx: OperatorExecutionContext,
    *,
    segment_len: int,
    kept_tokens: int,
    original_segment_len: int | None = None,
) -> None:
    storage_num_kv_heads = int(
        ctx.storage.layout.num_kv_heads
        if ctx.storage is not None
        else (ctx.storage_num_kv_heads or int(ctx.k.shape[1]))
    )
    head_dim = int(ctx.k.shape[-1])
    elem_bytes = int(ctx.k.element_size())
    bytes_per_token = storage_num_kv_heads * head_dim * 2 * elem_bytes
    baseline_tokens = segment_len if original_segment_len is None else original_segment_len
    original_storage = float(baseline_tokens * bytes_per_token)
    ideal_payload = float(kept_tokens * bytes_per_token)
    metadata_bytes = float(kept_tokens * 4)
    ctx.set_storage_snapshot(
        {
            "segment_tokens": baseline_tokens,
            "current_segment_tokens": segment_len,
            "kept_tokens": kept_tokens,
            "storage_num_kv_heads": storage_num_kv_heads,
            "original_storage_bytes": original_storage,
            "ideal_packed_payload_bytes": ideal_payload,
            "ideal_index_metadata_bytes": metadata_bytes,
            "ideal_metadata_bytes": metadata_bytes,
            "ideal_total_storage_bytes": ideal_payload + metadata_bytes,
            "gqa_union_payload_bytes": ideal_payload,
            "gqa_union_total_storage_bytes": ideal_payload + metadata_bytes,
        }
    )


class StreamingLLMEvictionOperator(ProcessingOperator):
    """Keep segment-local sink and recent tokens, retiring the middle."""

    name = "streamingllm_eviction"
    family = "eviction"
    stages = {"attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        layouts=frozenset({"attention_view", "dense"}),
        devices=frozenset({"cpu", "cuda", "mps"}),
        requires=frozenset({"segment_mask", "protected_mask", "cache_storage_view"}),
        produces_selection=True,
        stateful=True,
        mutation_scope="persistent_cache",
        physical_storage_mutation=True,
    )

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        _validated_keep_ratio(out)
        for name in ("sink_size", "recent_size", "max_capacity_prompt"):
            if name in out and int(out[name]) < 0:
                raise ValueError(f"{self.name} {name} must be non-negative")
        return out

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        segment_idx = torch.nonzero(ctx.segment_mask, as_tuple=False).flatten()
        segment_len = int(segment_idx.numel())
        if segment_len == 0:
            return ctx

        state_key = _state_key(ctx, self.name)
        state = ctx.state.get(state_key)
        target_ratio = _validated_keep_ratio(ctx.params)
        target_capacity = None
        if state is not None:
            target_capacity = int(state["capacity"])
        elif target_ratio is not None:
            target_capacity = _ratio_capacity(segment_len, target_ratio)
        sink_size = min(
            segment_len if target_capacity is None else target_capacity,
            int(ctx.params.get("sink_size", 4)),
        )
        configured_recent = int(ctx.params.get("recent_size", 0))
        capacity = int(ctx.params.get("max_capacity_prompt", 0))
        if target_capacity is not None:
            recent_size = (
                configured_recent
                if configured_recent > 0
                else target_capacity - sink_size
            )
            recent_size = min(max(0, target_capacity - sink_size), recent_size)
        elif configured_recent > 0:
            recent_size = configured_recent
        elif capacity > 0:
            recent_size = max(0, capacity - sink_size)
        else:
            recent_size = 128
        recent_size = min(max(0, segment_len - sink_size), recent_size)

        selected = ctx.protected_mask & ctx.segment_mask
        if sink_size:
            selected[segment_idx[:sink_size]] = True
        if recent_size:
            selected[segment_idx[-recent_size:]] = True

        keep_mask = torch.ones(int(ctx.k.shape[0]), device=ctx.k.device, dtype=torch.bool)
        keep_mask[ctx.segment_mask] = False
        keep_mask[selected] = True
        kept = int((keep_mask & ctx.segment_mask).sum().item())
        if state is None:
            state = {
                "initial_length": segment_len,
                "capacity": kept,
                "sink_size": sink_size,
                "recent_size": recent_size,
            }
            ctx.state[state_key] = state
        _record_token_storage(
            ctx,
            segment_len=segment_len,
            kept_tokens=kept,
            original_segment_len=int(state["initial_length"]),
        )
        ctx.decision = SelectionDecision(
            token_count=int(ctx.k.shape[0]),
            protected_mask=ctx.protected_mask & ctx.segment_mask,
            keep_mask=keep_mask,
            budget=kept,
            reason="streamingllm_sink_plus_recent",
            metadata={
                "segment_tokens": segment_len,
                "sink_tokens": sink_size,
                "recent_tokens": recent_size,
                "kept_tokens": kept,
                "evicted_tokens": segment_len - kept,
                "physical_eviction": True,
                "selection_reused": int(state["initial_length"]) != segment_len,
            },
        )
        ctx.record(
            {
                "streaming_segment_tokens": segment_len,
                "streaming_sink_tokens": sink_size,
                "streaming_recent_tokens": recent_size,
                "streaming_evicted_tokens": segment_len - kept,
                "streaming_reused_selection_calls": int(int(state["initial_length"]) != segment_len),
            }
        )
        return ctx


class ObservationWindowAttentionMaskOperator(ProcessingOperator):
    """Shared mask-only implementation for SnapKV and PyramidKV."""

    family = "eviction"
    stages = {"attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        layouts=frozenset({"attention_view", "dense"}),
        devices=frozenset({"cpu", "cuda", "mps"}),
        requires=frozenset({"query", "key", "segment_mask", "protected_mask"}),
        produces_selection=True,
        stateful=True,
        mutation_scope="attention_mask",
        physical_storage_mutation=False,
    )

    def __init__(self, name: str) -> None:
        self.name = name

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        _validated_keep_ratio(out)
        if int(out.get("observation_window", out.get("window_size", 64))) <= 0:
            raise ValueError(f"{self.name} observation_window must be positive")
        if int(out.get("kernel_size", 5)) <= 0:
            raise ValueError(f"{self.name} kernel_size must be positive")
        if out.get("pooling", "avgpool") not in {"avgpool", "maxpool"}:
            raise ValueError(f"{self.name} pooling must be avgpool or maxpool")
        return out

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        segment_idx = torch.nonzero(ctx.segment_mask, as_tuple=False).flatten()
        segment_len = int(segment_idx.numel())
        if segment_len == 0:
            return ctx

        state_key = _state_key(ctx, self.name)
        previous = ctx.state.get(state_key)
        if previous is not None and not bool(ctx.params.get("refresh_selection", False)):
            aligned = torch.ones(
                (int(ctx.q.shape[1]), int(ctx.k.shape[0])),
                dtype=torch.bool,
                device=ctx.k.device,
            )
            heads = min(int(aligned.shape[0]), int(previous.shape[0]))
            tokens = min(int(aligned.shape[1]), int(previous.shape[1]))
            aligned[:heads, :tokens] = previous[:heads, :tokens].to(device=ctx.k.device)
            ctx.attention_keep_mask_override = aligned
            ctx.record({f"{self.name}_reused_mask_calls": 1})
            return ctx

        window = int(ctx.params.get("observation_window", ctx.params.get("window_size", 64)))
        target_ratio = _validated_keep_ratio(ctx.params)
        if target_ratio is None:
            capacity = int(ctx.params.get("keep_k", ctx.params.get("max_capacity_prompt", segment_len)))
        else:
            average_capacity = _ratio_capacity(segment_len, target_ratio)
            capacity = average_capacity
            if self.name == "pyramidkv_attention_mask":
                beta = max(1.0, float(ctx.params.get("beta", 20.0)))
                configured_recent = int(ctx.params.get("recent_size", min(window, average_capacity)))
                pyramid_recent = min(configured_recent, max(0, average_capacity - 1))
                compressible = max(1, average_capacity - pyramid_recent)
                min_budget = max(1, int(compressible // beta))
                max_budget = max(min_budget, int(compressible * 2 - min_budget))
                total_layers = max(1, int(ctx.runtime.total_layers))
                progress = (
                    0.0
                    if total_layers == 1
                    else min(1.0, max(0.0, ctx.runtime.layer_idx / float(total_layers - 1)))
                )
                selected = int(round(max_budget - (max_budget - min_budget) * progress))
                capacity = pyramid_recent + max(1, selected)
        recent_size = min(
            segment_len,
            capacity,
            int(ctx.params.get("recent_size", window)),
        )
        capacity = min(segment_len, max(recent_size, capacity))
        candidate_budget = max(0, capacity - recent_size)

        recent_idx = segment_idx[-recent_size:] if recent_size else segment_idx[:0]
        candidate_idx = segment_idx[:-recent_size] if recent_size else segment_idx
        candidate_budget = min(candidate_budget, int(candidate_idx.numel()))
        probs = _attention_probabilities(ctx, window)
        token_scores = probs.sum(dim=1)
        candidate_scores = token_scores[:, candidate_idx]
        candidate_scores = _pool_scores(
            candidate_scores,
            kernel_size=int(ctx.params.get("kernel_size", 5)),
            pooling=str(ctx.params.get("pooling", "avgpool")),
        )

        keep = torch.ones(
            (int(ctx.q.shape[1]), int(ctx.k.shape[0])),
            dtype=torch.bool,
            device=ctx.k.device,
        )
        keep[:, segment_idx] = False
        if recent_idx.numel():
            keep[:, recent_idx] = True
        protected_idx = torch.nonzero(ctx.protected_mask & ctx.segment_mask, as_tuple=False).flatten()
        if protected_idx.numel():
            keep[:, protected_idx] = True
        if candidate_budget > 0:
            top = torch.topk(candidate_scores, k=candidate_budget, dim=-1, largest=True).indices
            selected = candidate_idx[top]
            keep.scatter_(1, selected, True)

        ctx.state[state_key] = keep.detach()
        ctx.attention_keep_mask_override = keep
        _record_mask_storage(ctx, segment_len=segment_len, keep=keep)
        per_head = (keep & ctx.segment_mask.unsqueeze(0)).sum(dim=-1)
        ctx.decision = SelectionDecision(
            token_count=int(ctx.k.shape[0]),
            protected_mask=ctx.protected_mask & ctx.segment_mask,
            scores=token_scores.mean(dim=0),
            budget=int(round(float(per_head.float().mean().item()))),
            reason=f"{self.name}_observation_window_topk",
            metadata={
                "segment_tokens": segment_len,
                "observation_window": min(window, int(ctx.q.shape[0])),
                "recent_tokens": recent_size,
                "candidate_budget": candidate_budget,
                "per_head_kept_tokens": [int(item) for item in per_head.tolist()],
                "pooling": str(ctx.params.get("pooling", "avgpool")),
                "kernel_size": int(ctx.params.get("kernel_size", 5)),
                "physical_eviction": False,
                "application_mode": (
                    "prompt_observation" if ctx.runtime.phase in {"prefill", "text_decode"}
                    else "denoise_query_adaptation"
                ),
            },
        )
        ctx.record(
            {
                f"{self.name}_segment_tokens": segment_len,
                f"{self.name}_mean_kept_tokens": float(per_head.float().mean().item()),
                f"{self.name}_masked_head_tokens": int((~keep[:, segment_idx]).sum().item()),
            }
        )
        return ctx


class SnapKVAttentionMaskOperator(ObservationWindowAttentionMaskOperator):
    def __init__(self) -> None:
        super().__init__("snapkv_attention_mask")


class PyramidKVAttentionMaskOperator(ObservationWindowAttentionMaskOperator):
    def __init__(self) -> None:
        super().__init__("pyramidkv_attention_mask")
