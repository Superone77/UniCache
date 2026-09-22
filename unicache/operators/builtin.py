"""Built-in processing operators for UniCache."""

from __future__ import annotations

import math

import torch

from ..core.capabilities import CapabilityDescriptor
from ..core.constants import SUPPORTED_FAKE_QUANT_BITS
from ..core.decisions import SelectionDecision
from ..core.registry import OperatorExecutionContext, ProcessingOperator
from ..storage import LogicalQuantizedHostArchive


def _attention_scores(q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
    q_h = q.transpose(0, 1).float()
    k_h = k.transpose(0, 1).float()
    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    return scores


def _safe_group_size(length: int, group_size: int) -> int:
    if length <= 0:
        return 1
    return max(1, min(int(group_size), int(length)))


def asymmetric_fake_quant(data: torch.Tensor, *, bits: int, dim: int, group_size: int) -> torch.Tensor:
    if data.numel() == 0 or bits >= 16:
        return data
    if bits not in SUPPORTED_FAKE_QUANT_BITS:
        raise ValueError(f"Unsupported fake quant bits: {bits}")
    work = data.float()
    dim = dim if dim >= 0 else work.dim() + dim
    length = int(work.shape[dim])
    group = _safe_group_size(length, group_size)
    chunks = []
    max_int = float((1 << bits) - 1)
    eps = torch.finfo(torch.float32).eps
    for start in range(0, length, group):
        stop = min(length, start + group)
        slc = [slice(None)] * work.dim()
        slc[dim] = slice(start, stop)
        chunk = work[tuple(slc)]
        mn = chunk.amin(dim=dim, keepdim=True)
        mx = chunk.amax(dim=dim, keepdim=True)
        scale = (mx - mn).clamp_min(eps) / max_int
        code = torch.round((chunk - mn) / scale).clamp_(0, max_int)
        chunks.append(code * scale + mn)
    return torch.cat(chunks, dim=dim).to(dtype=data.dtype)


def kivi_fake_quant_key(k_segment: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    # BAGEL uses [tokens, heads, head_dim]. KIVI groups keys along tokens,
    # retaining a scale/zero pair per head and channel.
    return asymmetric_fake_quant(k_segment, bits=bits, dim=0, group_size=group_size)


def kivi_fake_quant_value(v_segment: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    # KIVI groups values along head_dim, retaining a scale/zero pair per token
    # and head.
    return asymmetric_fake_quant(v_segment, bits=bits, dim=-1, group_size=group_size)


class ProtectOperator(ProcessingOperator):
    name = "protect"
    family = "protection"
    stages = {"cache_update", "pre_attention", "attention"}
    executable = True
    preserves_attention_dispatch = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        requires=frozenset({"segment_mask", "protected_mask"}),
        mutation_scope="protection_mask",
    )

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        ctx.protected_mask |= ctx.segment_mask
        ctx.record({"protected_tokens": int(ctx.segment_mask.sum().item())})
        return ctx


class IdentityOperator(ProcessingOperator):
    name = "identity"
    family = "identity"
    stages = {"cache_update", "pre_attention", "attention", "post_attention"}
    executable = True
    preserves_attention_dispatch = True
    capabilities = CapabilityDescriptor(stages=frozenset(stages))

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        ctx.record({"identity_tokens": int(ctx.segment_mask.sum().item())})
        return ctx


class HeavyHitterProtectOperator(ProcessingOperator):
    name = "heavy_hitter_protect"
    family = "protection"
    stages = {"attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        requires=frozenset({"query", "key", "segment_mask", "protected_mask"}),
        produces_selection=True,
        stateful=True,
        mutation_scope="protection_mask",
    )

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        segment_idx = torch.nonzero(ctx.segment_mask, as_tuple=False).flatten()
        segment_len = int(segment_idx.numel())
        if segment_len == 0:
            return ctx

        recent_ratio = float(ctx.params.get("recent_ratio", 0.1))
        heavy_ratio = float(ctx.params.get("heavy_ratio", 0.2))
        min_recent = int(ctx.params.get("min_recent", 4))
        min_heavy = int(ctx.params.get("min_heavy", 4))
        recent_k = min(segment_len, max(min_recent, int(math.ceil(segment_len * recent_ratio))))
        heavy_k = max(min_heavy, int(math.ceil(max(0, segment_len - recent_k) * heavy_ratio)))

        scores = _attention_scores(ctx.q, ctx.k, ctx.attn_mask)
        current_hh = torch.softmax(scores, dim=-1).sum(dim=1).detach()
        state_key = (
            ctx.runtime.run_id,
            ctx.runtime.phase,
            ctx.runtime.branch,
            ctx.runtime.cfg_branch,
            int(ctx.runtime.layer_idx),
            int(ctx.runtime.batch_idx),
            ctx.step.segment_id,
            ctx.step.id,
        )
        previous = ctx.state.get(state_key)
        if previous is None or previous.shape[0] != current_hh.shape[0] or previous.shape[-1] > current_hh.shape[-1]:
            cumulative = current_hh
        else:
            aligned = torch.zeros_like(current_hh)
            aligned[:, : previous.shape[-1]] = previous.to(device=current_hh.device, dtype=current_hh.dtype)
            cumulative = aligned + current_hh
        ctx.state[state_key] = cumulative.detach()

        if segment_len <= recent_k + heavy_k:
            decision_mask = ctx.segment_mask.clone()
            ctx.decision = SelectionDecision(
                token_count=int(ctx.k.shape[0]),
                protected_mask=decision_mask,
                scores=cumulative.mean(dim=0),
                budget=segment_len,
                reason="heavy_hitter_plus_recent",
                metadata={"recent_tokens": recent_k, "heavy_tokens": max(0, segment_len - recent_k)},
            )
            ctx.protected_mask = ctx.decision.apply_protection(ctx.protected_mask)
            ctx.record(
                {
                    "hh_segment_tokens": segment_len,
                    "hh_recent_tokens": recent_k,
                    "hh_heavy_tokens": max(0, segment_len - recent_k),
                    "hh_protected_tokens": segment_len,
                }
            )
            return ctx

        decision_mask = torch.zeros_like(ctx.protected_mask)
        recent_idx = segment_idx[-recent_k:]
        decision_mask[recent_idx] = True
        candidate_idx = segment_idx[:-recent_k]
        heavy_k = min(int(candidate_idx.numel()), heavy_k)
        if heavy_k > 0:
            token_scores = cumulative[:, candidate_idx].mean(dim=0)
            top_idx = torch.topk(token_scores, k=heavy_k, largest=True).indices
            decision_mask[candidate_idx[top_idx]] = True
        ctx.decision = SelectionDecision(
            token_count=int(ctx.k.shape[0]),
            protected_mask=decision_mask,
            scores=cumulative.mean(dim=0),
            budget=int(decision_mask.sum().item()),
            reason="heavy_hitter_plus_recent",
            metadata={"recent_tokens": recent_k, "heavy_tokens": heavy_k},
        )
        ctx.protected_mask = ctx.decision.apply_protection(ctx.protected_mask)
        ctx.record(
            {
                "hh_segment_tokens": segment_len,
                "hh_recent_tokens": recent_k,
                "hh_heavy_tokens": heavy_k,
                "hh_protected_tokens": int((ctx.protected_mask & ctx.segment_mask).sum().item()),
            }
        )
        return ctx


class TopKEvictionOperator(ProcessingOperator):
    name = "topk_eviction"
    family = "eviction"
    stages = {"attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        layouts=frozenset({"attention_view", "dense"}),
        devices=frozenset({"cpu", "cuda", "mps"}),
        requires=frozenset({"query", "key", "segment_mask", "protected_mask", "cache_storage_view"}),
        produces_selection=True,
        mutation_scope="persistent_cache",
        physical_storage_mutation=True,
    )

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        segment_idx = torch.nonzero(ctx.segment_mask, as_tuple=False).flatten()
        segment_len = int(segment_idx.numel())
        if segment_len == 0:
            return ctx

        configured_keep = ctx.params.get("keep_k", ctx.params.get("budget"))
        if configured_keep is not None and int(configured_keep) > 0:
            target_keep = int(configured_keep)
            original_segment_len = segment_len
        else:
            keep_ratio = max(0.0, min(1.0, float(ctx.params.get("keep_ratio", 1.0))))
            origin_key = (
                "topk_origin",
                ctx.runtime.run_id,
                ctx.runtime.phase,
                ctx.runtime.branch,
                ctx.runtime.cfg_branch,
                int(ctx.runtime.layer_idx),
                int(ctx.runtime.batch_idx),
                ctx.step.segment_id,
                ctx.step.id,
            )
            original_segment_len = max(segment_len, int(ctx.state.get(origin_key, 0)))
            ctx.state[origin_key] = original_segment_len
            target_keep = int(math.ceil(original_segment_len * keep_ratio))
        target_keep = min(segment_len, max(int(ctx.params.get("min_keep", 1)), target_keep))

        protected_segment = ctx.protected_mask & ctx.segment_mask
        protected_idx = torch.nonzero(protected_segment, as_tuple=False).flatten()
        candidate_idx = torch.nonzero(ctx.segment_mask & ~ctx.protected_mask, as_tuple=False).flatten()
        candidate_keep = min(int(candidate_idx.numel()), max(0, target_keep - int(protected_idx.numel())))

        scores = _attention_scores(ctx.q, ctx.k, ctx.attn_mask)
        token_scores = torch.softmax(scores, dim=-1).mean(dim=(0, 1)).detach()
        selected = protected_segment.clone()
        if candidate_keep > 0:
            top = torch.topk(token_scores[candidate_idx], k=candidate_keep, largest=True).indices
            selected[candidate_idx[top]] = True

        keep_mask = torch.ones(int(ctx.k.shape[0]), device=ctx.k.device, dtype=torch.bool)
        keep_mask[ctx.segment_mask] = False
        keep_mask[selected] = True
        selected_count = int((keep_mask & ctx.segment_mask).sum().item())
        ctx.decision = SelectionDecision(
            token_count=int(ctx.k.shape[0]),
            protected_mask=protected_segment,
            keep_mask=keep_mask,
            scores=token_scores,
            budget=selected_count,
            reason="attention_topk",
            metadata={
                "segment_tokens": segment_len,
                "original_segment_tokens": original_segment_len,
                "selected_tokens": selected_count,
                "evicted_tokens": segment_len - selected_count,
                "protected_tokens": int(protected_idx.numel()),
            },
        )
        ctx.record(
            {
                "topk_segment_tokens": segment_len,
                "topk_original_segment_tokens": original_segment_len,
                "topk_selected_tokens": selected_count,
                "topk_evicted_tokens": segment_len - selected_count,
            }
        )
        return ctx


class KIVIQuantizationOperator(ProcessingOperator):
    family = "quantization"
    stages = {"pre_attention", "attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        layouts=frozenset({"attention_view", "dense"}),
        devices=frozenset({"cpu", "cuda", "mps"}),
        requires=frozenset({"key", "value", "segment_mask", "cache_storage_view"}),
        mutation_scope="attention_view",
        physical_storage_mutation=False,
        accepts_quantized_input=False,
    )

    def __init__(self, name: str = "kivi_quantization") -> None:
        self.name = name

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        bits = int(out.get("bits", 4))
        for name, value in (("bits", bits), ("k_bits", out.get("k_bits", bits)), ("v_bits", out.get("v_bits", bits))):
            if int(value) not in SUPPORTED_FAKE_QUANT_BITS:
                raise ValueError(
                    f"{self.name} {name}={value} is unsupported; expected one of {sorted(SUPPORTED_FAKE_QUANT_BITS)}"
                )
        working_set_mode = str(out.get("working_set_mode", "attention_view"))
        if working_set_mode not in {"attention_view", "host_backed"}:
            raise ValueError(
                f"{self.name} working_set_mode must be attention_view or host_backed"
            )
        out["working_set_mode"] = working_set_mode
        return out

    @staticmethod
    def _state_key(ctx: OperatorExecutionContext) -> tuple[object, ...]:
        return (
            "kivi_host_archive",
            ctx.runtime.run_id,
            ctx.runtime.phase,
            ctx.runtime.branch,
            ctx.runtime.cfg_branch,
            int(ctx.runtime.layer_idx),
            int(ctx.runtime.batch_idx),
            ctx.step.segment_id,
            ctx.step.id,
        )

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        bits = int(ctx.params.get("bits", 4))
        k_bits = int(ctx.params.get("k_bits", bits))
        v_bits = int(ctx.params.get("v_bits", bits))
        group_size = int(ctx.params.get("group_size", 32))
        residual_length = int(ctx.params.get("residual_length", 32))
        preserve_protected = bool(ctx.params.get("preserve_protected", True))
        bypass_quantization = bool(ctx.params.get("bypass_quantization", False))

        quant_mask = ctx.segment_mask.clone()
        segment_idx = torch.nonzero(ctx.segment_mask, as_tuple=False).flatten()
        if residual_length > 0 and segment_idx.numel() > 0:
            quant_mask[segment_idx[-residual_length:]] = False
        if preserve_protected:
            quant_mask &= ~ctx.protected_mask
        if bypass_quantization:
            quant_mask.zero_()

        segment_tokens = int(segment_idx.numel())
        quant_tokens = int(quant_mask.sum().item())
        full_precision_tokens = segment_tokens - quant_tokens
        elem_bytes = max(1, ctx.k.element_size())
        attention_num_heads = int(ctx.k.shape[1])
        storage_num_kv_heads = int(
            ctx.storage.layout.num_kv_heads
            if ctx.storage is not None
            else (ctx.storage_num_kv_heads or attention_num_heads)
        )
        head_dim = int(ctx.k.shape[2])
        attention_elements_per_token = int(attention_num_heads * head_dim)
        storage_elements_per_token = int(storage_num_kv_heads * head_dim)
        original_storage = float(segment_tokens * storage_elements_per_token * 2 * elem_bytes)
        full_precision_bytes = float(full_precision_tokens * storage_elements_per_token * 2 * elem_bytes)

        key_groups = math.ceil(quant_tokens / max(1, group_size)) if quant_tokens else 0
        value_groups = math.ceil(head_dim / max(1, group_size)) if quant_tokens else 0
        # Fake quant uses float32 min/scale metadata. This is an ideal packed
        # estimate, not realized allocation in this prototype.
        metadata_bytes = float(
            (key_groups * storage_num_kv_heads * head_dim + quant_tokens * storage_num_kv_heads * value_groups)
            * 2
            * 4
        )
        packed_payload_bytes = float((quant_tokens * storage_elements_per_token * (k_bits + v_bits)) / 8.0)
        ideal_storage = full_precision_bytes + packed_payload_bytes + metadata_bytes
        working_set_mode = str(
            ctx.params.get("working_set_mode", "attention_view")
        )
        descriptor = (
            k_bits,
            v_bits,
            group_size,
            residual_length,
            quant_tokens,
            full_precision_tokens,
            quant_mask.detach().to(device="cpu", dtype=torch.uint8).numpy().tobytes(),
        )
        state = None
        replica_cache_hit = False
        if working_set_mode == "host_backed":
            state_key = self._state_key(ctx)
            state = ctx.state.get(state_key)
            segment_identity = ctx.segment_mask.detach().to(
                device="cpu", dtype=torch.bool
            )
            if state is None:
                state = {
                    "quantized_host_archive": LogicalQuantizedHostArchive(
                        full_archive_bytes=max(1, int(round(original_storage)))
                    ),
                    "segment_mask": segment_identity,
                }
                ctx.state[state_key] = state
            elif not torch.equal(state["segment_mask"], segment_identity):
                raise ValueError(
                    "KIVI Host archive segment identity changed after initialization"
                )
            tracker = state["quantized_host_archive"]
            if int(tracker.full_archive_bytes) != int(round(original_storage)):
                raise ValueError(
                    "KIVI Host archive shape changed after initialization"
                )
            replica_cache_hit = state.get("replica_descriptor") == descriptor

        if replica_cache_hit:
            k_out = ctx.k.clone()
            v_out = ctx.v.clone()
            k_out[segment_idx] = state["replica_k"]
            v_out[segment_idx] = state["replica_v"]
            ctx.k = k_out
            ctx.v = v_out
        else:
            if quant_tokens > 0:
                k_out = ctx.k.clone()
                v_out = ctx.v.clone()
                k_out[quant_mask] = kivi_fake_quant_key(
                    k_out[quant_mask], bits=k_bits, group_size=group_size
                )
                v_out[quant_mask] = kivi_fake_quant_value(
                    v_out[quant_mask], bits=v_bits, group_size=group_size
                )
                ctx.k = k_out
                ctx.v = v_out
            if state is not None:
                state["replica_descriptor"] = descriptor
                state["replica_k"] = ctx.k[segment_idx].detach().clone()
                state["replica_v"] = ctx.v[segment_idx].detach().clone()

        record_values = {
            "quantized_tokens": quant_tokens,
            "protected_tokens_preserved": int(
                (ctx.protected_mask & ctx.segment_mask).sum().item()
            )
            if preserve_protected
            else 0,
            "processed_input_kv_bytes": float(
                0
                if replica_cache_hit
                else quant_tokens * attention_elements_per_token * 2 * elem_bytes
            ),
            "quantized_replica_cache_hit": float(replica_cache_hit),
        }
        snapshot = {
                "segment_tokens": segment_tokens,
                "quantized_tokens": quant_tokens,
                "full_precision_tokens": full_precision_tokens,
                "attention_num_heads": attention_num_heads,
                "storage_num_kv_heads": storage_num_kv_heads,
                "gqa_repeat": float(attention_num_heads / max(storage_num_kv_heads, 1)),
                "original_storage_bytes": original_storage,
                "ideal_packed_payload_bytes": packed_payload_bytes,
                "ideal_metadata_bytes": metadata_bytes,
                "ideal_total_storage_bytes": ideal_storage,
                "logical_host_backed": float(
                    working_set_mode == "host_backed"
                ),
        }
        if working_set_mode == "host_backed":
            tracker = state["quantized_host_archive"]
            transition = tracker.materialize(
                replica_bytes=int(round(ideal_storage)),
                descriptor=descriptor,
            )
            record_values.update(
                {
                    "logical_host_h2d_bytes": transition.h2d_bytes,
                    "logical_host_released_bytes": transition.released_bytes,
                    "logical_gpu_resident_bytes": transition.resident_bytes,
                }
            )
            snapshot.update(
                {
                    "logical_full_host_archive_bytes": original_storage,
                    "logical_gpu_resident_bytes": float(
                        transition.resident_bytes
                    ),
                    "logical_host_h2d_bytes": float(transition.h2d_bytes),
                    "logical_host_released_bytes": float(
                        transition.released_bytes
                    ),
                }
            )
        ctx.record(record_values)
        ctx.set_storage_snapshot(snapshot)
        return ctx


class AttentionStatsOperator(ProcessingOperator):
    name = "attention_stats"
    family = "observation"
    stages = {"post_attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        requires=frozenset({"segment_mask"}),
    )

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        ctx.record({"observed_tokens": int(ctx.segment_mask.sum().item())})
        return ctx


class BlockAttentionMetricsOperator(ProcessingOperator):
    """Collect chunked EMA attention mass and K90 for one cache segment."""

    name = "block_attention_metrics"
    family = "observation"
    stages = {"post_attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        requires=frozenset({"segment_mask", "attention_probs"}),
        stateful=True,
    )

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        if "ema_window_size" in out:
            window = int(out["ema_window_size"])
            if window <= 0:
                raise ValueError("block_attention_metrics ema_window_size must be positive")
            # This convention gives an effective history of roughly ``window``
            # observations and makes the requested window directly auditable.
            out["ema_decay"] = 1.0 - 1.0 / float(window)
        decay = float(out.get("ema_decay", 0.8))
        if not 0.0 <= decay < 1.0:
            raise ValueError("block_attention_metrics ema_decay must be in [0, 1)")
        out["carry_ema_across_chunks"] = bool(out.get("carry_ema_across_chunks", False))
        for name in ("step_chunk_size", "layer_chunk_size"):
            if int(out.get(name, 1)) <= 0:
                raise ValueError(f"block_attention_metrics {name} must be positive")
        estimator = str(out.get("mass_estimator", "active_only"))
        if estimator not in {"active_only", "block_centroid"}:
            raise ValueError(
                "block_attention_metrics mass_estimator must be active_only or block_centroid"
            )
        block_size = int(out.get("mass_estimator_block_size", 8))
        representatives = int(out.get("mass_estimator_representatives", 4))
        calibration = float(out.get("mass_estimator_calibration_factor", 1.0))
        if block_size <= 0:
            raise ValueError("mass_estimator_block_size must be positive")
        if representatives <= 0 or representatives > block_size:
            raise ValueError(
                "mass_estimator_representatives must be in [1, mass_estimator_block_size]"
            )
        if calibration <= 0.0:
            raise ValueError("mass_estimator_calibration_factor must be positive")
        out.update(
            {
                "mass_estimator": estimator,
                "mass_estimator_block_size": block_size,
                "mass_estimator_representatives": representatives,
                "mass_estimator_calibration_factor": calibration,
            }
        )
        return out

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        from ..metrics import (
            DEFAULT_LAYER_CHUNK_SIZE,
            DEFAULT_STEP_CHUNK_SIZE,
            attention_mass_and_k90,
            block_summary_metadata_key,
            block_summary_corrected_attention_mass,
            block_metric_key,
            block_metric_stream_key,
            build_block_summary_metadata,
            chunk_indices,
        )

        if ctx.attention_probs is None:
            raise ValueError("block_attention_metrics requires final attention probabilities")
        if "ema_window_size" in ctx.params:
            decay = 1.0 - 1.0 / float(int(ctx.params["ema_window_size"]))
        else:
            decay = float(ctx.params.get("ema_decay", 0.8))
        layer_chunk, step_chunk = chunk_indices(
            layer_idx=ctx.runtime.layer_idx,
            step_index=ctx.runtime.step_index,
            decode_index=ctx.runtime.decode_index,
            layer_chunk_size=int(ctx.params.get("layer_chunk_size", DEFAULT_LAYER_CHUNK_SIZE)),
            step_chunk_size=int(ctx.params.get("step_chunk_size", DEFAULT_STEP_CHUNK_SIZE)),
            first_layer_separate=bool(ctx.params.get("first_layer_separate", False)),
            first_step_separate=bool(ctx.params.get("first_step_separate", False)),
        )
        active_mass, k90 = attention_mass_and_k90(
            ctx.attention_probs, ctx.segment_mask
        )
        mass = active_mass
        estimator_stats = {
            "active_attention_mass": active_mass,
            "corrected_attention_mass": active_mass,
            "estimated_offloaded_partition_ratio": 0.0,
            "offloaded_head_query_tokens": 0.0,
        }
        estimator = str(ctx.params.get("mass_estimator", "active_only"))
        if estimator == "block_centroid":
            active_keep_mask = ctx.runtime.kv_metadata.get(
                "effective_attention_keep_mask"
            )
            if active_keep_mask is None:
                raise ValueError(
                    "block_centroid mass estimator requires effective_attention_keep_mask"
                )
            if ctx.metadata_state is None:
                raise ValueError(
                    "block_centroid mass estimator requires a metadata state store"
                )
            metadata_key = block_summary_metadata_key(
                run_id=ctx.runtime.run_id,
                phase=ctx.runtime.phase,
                branch=ctx.runtime.branch,
                cfg_branch=ctx.runtime.cfg_branch,
                batch_idx=ctx.runtime.batch_idx,
                layer_idx=ctx.runtime.layer_idx,
                segment_id=ctx.step.segment_id,
                operator_id=ctx.step.id,
            )
            metadata_entry = ctx.metadata_state.get(metadata_key)
            if metadata_entry is None:
                metadata_entry = {
                    "metadata": build_block_summary_metadata(
                        k=ctx.k,
                        segment_mask=ctx.segment_mask,
                        active_keep_mask=active_keep_mask,
                        block_size=int(ctx.params["mass_estimator_block_size"]),
                        representatives=int(
                            ctx.params["mass_estimator_representatives"]
                        ),
                    ),
                    "builds": 1,
                }
                ctx.metadata_state[metadata_key] = metadata_entry
            mass, estimator_stats = block_summary_corrected_attention_mass(
                q=ctx.q,
                k=ctx.k,
                attention_probs=ctx.attention_probs,
                segment_mask=ctx.segment_mask,
                active_keep_mask=active_keep_mask,
                attn_mask=ctx.attn_mask,
                block_size=int(ctx.params["mass_estimator_block_size"]),
                representatives=int(ctx.params["mass_estimator_representatives"]),
                calibration_factor=float(
                    ctx.params["mass_estimator_calibration_factor"]
                ),
                summary_metadata=metadata_entry["metadata"],
            )
            estimator_stats["summary_metadata_builds"] = int(
                metadata_entry["builds"]
            )
        key = block_metric_key(
            run_id=ctx.runtime.run_id,
            phase=ctx.runtime.phase,
            branch=ctx.runtime.branch,
            cfg_branch=ctx.runtime.cfg_branch,
            batch_idx=ctx.runtime.batch_idx,
            segment_id=ctx.step.segment_id,
            layer_chunk=layer_chunk,
            step_chunk=step_chunk,
        )
        stream_key = block_metric_stream_key(
            run_id=ctx.runtime.run_id,
            phase=ctx.runtime.phase,
            branch=ctx.runtime.branch,
            cfg_branch=ctx.runtime.cfg_branch,
            batch_idx=ctx.runtime.batch_idx,
            segment_id=ctx.step.segment_id,
            layer_chunk=layer_chunk,
        )
        carry_across_chunks = bool(ctx.params.get("carry_ema_across_chunks", False))
        previous = ctx.state.get(stream_key) if carry_across_chunks else ctx.state.get(key)
        if previous is None:
            ema_mass = mass
            ema_active_mass = active_mass
            ema_k90 = k90
            stream_calls = 1
        else:
            ema_mass = decay * float(previous["ema_attention_mass"]) + (1.0 - decay) * mass
            ema_active_mass = decay * float(
                previous.get("ema_active_attention_mass", previous["ema_attention_mass"])
            ) + (1.0 - decay) * active_mass
            ema_k90 = decay * float(previous["ema_k90"]) + (1.0 - decay) * k90
            stream_calls = int(previous.get("stream_calls", previous.get("calls", 0))) + 1
        chunk_previous = ctx.state.get(key)
        chunk_calls = int(chunk_previous.get("calls", 0)) + 1 if chunk_previous else 1
        value = {
            "ema_attention_mass": ema_mass,
            "ema_active_attention_mass": ema_active_mass,
            "ema_k90": ema_k90,
            "latest_attention_mass": mass,
            "latest_active_attention_mass": active_mass,
            "latest_k90": k90,
            "calls": chunk_calls,
            "stream_calls": stream_calls,
            "segment_tokens": int(ctx.segment_mask.sum().item()),
            "layer_chunk": layer_chunk,
            "step_chunk": step_chunk,
            "ema_decay": decay,
            "ema_window_size": ctx.params.get("ema_window_size"),
            "carry_ema_across_chunks": carry_across_chunks,
            "mass_estimator": estimator,
            "mass_estimator_block_size": int(
                ctx.params.get("mass_estimator_block_size", 8)
            ),
            "mass_estimator_representatives": int(
                ctx.params.get("mass_estimator_representatives", 4)
            ),
            "mass_estimator_calibration_factor": float(
                ctx.params.get("mass_estimator_calibration_factor", 1.0)
            ),
            **estimator_stats,
        }
        ctx.state[key] = value
        if carry_across_chunks:
            ctx.state[stream_key] = dict(value)
        ctx.record(
            {
                "attention_mass": mass,
                "active_attention_mass": active_mass,
                "attention_mass_correction": mass - active_mass,
                "estimated_offloaded_partition_ratio": estimator_stats[
                    "estimated_offloaded_partition_ratio"
                ],
                "k90": k90,
                "ema_attention_mass": ema_mass,
                "ema_active_attention_mass": ema_active_mass,
                "ema_k90": ema_k90,
            }
        )
        return ctx


class ParallelBlockAttentionMetricsOperator(ProcessingOperator):
    """Correct multiple offloaded type masses with one shared denominator."""

    name = "parallel_block_attention_metrics"
    family = "observation"
    stages = {"post_attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        requires=frozenset(
            {"query", "key", "key_type_ids", "segment_mask", "attention_probs"}
        ),
        stateful=True,
    )

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        specs = dict(out.get("segment_estimators", {}) or {})
        if not specs:
            raise ValueError(
                "parallel_block_attention_metrics requires segment_estimators"
            )
        normalized = {}
        for segment_id, raw in specs.items():
            spec = dict(raw)
            kv_type = str(spec.get("kv_type", "")).strip()
            if not kv_type:
                raise ValueError(f"segment estimator {segment_id} requires kv_type")
            estimator = str(spec.get("estimator", "block_centroid"))
            if estimator not in {"active_only", "block_centroid"}:
                raise ValueError(
                    f"segment estimator {segment_id} estimator must be "
                    "active_only or block_centroid"
                )
            block_size = int(spec.get("block_size", 8))
            representatives = int(spec.get("representatives", 4))
            calibration = float(spec.get("calibration_factor", 1.0))
            if block_size <= 0:
                raise ValueError(f"segment estimator {segment_id} block_size must be positive")
            if representatives <= 0 or representatives > block_size:
                raise ValueError(
                    f"segment estimator {segment_id} representatives must be in [1, block_size]"
                )
            if calibration <= 0.0:
                raise ValueError(
                    f"segment estimator {segment_id} calibration_factor must be positive"
                )
            normalized[str(segment_id)] = {
                "kv_type": kv_type,
                "estimator": estimator,
                "block_size": block_size,
                "representatives": representatives,
                "calibration_factor": calibration,
            }
        if "ema_window_size" in out:
            window = int(out["ema_window_size"])
            if window <= 0:
                raise ValueError(
                    "parallel_block_attention_metrics ema_window_size must be positive"
                )
            out["ema_decay"] = 1.0 - 1.0 / float(window)
        decay = float(out.get("ema_decay", 0.8))
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                "parallel_block_attention_metrics ema_decay must be in [0, 1)"
            )
        for name in ("step_chunk_size", "layer_chunk_size"):
            if int(out.get(name, 1)) <= 0:
                raise ValueError(
                    f"parallel_block_attention_metrics {name} must be positive"
                )
        out["segment_estimators"] = normalized
        out["carry_ema_across_chunks"] = bool(
            out.get("carry_ema_across_chunks", False)
        )
        out["record_oracle_mass"] = bool(out.get("record_oracle_mass", False))
        oracle_source = str(
            out.get("oracle_score_source", "pre_eviction_attention")
        )
        if oracle_source != "pre_eviction_attention":
            raise ValueError(
                "parallel_block_attention_metrics oracle_score_source must be "
                "pre_eviction_attention"
            )
        out["oracle_score_source"] = oracle_source
        return out

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        from ..metrics import (
            DEFAULT_LAYER_CHUNK_SIZE,
            DEFAULT_STEP_CHUNK_SIZE,
            attention_mass_and_k90,
            block_summary_metadata_key,
            block_metric_key,
            block_metric_stream_key,
            block_summary_partition_components,
            build_block_summary_metadata,
            chunk_indices,
        )

        if ctx.attention_probs is None or ctx.key_type_ids is None:
            raise ValueError(
                "parallel_block_attention_metrics requires attention probabilities and key types"
            )
        active_keep_mask = ctx.runtime.kv_metadata.get(
            "effective_attention_keep_mask"
        )
        if active_keep_mask is None:
            raise ValueError(
                "parallel_block_attention_metrics requires effective_attention_keep_mask"
            )
        if ctx.metadata_state is None:
            raise ValueError(
                "parallel_block_attention_metrics requires a metadata state store"
            )
        layer_chunk, step_chunk = chunk_indices(
            layer_idx=ctx.runtime.layer_idx,
            step_index=ctx.runtime.step_index,
            decode_index=ctx.runtime.decode_index,
            layer_chunk_size=int(
                ctx.params.get("layer_chunk_size", DEFAULT_LAYER_CHUNK_SIZE)
            ),
            step_chunk_size=int(
                ctx.params.get("step_chunk_size", DEFAULT_STEP_CHUNK_SIZE)
            ),
            first_layer_separate=bool(ctx.params.get("first_layer_separate", False)),
            first_step_separate=bool(ctx.params.get("first_step_separate", False)),
        )
        decay = float(ctx.params.get("ema_decay", 0.8))
        carry = bool(ctx.params.get("carry_ema_across_chunks", False))
        oracle_probs = None
        oracle_source = str(
            ctx.params.get("oracle_score_source", "pre_eviction_attention")
        )
        if bool(ctx.params.get("record_oracle_mass", False)):
            sources = dict(ctx.attention_prob_sources or {})
            oracle_probs = sources.get(oracle_source)
            if oracle_probs is None:
                raise ValueError(
                    "parallel_block_attention_metrics requested oracle mass from "
                    f"{oracle_source!r}, but that attention source is unavailable"
                )
            if tuple(oracle_probs.shape) != tuple(ctx.attention_probs.shape):
                raise ValueError(
                    "oracle attention probabilities must match effective attention shape"
                )
        compressed_oracle_call = bool(
            (
                ctx.segment_mask.to(device=active_keep_mask.device)[None, :]
                & ~active_keep_mask.to(dtype=torch.bool)
            ).any()
        )
        components = {}
        total_partition_ratio = None
        kv_type_name_to_id = dict(
            ctx.runtime.kv_metadata.get("kv_type_name_to_id", {}) or {}
        )
        for segment_id, spec in ctx.params["segment_estimators"].items():
            kv_type = str(spec["kv_type"])
            if kv_type not in kv_type_name_to_id:
                raise ValueError(
                    f"parallel block estimator cannot resolve kv_type {kv_type!r}"
                )
            mask = ctx.segment_mask & (
                ctx.key_type_ids.to(device=ctx.k.device)
                == int(kv_type_name_to_id[kv_type])
            )
            if not bool(mask.any()):
                continue
            estimator = str(spec.get("estimator", "block_centroid"))
            if estimator == "active_only":
                active_mass = ctx.attention_probs.float()[..., mask].sum(dim=-1)
                partition_ratio = torch.zeros_like(active_mass)
                stats = {
                    "active_attention_mass": float(active_mass.mean().item()),
                    "corrected_attention_mass": float(active_mass.mean().item()),
                    "estimated_offloaded_partition_ratio": 0.0,
                    "offloaded_head_query_tokens": 0.0,
                    "summary_metadata_source": "not_required",
                    "summary_metadata_bytes": 0.0,
                    "summary_metadata_builds": 0,
                }
            else:
                metadata_key = block_summary_metadata_key(
                    run_id=ctx.runtime.run_id,
                    phase=ctx.runtime.phase,
                    branch=ctx.runtime.branch,
                    cfg_branch=ctx.runtime.cfg_branch,
                    batch_idx=ctx.runtime.batch_idx,
                    layer_idx=ctx.runtime.layer_idx,
                    segment_id=segment_id,
                    operator_id=ctx.step.id,
                )
                metadata_entry = ctx.metadata_state.get(metadata_key)
                if metadata_entry is None:
                    metadata_entry = {
                        "metadata": build_block_summary_metadata(
                            k=ctx.k,
                            segment_mask=mask,
                            active_keep_mask=active_keep_mask,
                            block_size=int(spec["block_size"]),
                            representatives=int(spec["representatives"]),
                        ),
                        "builds": 1,
                    }
                    ctx.metadata_state[metadata_key] = metadata_entry
                active_mass, partition_ratio, stats = block_summary_partition_components(
                    q=ctx.q,
                    k=ctx.k,
                    attention_probs=ctx.attention_probs,
                    segment_mask=mask,
                    active_keep_mask=active_keep_mask,
                    attn_mask=ctx.attn_mask,
                    block_size=int(spec["block_size"]),
                    representatives=int(spec["representatives"]),
                    calibration_factor=float(spec["calibration_factor"]),
                    summary_metadata=metadata_entry["metadata"],
                )
                stats["summary_metadata_builds"] = int(metadata_entry["builds"])
            _, k90 = attention_mass_and_k90(ctx.attention_probs, mask)
            oracle_mass = (
                oracle_probs.float()[..., mask].sum(dim=-1)
                if oracle_probs is not None
                else None
            )
            components[segment_id] = (
                mask,
                active_mass,
                partition_ratio,
                k90,
                stats,
                oracle_mass,
            )
            total_partition_ratio = (
                partition_ratio
                if total_partition_ratio is None
                else total_partition_ratio + partition_ratio
            )

        if not components:
            return ctx
        denominator = 1.0 + total_partition_ratio
        record_values = {
            "parallel_estimated_total_offloaded_partition_ratio": float(
                total_partition_ratio.mean().item()
            )
        }
        for segment_id, (
            mask,
            active_mass,
            partition_ratio,
            k90,
            stats,
            oracle_mass,
        ) in components.items():
            corrected = (active_mass + partition_ratio) / denominator
            mass = float(corrected.mean().item())
            active = float(active_mass.mean().item())
            oracle = (
                float(oracle_mass.mean().item())
                if oracle_mass is not None
                else None
            )
            key = block_metric_key(
                run_id=ctx.runtime.run_id,
                phase=ctx.runtime.phase,
                branch=ctx.runtime.branch,
                cfg_branch=ctx.runtime.cfg_branch,
                batch_idx=ctx.runtime.batch_idx,
                segment_id=segment_id,
                layer_chunk=layer_chunk,
                step_chunk=step_chunk,
            )
            stream_key = block_metric_stream_key(
                run_id=ctx.runtime.run_id,
                phase=ctx.runtime.phase,
                branch=ctx.runtime.branch,
                cfg_branch=ctx.runtime.cfg_branch,
                batch_idx=ctx.runtime.batch_idx,
                segment_id=segment_id,
                layer_chunk=layer_chunk,
            )
            previous = ctx.state.get(stream_key) if carry else ctx.state.get(key)
            if previous is None:
                ema_mass = mass
                ema_active = active
                ema_oracle = oracle
                ema_k90 = k90
                stream_calls = 1
            else:
                ema_mass = decay * float(previous["ema_attention_mass"]) + (1.0 - decay) * mass
                ema_active = decay * float(
                    previous.get("ema_active_attention_mass", previous["ema_attention_mass"])
                ) + (1.0 - decay) * active
                ema_oracle = (
                    decay
                    * float(
                        previous.get(
                            "ema_oracle_attention_mass",
                            previous.get("latest_oracle_attention_mass", oracle),
                        )
                    )
                    + (1.0 - decay) * float(oracle)
                    if oracle is not None
                    else None
                )
                ema_k90 = decay * float(previous["ema_k90"]) + (1.0 - decay) * k90
                stream_calls = int(
                    previous.get("stream_calls", previous.get("calls", 0))
                ) + 1
            chunk_previous = ctx.state.get(key)
            chunk_calls = int(chunk_previous.get("calls", 0)) + 1 if chunk_previous else 1
            oracle_fields = {}
            if oracle is not None:
                active_error = abs(active - oracle)
                corrected_error = abs(mass - oracle)
                active_error_sum = active_error + float(
                    (chunk_previous or {}).get("active_abs_error_sum_vs_oracle", 0.0)
                )
                corrected_error_sum = corrected_error + float(
                    (chunk_previous or {}).get("corrected_abs_error_sum_vs_oracle", 0.0)
                )
                oracle_calls = int((chunk_previous or {}).get("oracle_calls", 0)) + 1
                correction_win_calls = int(
                    (chunk_previous or {}).get("correction_win_calls", 0)
                ) + int(corrected_error < active_error)
                compressed_oracle_calls = int(
                    (chunk_previous or {}).get("compressed_oracle_calls", 0)
                ) + int(compressed_oracle_call)
                compressed_active_error_sum = float(
                    (chunk_previous or {}).get(
                        "compressed_active_abs_error_sum_vs_oracle", 0.0
                    )
                ) + (active_error if compressed_oracle_call else 0.0)
                compressed_corrected_error_sum = float(
                    (chunk_previous or {}).get(
                        "compressed_corrected_abs_error_sum_vs_oracle", 0.0
                    )
                ) + (corrected_error if compressed_oracle_call else 0.0)
                compressed_win_calls = int(
                    (chunk_previous or {}).get("compressed_correction_win_calls", 0)
                ) + int(compressed_oracle_call and corrected_error < active_error)
                active_mae = active_error_sum / oracle_calls
                corrected_mae = corrected_error_sum / oracle_calls
                oracle_fields = {
                    "oracle_score_source": oracle_source,
                    "latest_oracle_attention_mass": oracle,
                    "ema_oracle_attention_mass": ema_oracle,
                    "latest_active_abs_error_vs_oracle": active_error,
                    "latest_corrected_abs_error_vs_oracle": corrected_error,
                    "active_abs_error_sum_vs_oracle": active_error_sum,
                    "corrected_abs_error_sum_vs_oracle": corrected_error_sum,
                    "active_mae_vs_oracle": active_mae,
                    "corrected_mae_vs_oracle": corrected_mae,
                    "correction_error_reduction_fraction": (
                        1.0 - corrected_mae / active_mae
                        if active_mae > 0.0
                        else 0.0
                    ),
                    "correction_win_calls": correction_win_calls,
                    "oracle_calls": oracle_calls,
                    "compressed_oracle_calls": compressed_oracle_calls,
                    "compressed_active_abs_error_sum_vs_oracle": (
                        compressed_active_error_sum
                    ),
                    "compressed_corrected_abs_error_sum_vs_oracle": (
                        compressed_corrected_error_sum
                    ),
                    "compressed_active_mae_vs_oracle": (
                        compressed_active_error_sum / compressed_oracle_calls
                        if compressed_oracle_calls
                        else 0.0
                    ),
                    "compressed_corrected_mae_vs_oracle": (
                        compressed_corrected_error_sum / compressed_oracle_calls
                        if compressed_oracle_calls
                        else 0.0
                    ),
                    "compressed_correction_win_calls": compressed_win_calls,
                }
            value = {
                "ema_attention_mass": ema_mass,
                "ema_active_attention_mass": ema_active,
                "ema_k90": ema_k90,
                "latest_attention_mass": mass,
                "latest_active_attention_mass": active,
                "latest_k90": k90,
                "calls": chunk_calls,
                "stream_calls": stream_calls,
                "segment_tokens": int(mask.sum().item()),
                "layer_chunk": layer_chunk,
                "step_chunk": step_chunk,
                "ema_decay": decay,
                "carry_ema_across_chunks": carry,
                "mass_estimator": str(
                    ctx.params["segment_estimators"][segment_id]["estimator"]
                ),
                "mass_estimator_block_size": int(
                    ctx.params["segment_estimators"][segment_id]["block_size"]
                ),
                "mass_estimator_representatives": int(
                    ctx.params["segment_estimators"][segment_id]["representatives"]
                ),
                "active_attention_mass": active,
                "corrected_attention_mass": mass,
                "estimated_offloaded_partition_ratio": float(
                    partition_ratio.mean().item()
                ),
                "estimated_total_offloaded_partition_ratio": float(
                    total_partition_ratio.mean().item()
                ),
                "offloaded_head_query_tokens": float(
                    stats["offloaded_head_query_tokens"]
                ),
                "summary_metadata_source": str(stats["summary_metadata_source"]),
                "summary_metadata_bytes": float(stats["summary_metadata_bytes"]),
                "summary_metadata_builds": int(stats["summary_metadata_builds"]),
                **oracle_fields,
            }
            ctx.state[key] = value
            if carry:
                ctx.state[stream_key] = dict(value)
            record_values.update(
                {
                    f"attention_mass_{segment_id}": mass,
                    f"active_attention_mass_{segment_id}": active,
                    f"attention_mass_correction_{segment_id}": mass - active,
                    f"ema_attention_mass_{segment_id}": ema_mass,
                }
            )
            if oracle is not None:
                record_values.update(
                    {
                        f"oracle_attention_mass_{segment_id}": oracle,
                        f"active_abs_error_vs_oracle_{segment_id}": active_error,
                        f"corrected_abs_error_vs_oracle_{segment_id}": corrected_error,
                    }
                )
        ctx.record(record_values)
        return ctx


class UnsupportedOperator(ProcessingOperator):
    def __init__(self, name: str, family: str, stages: set[str]) -> None:
        self.name = name
        self.family = family
        self.stages = stages
        self.executable = False
        self.capabilities = CapabilityDescriptor(stages=frozenset(stages))


def built_in_operators() -> list[ProcessingOperator]:
    from .classic import (
        PyramidKVAttentionMaskOperator,
        SnapKVAttentionMaskOperator,
        StreamingLLMEvictionOperator,
    )
    from .lifetime import AttentionMassLifetimeObserver, LifetimeRetirementOperator
    from .h2o import H2OAttentionMaskOperator, H2OSegmentAttentionMaskOperator
    from .physical import H2OPhysicalGQAOperator, KIVIPackedQuantizationOperator

    return [
        ProtectOperator(),
        IdentityOperator(),
        HeavyHitterProtectOperator(),
        H2OAttentionMaskOperator(),
        H2OSegmentAttentionMaskOperator(),
        H2OPhysicalGQAOperator(),
        TopKEvictionOperator(),
        StreamingLLMEvictionOperator(),
        SnapKVAttentionMaskOperator(),
        PyramidKVAttentionMaskOperator(),
        LifetimeRetirementOperator(),
        AttentionMassLifetimeObserver(),
        UnsupportedOperator("h2o_eviction", "eviction", {"attention"}),
        UnsupportedOperator("pyramid_budget_eviction", "eviction", {"attention"}),
        KIVIQuantizationOperator("fake_quant"),
        KIVIQuantizationOperator("kivi_quantization"),
        KIVIQuantizationOperator("bit_decay_quant"),
        KIVIPackedQuantizationOperator(),
        AttentionStatsOperator(),
        BlockAttentionMetricsOperator(),
        ParallelBlockAttentionMetricsOperator(),
    ]
