"""Executable physical-cache operators for the UniCache efficiency MVE."""

from __future__ import annotations

import math

import torch

from ..core.capabilities import CapabilityDescriptor
from ..core.decisions import PhysicalCacheRequest
from ..core.registry import OperatorExecutionContext, ProcessingOperator


class H2OPhysicalGQAOperator(ProcessingOperator):
    """H2O real-drop semantics adapted from MHA to BAGEL GQA storage."""

    name = "h2o_physical_gqa"
    family = "eviction"
    stages = {"attention"}
    executable = True
    uses_final_attention = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        layouts=frozenset({"attention_view", "dense"}),
        devices=frozenset({"cpu", "cuda"}),
        requires=frozenset(
            {"query", "key", "value", "segment_mask", "cache_storage_view"}
        ),
        produces_selection=True,
        stateful=True,
        mutation_scope="persistent_cache",
        physical_storage_mutation=True,
    )

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        ratio = float(out.get("target_budget_ratio", 0.2))
        if not 0.0 < ratio <= 1.0:
            raise ValueError("h2o_physical_gqa target_budget_ratio must be in (0, 1]")
        recent_fraction = float(out.get("recent_budget_fraction", 0.25))
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("h2o_physical_gqa recent_budget_fraction must be in [0, 1]")
        out["target_budget_ratio"] = ratio
        out["recent_budget_fraction"] = recent_fraction
        out["selection_frozen"] = bool(out.get("selection_frozen", False))
        return out

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        return ctx

    def update_after_attention(
        self,
        ctx: OperatorExecutionContext,
        attention_probs: torch.Tensor,
    ) -> OperatorExecutionContext:
        if ctx.storage_num_kv_heads is None:
            raise RuntimeError("Physical H2O requires native KV-head metadata")
        kv_heads = int(ctx.storage_num_kv_heads)
        query_heads = int(attention_probs.shape[0])
        if query_heads % kv_heads != 0:
            raise ValueError("BAGEL query heads must be divisible by KV heads")
        groups = query_heads // kv_heads
        per_kv_head_scores = attention_probs.float().sum(dim=1).reshape(
            kv_heads, groups, attention_probs.shape[-1]
        ).sum(dim=1)
        ctx.physical_request = PhysicalCacheRequest(
            operator=self.name,
            segment_id=ctx.step.segment_id,
            token_count=int(ctx.k.shape[0]),
            segment_mask=ctx.segment_mask.detach(),
            params=dict(ctx.params),
            per_kv_head_scores=per_kv_head_scores.detach(),
            metadata={
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "gqa_groups": groups,
                "observation": "full_kv_first_call",
                "token_type_id": ctx.runtime.kv_metadata.get(
                    "current_segment_type_id"
                ),
                "token_span": ctx.runtime.kv_metadata.get("current_segment_span"),
            },
        )
        segment_tokens = ctx.runtime.kv_metadata.get("current_segment_tokens")
        if segment_tokens is None:
            segment_tokens = int(ctx.segment_mask.sum().item())
        segment_tokens = int(segment_tokens)
        requested = ctx.params.get("budget", ctx.params.get("keep_k"))
        if requested is None:
            requested = max(
                1,
                int(round(segment_tokens * float(ctx.params["target_budget_ratio"]))),
            )
        ctx.record(
            {
                "physical_h2o_requests": 1,
                "physical_h2o_source_tokens": segment_tokens,
                "physical_h2o_target_tokens_per_kv_head": int(requested),
            }
        )
        return ctx


class KIVIPackedQuantizationOperator(ProcessingOperator):
    """Move a typed dense segment into KIVI packed physical storage."""

    name = "kivi_packed_quantization"
    family = "quantization"
    stages = {"pre_attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        layouts=frozenset({"attention_view", "dense"}),
        devices=frozenset({"cpu", "cuda"}),
        requires=frozenset({"key", "value", "segment_mask", "cache_storage_view"}),
        mutation_scope="persistent_cache",
        physical_storage_mutation=True,
        accepts_quantized_input=False,
    )

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        bits = int(out.get("bits", 4))
        k_bits = int(out.get("k_bits", bits))
        v_bits = int(out.get("v_bits", bits))
        if k_bits not in {2, 4, 8} or v_bits not in {2, 4, 8}:
            raise ValueError("kivi_packed_quantization supports 2, 4, or 8 bits")
        group_size = int(out.get("group_size", 64))
        if group_size <= 0:
            raise ValueError("kivi_packed_quantization group_size must be positive")
        backend = str(out.get("backend", "cuda"))
        if backend not in {"cuda", "reference"}:
            raise ValueError("kivi_packed_quantization backend must be cuda or reference")
        out.update(
            {
                "bits": bits,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "group_size": group_size,
                "residual_length": max(0, int(out.get("residual_length", 32))),
                "backend": backend,
            }
        )
        return out

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        if ctx.params["backend"] == "cuda" and ctx.k.device.type != "cuda":
            raise RuntimeError(
                "Physical KIVI backend=cuda cannot fall back to fake quantization or CPU"
            )
        segment_tokens = ctx.runtime.kv_metadata.get("current_segment_tokens")
        if segment_tokens is None:
            segment_tokens = int(ctx.segment_mask.sum().item())
        segment_tokens = int(segment_tokens)
        ctx.physical_request = PhysicalCacheRequest(
            operator=self.name,
            segment_id=ctx.step.segment_id,
            token_count=int(ctx.k.shape[0]),
            segment_mask=ctx.segment_mask.detach(),
            params=dict(ctx.params),
            metadata={
                "observation": "dense_prefill_cache",
                "token_type_id": ctx.runtime.kv_metadata.get(
                    "current_segment_type_id"
                ),
                "token_span": ctx.runtime.kv_metadata.get("current_segment_span"),
            },
        )
        quantized = max(0, segment_tokens - int(ctx.params["residual_length"]))
        ctx.record(
            {
                "physical_kivi_requests": 1,
                "physical_kivi_source_tokens": segment_tokens,
                "physical_kivi_quantized_tokens": quantized,
                "physical_kivi_bits": int(ctx.params["bits"]),
            }
        )
        return ctx
