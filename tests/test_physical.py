"""Correctness contracts for physical H2O/KIVI cache layouts."""

from __future__ import annotations

import math
import os
import unittest
from unittest import mock

import torch

from unicache.adapters import UniCacheHookPolicy
from unicache.adapters.bagel import (
    flash_attn_func,
    flash_attn_varlen_func,
    merge_attention_summaries,
)
from unicache.runtime import build_plan_from_config
from unicache.storage import GQAH2OCache, PackedKIVICache, TensorCacheStorageBackend
from unicache.storage.kivi_cuda import (
    dequantize_kivi_chunk,
    packed_kivi_quantized_attention_summary,
    select_kivi_kernel_backend,
)
from modeling.bagel.qwen2_navit import (
    clear_topk_kv_policy,
    set_topk_kv_policy,
    varlen_attention,
)


def _config(operator: str, segment_id: str, params: dict) -> dict:
    return {
        "task": {
            "task_type": "editing",
            "prompt": "edit",
            "source_images": ["source.jpg"],
            "output_modality": "image",
        },
        "planners": [{"id": "bagel_typed_segments", "enabled": True}],
        "rules": [
            {
                "id": f"{segment_id}_{operator}",
                "match": {"segment_id": segment_id, "phases": ["denoise"]},
                "operator": operator,
                "params": params,
                "stage": "pre_attention" if operator.startswith("kivi") else "attention",
            }
        ],
        "runtime": {"plan_only": False, "fail_on_conflict": True},
    }


class PhysicalEfficiencyTests(unittest.TestCase):
    def test_denoise_segment_token_count_reuses_cross_layer_metadata(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                _config(
                    "kivi_packed_quantization",
                    "source_vae",
                    {
                        "bits": 2,
                        "group_size": 128,
                        "residual_length": 32,
                        "backend": "reference",
                    },
                )
            ),
            collect_diagnostics=False,
        )
        policy.begin_run("segment-count")
        policy.set_context(
            phase="denoise", branch="generation", cfg_branch="main", step_index=0
        )
        step = policy.steps[0]
        mask = torch.tensor([False, True, True, False])
        first = policy._segment_token_count(
            step=step,
            runtime=policy._runtime_context(layer_idx=0, mode="gen", sample_idx=0),
            segment_mask=mask,
            key_len=4,
        )
        second = policy._segment_token_count(
            step=step,
            runtime=policy._runtime_context(layer_idx=27, mode="gen", sample_idx=0),
            segment_mask=mask,
            key_len=4,
        )
        first_span = policy._segment_contiguous_span(
            step=step,
            runtime=policy._runtime_context(layer_idx=0, mode="gen", sample_idx=0),
            segment_mask=mask,
            segment_tokens=first,
            key_len=4,
        )
        second_span = policy._segment_contiguous_span(
            step=step,
            runtime=policy._runtime_context(layer_idx=27, mode="gen", sample_idx=0),
            segment_mask=mask,
            segment_tokens=second,
            key_len=4,
        )

        self.assertEqual(2, first)
        self.assertEqual(first, second)
        self.assertEqual((1, 3), first_span)
        self.assertEqual(first_span, second_span)
        self.assertEqual(1, policy.realized_storage_totals["segment_count_builds"])
        self.assertEqual(1, policy.realized_storage_totals["segment_count_reuses"])
        self.assertEqual(1, policy.realized_storage_totals["segment_span_builds"])
        self.assertEqual(1, policy.realized_storage_totals["segment_span_reuses"])
        policy.begin_run("segment-count-next")
        self.assertEqual({}, policy.segment_count_cache)
        self.assertEqual({}, policy.segment_span_cache)

    def test_step_invariant_should_apply_is_cached_per_runtime_branch(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                _config(
                    "h2o_physical_gqa",
                    "instruction",
                    {"target_tokens": 8, "recent_tokens": 2},
                )
            ),
            collect_diagnostics=False,
        )
        policy.begin_run("should-apply-cache")
        policy.set_context(
            phase="denoise", branch="generation", cfg_branch="main", step_index=1
        )
        calls = 0
        original = policy._runtime_step_matches

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        policy._runtime_step_matches = counted
        first = policy.should_apply(0, "gen")
        policy.set_context(step_index=2)
        second = policy.should_apply(0, "gen")
        self.assertEqual(first, second)
        self.assertEqual(1, calls)
        self.assertEqual(1, len(policy.should_apply_cache))

        policy.set_context(cfg_branch="cfg_text")
        policy.should_apply(0, "gen")
        self.assertEqual(2, calls)
        self.assertEqual(2, len(policy.should_apply_cache))

    def test_progress_dependent_should_apply_is_not_cached(self):
        config = _config(
            "h2o_physical_gqa",
            "instruction",
            {"target_tokens": 8, "recent_tokens": 2},
        )
        config["rules"][0]["match"]["step_range"] = {"min": 1}
        policy = UniCacheHookPolicy(
            build_plan_from_config(config), collect_diagnostics=False
        )
        policy.begin_run("should-apply-dynamic")
        policy.set_context(
            phase="denoise", branch="generation", cfg_branch="main", step_index=1
        )
        calls = 0
        original = policy._runtime_step_matches

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        policy._runtime_step_matches = counted
        policy.should_apply(0, "gen")
        policy.set_context(step_index=2)
        policy.should_apply(0, "gen")
        self.assertEqual(2, calls)
        self.assertEqual({}, policy.should_apply_cache)

    def test_attention_summary_merge_matches_global_softmax(self):
        torch.manual_seed(11)
        scores_a = torch.randn(4, 3, 7)
        scores_b = torch.randn(4, 3, 5)
        value_a = torch.randn(4, 7, 8)
        value_b = torch.randn(4, 5, 8)
        probs_a = torch.softmax(scores_a, dim=-1)
        probs_b = torch.softmax(scores_b, dim=-1)
        summaries = [
            (torch.matmul(probs_a, value_a).transpose(0, 1), torch.logsumexp(scores_a, dim=-1)),
            (torch.matmul(probs_b, value_b).transpose(0, 1), torch.logsumexp(scores_b, dim=-1)),
        ]
        output, logsumexp = merge_attention_summaries(summaries)
        all_scores = torch.cat([scores_a, scores_b], dim=-1)
        all_values = torch.cat([value_a, value_b], dim=1)
        expected = torch.matmul(
            torch.softmax(all_scores, dim=-1), all_values
        ).transpose(0, 1)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5, rtol=1e-5))
        self.assertTrue(
            torch.allclose(logsumexp, torch.logsumexp(all_scores, dim=-1), atol=1e-6)
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and bool(os.environ.get("UNICACHE_KIVI_ROOT"))
        and flash_attn_func is not None,
        "requires CUDA, FlashAttention, and the pinned KIVI backend",
    )
    def test_fused_mixed_attention_matches_explicit_reference(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                {
                    "task": {
                        "task_type": "editing",
                        "prompt": "edit",
                        "source_images": ["source.jpg"],
                        "output_modality": "image",
                    },
                    "planners": [{"id": "bagel_typed_segments", "enabled": True}],
                    "rules": [],
                    "runtime": {"plan_only": False, "fail_on_conflict": True},
                }
            ),
            collect_diagnostics=False,
        )
        policy.begin_run("fused-mixed")
        policy.set_context(phase="denoise", branch="generation", step_index=1, total_steps=2)
        torch.manual_seed(12)
        query = torch.randn(37, 28, 128, device="cuda", dtype=torch.bfloat16)
        dense_key = torch.randn(64, 4, 128, device="cuda", dtype=torch.bfloat16)
        dense_value = torch.randn_like(dense_key)
        source_key = torch.randn(256, 4, 128, device="cuda", dtype=torch.bfloat16)
        source_value = torch.randn_like(source_key)
        h2o_source_key = torch.randn(
            128, 4, 128, device="cuda", dtype=torch.bfloat16
        )
        h2o_source_value = torch.randn_like(h2o_source_key)
        h2o = GQAH2OCache.from_dense(
            h2o_source_key,
            h2o_source_value,
            logical_token_ids=torch.arange(128, device="cuda"),
            token_type_id=2,
            per_kv_head_scores=torch.rand(4, 128, device="cuda"),
            target_tokens=32,
            recent_tokens=8,
            selection_frozen=True,
        )
        h2o_scores_before = h2o.cumulative_scores.clone()
        packed = PackedKIVICache.from_dense(
            source_key,
            source_value,
            logical_token_ids=torch.arange(256, device="cuda"),
            token_type_id=3,
            k_bits=2,
            v_bits=2,
            group_size=128,
            residual_length=0,
            backend="cuda",
        )
        policy._execute_stages = lambda *args, **kwargs: self.fail(
            "stable mixed physical cache should bypass operator dispatch"
        )
        classifier_calls = 0
        classify = policy._can_bypass_execution_for_static_mixed

        def counted_classify(**kwargs):
            nonlocal classifier_calls
            classifier_calls += 1
            return classify(**kwargs)

        policy._can_bypass_execution_for_static_mixed = counted_classify
        output = policy.apply_to_attention(
            q_i=query,
            k_i=dense_key,
            v_i=dense_value,
            attn_mask=None,
            key_type_ids=torch.full((64,), 4, device="cuda"),
            protected_mask=torch.ones(64, device="cuda", dtype=torch.bool),
            layer_idx=0,
            mode="gen",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=4,
            physical_segments=[h2o, packed],
        )
        repeated_output = policy.apply_to_attention(
            q_i=query,
            k_i=dense_key,
            v_i=dense_value,
            attn_mask=None,
            key_type_ids=torch.full((64,), 4, device="cuda"),
            protected_mask=torch.ones(64, device="cuda", dtype=torch.bool),
            layer_idx=0,
            mode="gen",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=4,
            physical_segments=[h2o, packed],
        )
        restored_key, restored_value = packed.dequantize(dtype=query.dtype)
        all_key = torch.cat(
            [dense_key, h2o.key.transpose(0, 1), restored_key], dim=0
        ).repeat_interleave(7, dim=1)
        all_value = torch.cat(
            [dense_value, h2o.value.transpose(0, 1), restored_value], dim=0
        ).repeat_interleave(7, dim=1)
        scores = torch.matmul(
            query.transpose(0, 1).float(),
            all_key.transpose(0, 1).float().transpose(-1, -2),
        ) / math.sqrt(query.shape[-1])
        expected = torch.matmul(
            torch.softmax(scores, dim=-1).to(all_value.dtype),
            all_value.transpose(0, 1),
        ).transpose(0, 1)
        self.assertTrue(torch.allclose(output, expected, atol=8e-2, rtol=8e-2))
        self.assertTrue(torch.equal(output, repeated_output))
        self.assertEqual(1, classifier_calls)
        self.assertTrue(torch.equal(h2o_scores_before, h2o.cumulative_scores))
        self.assertEqual(2, int(policy.realized_storage_totals["fused_mixed_attention_calls"]))
        self.assertEqual(1, int(policy.realized_storage_totals["mixed_workspace_allocations"]))
        self.assertEqual(1, int(policy.realized_storage_totals["mixed_workspace_reuses"]))
        self.assertGreater(
            int(policy.realized_storage_totals["mixed_workspace_peak_bytes"]), 0
        )
        self.assertEqual(1, len(policy.mixed_layout_cache))
        self.assertEqual(1, len(policy.mixed_workspace_cache))

    @unittest.skipUnless(
        torch.cuda.is_available()
        and bool(os.environ.get("UNICACHE_KIVI_ROOT"))
        and flash_attn_varlen_func is not None,
        "requires CUDA, FlashAttention, and the pinned KIVI backend",
    )
    def test_batched_physical_varlen_matches_per_sample_reference(self):
        with mock.patch.dict(
            os.environ,
            {
                "UNICACHE_BATCH_PHYSICAL_ATTENTION": "1",
                "UNICACHE_OVERLAP_BATCH_DEQUANT": "1",
            },
        ):
            policy = UniCacheHookPolicy(
                build_plan_from_config(
                    {
                        "task": {
                            "task_type": "editing",
                            "prompt": "edit",
                            "source_images": ["source.jpg"],
                            "output_modality": "image",
                        },
                        "planners": [
                            {"id": "bagel_typed_segments", "enabled": True}
                        ],
                        "rules": [],
                        "runtime": {"plan_only": False, "fail_on_conflict": True},
                    }
                ),
                collect_diagnostics=False,
            )
        policy.begin_run("physical-varlen")
        policy.set_context(
            phase="denoise",
            branch="generation",
            cfg_branch="main",
            cfg_branches=("main", "cfg_text"),
            step_index=1,
            total_steps=2,
        )
        torch.manual_seed(20260626)
        query_lens = (17, 23)
        dense_lens = (32, 48)
        source_lens = (128, 256)
        queries = [
            torch.randn(length, 28, 128, device="cuda", dtype=torch.bfloat16)
            for length in query_lens
        ]
        dense_keys = [
            torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16)
            for length in dense_lens
        ]
        dense_values = [torch.randn_like(key) for key in dense_keys]
        physical = []
        expected = []
        for sample_idx, source_len in enumerate(source_lens):
            h2o_source_key = torch.randn(
                64, 4, 128, device="cuda", dtype=torch.bfloat16
            )
            h2o_source_value = torch.randn_like(h2o_source_key)
            h2o = GQAH2OCache.from_dense(
                h2o_source_key,
                h2o_source_value,
                logical_token_ids=torch.arange(64, device="cuda"),
                token_type_id=2,
                per_kv_head_scores=torch.rand(4, 64, device="cuda"),
                target_tokens=24,
                recent_tokens=8,
                selection_frozen=True,
            )
            source_key = torch.randn(
                source_len, 4, 128, device="cuda", dtype=torch.bfloat16
            )
            source_value = torch.randn_like(source_key)
            packed = PackedKIVICache.from_dense(
                source_key,
                source_value,
                logical_token_ids=torch.arange(source_len, device="cuda"),
                token_type_id=3,
                k_bits=2,
                v_bits=2,
                group_size=128,
                residual_length=0,
                backend="cuda",
            )
            physical.append([h2o, packed])
            restored_key, restored_value = packed.dequantize(
                dtype=queries[sample_idx].dtype
            )
            joined_key = torch.cat(
                [dense_keys[sample_idx], h2o.key.transpose(0, 1), restored_key],
                dim=0,
            )
            joined_value = torch.cat(
                [
                    dense_values[sample_idx],
                    h2o.value.transpose(0, 1),
                    restored_value,
                ],
                dim=0,
            )
            expected.append(
                flash_attn_func(
                    queries[sample_idx].unsqueeze(0),
                    joined_key.unsqueeze(0),
                    joined_value.unsqueeze(0),
                    dropout_p=0.0,
                    causal=False,
                ).squeeze(0)
            )

        cu_q = torch.tensor(
            [0, query_lens[0], sum(query_lens)], device="cuda", dtype=torch.int32
        )
        cu_k = torch.tensor(
            [0, dense_lens[0], sum(dense_lens)],
            device="cuda",
            dtype=torch.int32,
        )
        output = policy.apply_static_physical_attention_batch(
            q=torch.cat(queries, dim=0),
            k=torch.cat(dense_keys, dim=0),
            v=torch.cat(dense_values, dim=0),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            layer_idx=0,
            mode="gen",
            causal=False,
            physical_cache_by_sample=physical,
        )
        repeated_output = policy.apply_static_physical_attention_batch(
            q=torch.cat(queries, dim=0),
            k=torch.cat(dense_keys, dim=0),
            v=torch.cat(dense_values, dim=0),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            layer_idx=0,
            mode="gen",
            causal=False,
            physical_cache_by_sample=physical,
        )
        torch.cuda.synchronize()
        self.assertIsNotNone(output)
        self.assertTrue(
            torch.allclose(
                output,
                torch.cat(expected, dim=0),
                atol=8e-2,
                rtol=8e-2,
            )
        )
        self.assertTrue(torch.equal(output, repeated_output))
        self.assertEqual(
            2,
            int(policy.realized_storage_totals["physical_varlen_attention_calls"]),
        )
        self.assertEqual(
            4,
            int(policy.realized_storage_totals["physical_varlen_attention_samples"]),
        )
        self.assertEqual(
            1, int(policy.realized_storage_totals["physical_varlen_cu_builds"])
        )
        self.assertEqual(
            1, int(policy.realized_storage_totals["physical_varlen_cu_reuses"])
        )
        self.assertEqual(
            2,
            int(
                policy.realized_storage_totals[
                    "physical_varlen_overlap_dequant_calls"
                ]
            ),
        )

    def test_kivi_dispatches_multi_query_attention_to_tiled_kernel(self):
        self.assertEqual("official_gemv", select_kivi_kernel_backend(1))
        self.assertEqual("triton_tiled", select_kivi_kernel_backend(2))
        self.assertEqual("triton_tiled", select_kivi_kernel_backend(802))

    @unittest.skipUnless(
        torch.cuda.is_available()
        and bool(os.environ.get("UNICACHE_KIVI_ROOT"))
        and flash_attn_func is not None,
        "requires CUDA, FlashAttention, and the pinned KIVI backend",
    )
    def test_streamed_mixed_attention_matches_joined_flash(self):
        torch.manual_seed(20260626)
        query = torch.randn(37, 28, 128, device="cuda", dtype=torch.bfloat16)
        dense_key = torch.randn(96, 4, 128, device="cuda", dtype=torch.bfloat16)
        dense_value = torch.randn_like(dense_key)
        source_key = torch.randn(256, 4, 128, device="cuda", dtype=torch.bfloat16)
        source_value = torch.randn_like(source_key)
        packed = PackedKIVICache.from_dense(
            source_key,
            source_value,
            logical_token_ids=torch.arange(256, device="cuda"),
            token_type_id=3,
            k_bits=2,
            v_bits=2,
            group_size=128,
            residual_length=0,
            backend="cuda",
        )

        def run(backend: str) -> tuple[torch.Tensor, UniCacheHookPolicy]:
            with mock.patch.dict(
                os.environ, {"UNICACHE_KIVI_ATTENTION_BACKEND": backend}
            ):
                policy = UniCacheHookPolicy(
                    build_plan_from_config(
                        {
                            "task": {
                                "task_type": "editing",
                                "prompt": "edit",
                                "source_images": ["source.jpg"],
                                "output_modality": "image",
                            },
                            "planners": [
                                {"id": "bagel_typed_segments", "enabled": True}
                            ],
                            "rules": [],
                            "runtime": {
                                "plan_only": False,
                                "fail_on_conflict": True,
                            },
                        }
                    ),
                    collect_diagnostics=False,
                )
            policy.begin_run(f"mixed-{backend}")
            policy.set_context(
                phase="denoise",
                branch="generation",
                cfg_branch="main",
                step_index=1,
                total_steps=2,
            )
            runtime = policy._runtime_context(
                layer_idx=0, mode="gen", sample_idx=0
            )
            output = policy._run_fused_mixed_attention(
                query=query,
                dense_key=dense_key,
                dense_value=dense_value,
                physical_segments=[packed],
                runtime=runtime,
            )
            return output, policy

        joined, _ = run("chunked_flash")
        overlap, overlap_policy = run("chunked_flash_overlap")
        dual_stream, dual_policy = run("chunked_flash_dual_stream")
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(joined, overlap))
        self.assertTrue(
            torch.allclose(joined, dual_stream, atol=8e-2, rtol=8e-2)
        )
        self.assertEqual(
            1,
            int(
                overlap_policy.realized_storage_totals[
                    "fused_mixed_overlap_calls"
                ]
            ),
        )
        self.assertEqual(
            1,
            int(
                dual_policy.realized_storage_totals[
                    "fused_mixed_dual_stream_calls"
                ]
            ),
        )
        self.assertEqual(1, len(dual_policy.mixed_auxiliary_streams))

    @unittest.skipUnless(
        torch.cuda.is_available() and bool(os.environ.get("UNICACHE_KIVI_ROOT")),
        "requires the pinned KIVI CUDA extension",
    )
    def test_kivi_fused_summary_matches_explicit_dequant_reference(self):
        torch.manual_seed(20260626)
        for bits in (2, 4):
            key = torch.randn(256, 4, 128, device="cuda", dtype=torch.bfloat16)
            value = torch.randn_like(key)
            query = torch.randn(37, 28, 128, device="cuda", dtype=torch.bfloat16)
            cache = PackedKIVICache.from_dense(
                key,
                value,
                logical_token_ids=torch.arange(256, device="cuda"),
                token_type_id=3,
                k_bits=bits,
                v_bits=bits,
                group_size=128,
                residual_length=0,
                backend="cuda",
            )
            output, logsumexp = packed_kivi_quantized_attention_summary(cache, query)
            restored_key, restored_value = cache.dequantize(dtype=query.dtype)
            restored_key = restored_key.repeat_interleave(7, dim=1)
            restored_value = restored_value.repeat_interleave(7, dim=1)
            scores = torch.matmul(
                query.transpose(0, 1).float(),
                restored_key.transpose(0, 1).float().transpose(-1, -2),
            ) / math.sqrt(query.shape[-1])
            expected_output = torch.matmul(
                torch.softmax(scores, dim=-1).to(restored_value.dtype),
                restored_value.transpose(0, 1),
            ).transpose(0, 1)
            self.assertTrue(
                torch.allclose(output, expected_output, atol=8e-2, rtol=8e-2)
            )
            self.assertTrue(
                torch.allclose(logsumexp, torch.logsumexp(scores, dim=-1), atol=5e-2, rtol=5e-2)
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and bool(os.environ.get("UNICACHE_KIVI_ROOT")),
        "requires the pinned KIVI CUDA extension",
    )
    def test_kivi_chunk_dequantization_matches_packed_reference(self):
        torch.manual_seed(13)
        for bits in (2, 4):
            key = torch.randn(384, 4, 128, device="cuda", dtype=torch.bfloat16)
            value = torch.randn_like(key)
            cache = PackedKIVICache.from_dense(
                key,
                value,
                logical_token_ids=torch.arange(384, device="cuda"),
                token_type_id=3,
                k_bits=bits,
                v_bits=bits,
                group_size=128,
                residual_length=0,
                backend="cuda",
            )
            expected_key, expected_value = cache.dequantize(dtype=torch.bfloat16)
            actual_key, actual_value = dequantize_kivi_chunk(
                cache, start=128, stop=384, dtype=torch.bfloat16
            )
            self.assertTrue(torch.equal(actual_key, expected_key[128:384]))
            self.assertTrue(torch.equal(actual_value, expected_value[128:384]))
            workspace_key = torch.full(
                (300, 4, 128), -7.0, device="cuda", dtype=torch.bfloat16
            )
            workspace_value = torch.full_like(workspace_key, -9.0)
            direct_key, direct_value = dequantize_kivi_chunk(
                cache,
                start=128,
                stop=384,
                dtype=torch.bfloat16,
                output_key=workspace_key,
                output_value=workspace_value,
                output_offset=16,
            )
            self.assertTrue(torch.equal(direct_key, expected_key[128:384]))
            self.assertTrue(torch.equal(direct_value, expected_value[128:384]))
            self.assertTrue(torch.all(workspace_key[:16] == -7.0))
            self.assertTrue(torch.all(workspace_value[:16] == -9.0))

    def test_varlen_attention_preserves_native_gqa_for_capable_policy(self):
        class NativeGQASpy:
            enabled = True
            phase = "denoise"
            accepts_native_gqa = True
            accepts_storage_kv_metadata = True
            accepts_physical_cache_metadata = True

            def __init__(self):
                self.observed_kv_heads = None
                self.observed_storage_heads = None

            def should_apply(self, layer_idx, mode):
                return True

            def apply_to_attention(
                self,
                *,
                q_i,
                k_i,
                v_i,
                storage_num_kv_heads,
                **kwargs,
            ):
                del v_i, kwargs
                self.observed_kv_heads = int(k_i.shape[1])
                self.observed_storage_heads = int(storage_num_kv_heads)
                return torch.zeros_like(q_i)

        policy = NativeGQASpy()
        query = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        key = torch.randn(5, 2, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        set_topk_kv_policy(policy)
        try:
            output = varlen_attention(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
                max_seqlen_q=3,
                max_seqlen_k=5,
                causal=False,
                layer_idx=0,
                mode="gen",
            )
        finally:
            clear_topk_kv_policy()
        self.assertEqual(query.shape, output.shape)
        self.assertEqual(2, policy.observed_storage_heads)
        self.assertEqual(2, policy.observed_kv_heads)

    def test_varlen_attention_bypasses_generic_preparation_for_static_cache(self):
        class StaticPhysicalSpy:
            enabled = True
            phase = "denoise"
            accepts_native_gqa = True

            def __init__(self):
                self.calls = 0

            def should_apply(self, layer_idx, mode):
                return True

            def apply_static_physical_attention(self, *, q_i, **kwargs):
                del kwargs
                self.calls += 1
                return torch.zeros_like(q_i)

            def apply_to_attention(self, **kwargs):
                del kwargs
                raise AssertionError("generic hook preparation should be bypassed")

        policy = StaticPhysicalSpy()
        query = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        key = torch.randn(5, 2, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        set_topk_kv_policy(policy)
        try:
            output = varlen_attention(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
                max_seqlen_q=3,
                max_seqlen_k=5,
                causal=False,
                layer_idx=0,
                mode="gen",
                physical_cache_by_sample=[[object()]],
            )
        finally:
            clear_topk_kv_policy()
        self.assertEqual(query.shape, output.shape)
        self.assertEqual(1, policy.calls)

    def test_mixed_physical_attention_accepts_native_gqa_dense_cache(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                {
                    "task": {
                        "task_type": "editing",
                        "prompt": "edit",
                        "source_images": ["source.jpg"],
                        "output_modality": "image",
                    },
                    "planners": [{"id": "bagel_typed_segments", "enabled": True}],
                    "rules": [
                        {
                            "id": "identity_current",
                            "match": {"segment_id": "current_vae"},
                            "operator": "identity",
                            "stage": "attention",
                        }
                    ],
                    "runtime": {"plan_only": False, "fail_on_conflict": True},
                }
            )
        )
        policy.begin_run("native-gqa")
        policy.set_context(
            phase="denoise", branch="generation", step_index=1, total_steps=2
        )
        torch.manual_seed(5)
        query = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        dense_key = torch.randn(5, 2, 8, dtype=torch.bfloat16)
        dense_value = torch.randn_like(dense_key)
        source_key = torch.randn(12, 2, 8, dtype=torch.bfloat16)
        source_value = torch.randn_like(source_key)
        packed = PackedKIVICache.from_dense(
            source_key,
            source_value,
            logical_token_ids=torch.arange(12),
            token_type_id=3,
            k_bits=4,
            v_bits=4,
            group_size=4,
            residual_length=4,
            backend="reference",
        )
        output = policy.apply_to_attention(
            q_i=query,
            k_i=dense_key,
            v_i=dense_value,
            attn_mask=None,
            key_type_ids=torch.tensor([4] * 5),
            protected_mask=torch.zeros(5, dtype=torch.bool),
            layer_idx=0,
            mode="gen",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=2,
            physical_segments=[packed],
        )
        repeated_dense_key = dense_key.repeat_interleave(2, dim=1)
        repeated_dense_value = dense_value.repeat_interleave(2, dim=1)
        restored_key, restored_value = packed.dequantize(dtype=query.dtype)
        restored_key = restored_key.repeat_interleave(2, dim=1)
        restored_value = restored_value.repeat_interleave(2, dim=1)
        all_key = torch.cat([repeated_dense_key, restored_key], dim=0)
        all_value = torch.cat([repeated_dense_value, restored_value], dim=0)
        scores = torch.matmul(
            query.transpose(0, 1).float(),
            all_key.transpose(0, 1).float().transpose(-1, -2),
        ) / math.sqrt(query.shape[-1])
        expected = torch.matmul(
            torch.softmax(scores, dim=-1).to(all_value.dtype),
            all_value.transpose(0, 1),
        ).transpose(0, 1)
        self.assertTrue(torch.allclose(output, expected, atol=2e-2, rtol=2e-2))

    @unittest.skipUnless(
        torch.cuda.is_available() and bool(os.environ.get("UNICACHE_KIVI_ROOT")),
        "requires the pinned KIVI CUDA extension",
    )
    def test_kivi_cuda_qk_av_matches_explicit_dequant_reference(self):
        torch.manual_seed(20260626)
        for bits in (2, 4):
            key = torch.randn(64, 4, 128, device="cuda", dtype=torch.bfloat16)
            value = torch.randn_like(key)
            query = torch.randn(1, 28, 128, device="cuda", dtype=torch.bfloat16)
            cache = PackedKIVICache.from_dense(
                key,
                value,
                logical_token_ids=torch.arange(64, device="cuda"),
                token_type_id=3,
                k_bits=bits,
                v_bits=bits,
                group_size=64,
                residual_length=0,
                backend="cuda",
            )
            logits, apply_value = cache.attention_parts(query)
            restored_key, restored_value = cache.dequantize(dtype=query.dtype)
            restored_key = restored_key.repeat_interleave(7, dim=1)
            restored_value = restored_value.repeat_interleave(7, dim=1)
            expected_logits = torch.matmul(
                query.transpose(0, 1).float(),
                restored_key.transpose(0, 1).float().transpose(-1, -2),
            ) / math.sqrt(query.shape[-1])
            self.assertTrue(
                torch.allclose(logits, expected_logits, atol=8e-2, rtol=8e-2),
                f"{bits}-bit KIVI QK kernel diverged from explicit dequantization",
            )
            probabilities = torch.softmax(expected_logits, dim=-1).to(torch.bfloat16)
            expected_output = torch.matmul(
                probabilities, restored_value.transpose(0, 1)
            )
            self.assertTrue(
                torch.allclose(
                    apply_value(probabilities), expected_output, atol=8e-2, rtol=8e-2
                ),
                f"{bits}-bit KIVI AV kernel diverged from explicit dequantization",
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and bool(os.environ.get("UNICACHE_KIVI_ROOT")),
        "requires the pinned KIVI CUDA extension",
    )
    def test_kivi_cuda_long_query_matches_explicit_dequant_reference(self):
        """BAGEL denoising uses hundreds of queries, unlike LLM decoding."""

        torch.manual_seed(20260626)
        for bits in (2, 4):
            key = torch.randn(256, 4, 128, device="cuda", dtype=torch.bfloat16)
            value = torch.randn_like(key)
            cache = PackedKIVICache.from_dense(
                key,
                value,
                logical_token_ids=torch.arange(256, device="cuda"),
                token_type_id=3,
                k_bits=bits,
                v_bits=bits,
                group_size=128,
                residual_length=0,
                backend="cuda",
            )
            restored_key, restored_value = cache.dequantize(dtype=key.dtype)
            restored_key = restored_key.repeat_interleave(7, dim=1)
            restored_value = restored_value.repeat_interleave(7, dim=1)
            for query_length in (37, 802):
                query = torch.randn(
                    query_length, 28, 128, device="cuda", dtype=torch.bfloat16
                )
                logits, apply_value = cache.attention_parts(query)
                expected_logits = torch.matmul(
                    query.transpose(0, 1).float(),
                    restored_key.transpose(0, 1).float().transpose(-1, -2),
                ) / math.sqrt(query.shape[-1])
                self.assertTrue(
                    torch.allclose(logits, expected_logits, atol=8e-2, rtol=8e-2),
                    f"{bits}-bit KIVI QK diverged for q_len={query_length}",
                )
                probabilities = torch.softmax(expected_logits, dim=-1).to(torch.bfloat16)
                expected_output = torch.matmul(
                    probabilities, restored_value.transpose(0, 1)
                )
                self.assertTrue(
                    torch.allclose(
                        apply_value(probabilities),
                        expected_output,
                        atol=8e-2,
                        rtol=8e-2,
                    ),
                    f"{bits}-bit KIVI AV diverged for q_len={query_length}",
                )

    def test_kivi_pack_round_trip_and_real_byte_accounting(self):
        torch.manual_seed(0)
        key = torch.randn(96, 4, 128, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        for bits in (2, 4, 8):
            cache = PackedKIVICache.from_dense(
                key,
                value,
                logical_token_ids=torch.arange(96),
                token_type_id=3,
                k_bits=bits,
                v_bits=bits,
                group_size=32,
                residual_length=16,
                backend="reference",
            )
            restored_key, restored_value = cache.dequantize()
            self.assertEqual(key.shape, restored_key.shape)
            self.assertEqual(value.shape, restored_value.shape)
            self.assertTrue(torch.equal(key[-16:], restored_key[-16:]))
            self.assertTrue(torch.equal(value[-16:], restored_value[-16:]))
            self.assertEqual(0, cache.quantized_tokens % cache.group_size)
            self.assertLess(cache.resident_bytes, cache.full_precision_bytes)
            self.assertGreater(float((key - restored_key).abs().float().mean()), 0.0)

    def test_vectorized_kivi_group_quantization_matches_reference(self):
        from unicache.storage.physical import (
            _quantize_grouped,
            _quantize_grouped_reference,
        )

        torch.manual_seed(19)
        cases = (
            (torch.randn(256, 4, 128, dtype=torch.bfloat16), 0, 0),
            (torch.randn(256, 4, 128, dtype=torch.bfloat16), 2, 2),
        )
        for data, group_dim, pack_dim in cases:
            for group_size in (32, 64, 128):
                for bits in (2, 4, 8):
                    expected = _quantize_grouped_reference(
                        data,
                        bits=bits,
                        group_dim=group_dim,
                        group_size=group_size,
                        pack_dim=pack_dim,
                    )
                    actual = _quantize_grouped(
                        data,
                        bits=bits,
                        group_dim=group_dim,
                        group_size=group_size,
                        pack_dim=pack_dim,
                    )
                    self.assertEqual(expected[3], actual[3])
                    self.assertTrue(torch.equal(expected[0], actual[0]))
                    self.assertTrue(torch.equal(expected[1], actual[1]))
                    self.assertTrue(torch.equal(expected[2], actual[2]))

    @unittest.skipUnless(
        torch.cuda.is_available() and bool(os.environ.get("UNICACHE_KIVI_ROOT")),
        "requires CUDA and the pinned KIVI backend",
    )
    def test_kivi_cuda_cache_persists_kernel_native_layout(self):
        torch.manual_seed(6)
        key = torch.randn(128, 2, 64, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        cuda_layout = PackedKIVICache.from_dense(
            key,
            value,
            logical_token_ids=torch.arange(128, device="cuda"),
            token_type_id=3,
            k_bits=2,
            v_bits=2,
            group_size=64,
            residual_length=0,
            backend="cuda",
        )
        reference = PackedKIVICache.from_dense(
            key,
            value,
            logical_token_ids=torch.arange(128, device="cuda"),
            token_type_id=3,
            k_bits=2,
            v_bits=2,
            group_size=64,
            residual_length=0,
            backend="reference",
        )
        self.assertTrue(cuda_layout.kernel_native_layout)
        self.assertEqual((2, 64, 8), tuple(cuda_layout.key_code.shape))
        self.assertEqual((2, 64, 2), tuple(cuda_layout.key_scale.shape))
        self.assertEqual((2, 128, 4), tuple(cuda_layout.value_code.shape))
        self.assertEqual((2, 128, 1), tuple(cuda_layout.value_scale.shape))
        self.assertEqual(torch.float16, cuda_layout.key_scale.dtype)
        self.assertEqual(torch.float16, cuda_layout.value_scale.dtype)
        restored = cuda_layout.dequantize(dtype=torch.bfloat16)
        expected = reference.dequantize(dtype=torch.bfloat16)
        for original, native, oracle in zip(
            (key, value), restored, expected
        ):
            self.assertTrue(bool(torch.isfinite(native).all()))
            native_error = (native.float() - original.float()).abs().mean()
            oracle_error = (oracle.float() - original.float()).abs().mean()
            # Both paths implement KIVI asymmetric min-max quantization.  The
            # official Triton packer performs its reductions/rounding in the
            # input precision, whereas the CPU/reference oracle uses FP32
            # temporaries, so packed codes need not be bit-identical.
            self.assertLessEqual(
                float(native_error), float(oracle_error) * 1.10 + 1e-3
            )

    def test_kivi_expands_residual_to_avoid_partial_cuda_group(self):
        key = torch.randn(320, 4, 128, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        cache = PackedKIVICache.from_dense(
            key,
            value,
            logical_token_ids=torch.arange(320),
            token_type_id=3,
            k_bits=2,
            v_bits=2,
            group_size=128,
            residual_length=32,
            backend="reference",
        )
        self.assertEqual(256, cache.quantized_tokens)
        self.assertEqual(64, cache.residual_key.shape[0])
        self.assertEqual(
            cache.residual_key.numel() * cache.residual_key.element_size(),
            cache.residual_key.untyped_storage().nbytes(),
        )
        self.assertEqual(
            cache.residual_value.numel() * cache.residual_value.element_size(),
            cache.residual_value.untyped_storage().nbytes(),
        )
        restored_key, restored_value = cache.dequantize()
        self.assertEqual(key.shape, restored_key.shape)
        self.assertTrue(torch.equal(key[-64:], restored_key[-64:]))
        self.assertTrue(torch.equal(value[-64:], restored_value[-64:]))

    def test_kivi_precision_and_h2o_eviction_are_monotonic(self):
        key = torch.randn(64, 2, 16, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        packed = PackedKIVICache.from_dense(
            key,
            value,
            logical_token_ids=torch.arange(64),
            token_type_id=3,
            k_bits=4,
            v_bits=4,
            group_size=8,
            residual_length=8,
            backend="reference",
        )
        self.assertTrue(packed.repack_monotonic(bits=2, group_size=16))
        self.assertEqual(2, packed.k_bits)
        self.assertFalse(packed.repack_monotonic(bits=4, group_size=8))
        self.assertEqual(1, packed.budget_shortfall_events)

        scores = torch.rand(2, 64)
        h2o = GQAH2OCache.from_dense(
            key,
            value,
            logical_token_ids=torch.arange(64),
            token_type_id=1,
            per_kv_head_scores=scores,
            target_tokens=16,
            recent_tokens=4,
        )
        self.assertEqual(8, h2o.request_budget(8, 2))
        self.assertEqual(8, h2o.retained_tokens)
        self.assertEqual(8, h2o.request_budget(16, 4))
        self.assertEqual(8, h2o.retained_tokens)
        self.assertEqual(8, h2o.budget_shortfall_tokens)

    def test_physical_kivi_request_removes_dense_bf16_segment(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                _config(
                    "kivi_packed_quantization",
                    "source_vae",
                    {
                        "bits": 4,
                        "group_size": 8,
                        "residual_length": 4,
                        "backend": "reference",
                    },
                )
            )
        )
        policy.begin_run("physical-kivi")
        policy.set_context(phase="denoise", branch="generation", step_index=0, total_steps=2)
        torch.manual_seed(1)
        native_key = torch.randn(20, 2, 8, dtype=torch.bfloat16)
        native_value = torch.randn_like(native_key)
        type_ids = torch.tensor([1] * 4 + [3] * 12 + [4] * 4)
        query = torch.randn(4, 4, 8, dtype=torch.bfloat16)
        repeated_key = native_key.repeat_interleave(2, dim=1)
        repeated_value = native_value.repeat_interleave(2, dim=1)
        policy.apply_to_attention(
            q_i=query,
            k_i=repeated_key,
            v_i=repeated_value,
            attn_mask=None,
            key_type_ids=type_ids,
            protected_mask=torch.zeros(20, dtype=torch.bool),
            layer_idx=0,
            mode="gen",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=2,
        )
        with mock.patch(
            "unicache.storage.tensor_backend.torch.unique",
            side_effect=AssertionError("typed physical request should not rediscover its type"),
        ), mock.patch(
            "unicache.storage.tensor_backend.torch.nonzero",
            side_effect=AssertionError("contiguous typed request should use its cached span"),
        ):
            mutation = policy.apply_to_cache_update(
                k=native_key,
                v=native_value,
                key_type_ids=type_ids,
                sample_lens=torch.tensor([20]),
                layer_idx=0,
                mode="gen",
            )
        self.assertIsNotNone(mutation)
        self.assertEqual(8, mutation.key.shape[0])
        self.assertFalse(bool((mutation.key_type_ids == 3).any()))
        packed = mutation.physical_segments_by_sample[0][0]
        self.assertIsInstance(packed, PackedKIVICache)
        self.assertEqual(3, packed.token_type_id)
        self.assertEqual(12, packed.token_count)
        self.assertLess(packed.resident_bytes, packed.full_precision_bytes)

    def test_h2o_gqa_selects_independently_per_kv_head(self):
        key = torch.arange(8 * 2 * 4, dtype=torch.float32).reshape(8, 2, 4)
        value = key + 1000
        scores = torch.tensor(
            [
                [9.0, 8.0, 5.0, 4.0, 3.0, 2.0, 0.0, 0.0],
                [0.0, 8.0, 1.0, 1.0, 1.0, 9.0, 0.0, 0.0],
            ]
        )
        cache = GQAH2OCache.from_dense(
            key,
            value,
            logical_token_ids=torch.arange(8),
            token_type_id=1,
            per_kv_head_scores=scores,
            target_tokens=4,
            recent_tokens=2,
        )
        self.assertEqual([0, 1, 6, 7], cache.logical_token_ids[0].tolist())
        self.assertEqual([1, 5, 6, 7], cache.logical_token_ids[1].tolist())
        self.assertLess(cache.resident_bytes, cache.full_precision_bytes)

    def test_physical_h2o_observes_full_then_compacts_native_gqa(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                _config(
                    "h2o_physical_gqa",
                    "instruction",
                    {
                        "target_budget_ratio": 0.5,
                        "recent_budget_fraction": 0.25,
                        "selection_frozen": True,
                    },
                )
            )
        )
        policy.begin_run("physical-h2o")
        policy.set_context(phase="denoise", branch="generation", step_index=0, total_steps=2)
        torch.manual_seed(2)
        native_key = torch.randn(16, 2, 8, dtype=torch.bfloat16)
        native_value = torch.randn_like(native_key)
        type_ids = torch.tensor([1] * 12 + [4] * 4)
        query = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        output = policy.apply_to_attention(
            q_i=query,
            k_i=native_key.repeat_interleave(2, dim=1),
            v_i=native_value.repeat_interleave(2, dim=1),
            attn_mask=None,
            key_type_ids=type_ids,
            protected_mask=torch.zeros(16, dtype=torch.bool),
            layer_idx=0,
            mode="gen",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=2,
        )
        dense_reference = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            native_key.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
            native_value.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
        ).squeeze(0).transpose(0, 1)
        self.assertTrue(torch.allclose(output, dense_reference, atol=2e-2, rtol=2e-2))
        mutation = policy.apply_to_cache_update(
            k=native_key,
            v=native_value,
            key_type_ids=type_ids,
            sample_lens=torch.tensor([16]),
            layer_idx=0,
            mode="gen",
        )
        self.assertEqual(4, mutation.key.shape[0])
        h2o = mutation.physical_segments_by_sample[0][0]
        self.assertIsInstance(h2o, GQAH2OCache)
        self.assertEqual(1, h2o.token_type_id)
        self.assertEqual(6, h2o.retained_tokens)
        self.assertTrue(h2o.selection_frozen)

    def test_deferred_h2o_requests_keep_per_segment_type_metadata(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                {
                    "task": {
                        "task_type": "understanding",
                        "prompt": "read",
                        "source_images": ["source.jpg"],
                        "output_modality": "text",
                    },
                    "planners": [{"id": "bagel_typed_segments", "enabled": True}],
                    "rules": [
                        {
                            "id": "instruction_h2o",
                            "match": {
                                "segment_id": "instruction",
                                "phases": ["text_decode"],
                            },
                            "operator": "h2o_physical_gqa",
                            "params": {
                                "target_budget_ratio": 0.5,
                                "recent_budget_fraction": 0.25,
                                "selection_frozen": True,
                            },
                            "stage": "attention",
                        },
                        {
                            "id": "source_vit_h2o",
                            "match": {
                                "segment_id": "source_vit",
                                "phases": ["text_decode"],
                            },
                            "operator": "h2o_physical_gqa",
                            "params": {
                                "target_budget_ratio": 0.5,
                                "recent_budget_fraction": 0.25,
                                "selection_frozen": True,
                            },
                            "stage": "attention",
                        },
                    ],
                    "runtime": {"plan_only": False, "fail_on_conflict": True},
                }
            ),
            collect_diagnostics=False,
        )
        policy.begin_run("typed-deferred-h2o")
        policy.set_context(
            phase="text_decode", branch="understanding", decode_index=0
        )
        torch.manual_seed(13)
        key = torch.randn(16, 2, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        type_ids = torch.tensor([1] * 8 + [2] * 8)
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        policy.apply_to_attention(
            q_i=query,
            k_i=key,
            v_i=value,
            attn_mask=None,
            key_type_ids=type_ids,
            protected_mask=torch.zeros(16, dtype=torch.bool),
            layer_idx=0,
            mode="und",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=2,
        )
        mutation = policy.apply_to_cache_update(
            k=key,
            v=value,
            key_type_ids=type_ids,
            sample_lens=torch.tensor([16]),
            layer_idx=0,
            mode="und",
        )
        self.assertEqual(
            [1, 2],
            sorted(
                segment.token_type_id
                for segment in mutation.physical_segments_by_sample[0]
            ),
        )

    def test_disabling_runtime_diagnostics_keeps_physical_h2o_execution(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                _config(
                    "h2o_physical_gqa",
                    "instruction",
                    {"target_budget_ratio": 0.5, "recent_budget_fraction": 0.25},
                )
            ),
            collect_diagnostics=False,
        )
        policy.begin_run("physical-h2o-no-diagnostics")
        policy.set_context(
            phase="denoise", branch="generation", step_index=0, total_steps=2
        )
        torch.manual_seed(7)
        key = torch.randn(16, 2, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        type_ids = torch.tensor([1] * 12 + [4] * 4)
        query = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        output = policy.apply_to_attention(
            q_i=query,
            k_i=key,
            v_i=value,
            attn_mask=None,
            key_type_ids=type_ids,
            protected_mask=torch.zeros(16, dtype=torch.bool),
            layer_idx=0,
            mode="gen",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=2,
        )
        mutation = policy.apply_to_cache_update(
            k=key,
            v=value,
            key_type_ids=type_ids,
            sample_lens=torch.tensor([16]),
            layer_idx=0,
            mode="gen",
        )
        self.assertEqual(query.shape, output.shape)
        self.assertIsInstance(
            mutation.physical_segments_by_sample[0][0], GQAH2OCache
        )
        self.assertEqual([], policy.context_history)
        self.assertEqual([], policy.decision_history)
        self.assertEqual([], policy.tensor_shape_trace)
        self.assertEqual({}, dict(policy.coverage_totals))
        self.assertEqual({}, dict(policy.coverage_rules))
        self.assertEqual({}, dict(policy.stats))
        type_accounting = policy.summary()["realized_memory"]["physical_cache_types"]
        self.assertEqual(3, policy.summary()["realized_memory"]["max_denoise_query_tokens"])
        self.assertEqual(12, type_accounting["instruction"]["original_tokens"])
        self.assertEqual(6, type_accounting["instruction"]["retained_tokens"])
        self.assertLess(
            type_accounting["instruction"]["resident_bytes"],
            type_accounting["instruction"]["full_bf16_bytes"],
        )

    def test_frozen_h2o_segments_use_fused_global_attention(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                {
                    "task": {
                        "task_type": "understanding",
                        "prompt": "read",
                        "source_images": ["source.jpg"],
                        "output_modality": "text",
                    },
                    "planners": [{"id": "bagel_typed_segments", "enabled": True}],
                    "rules": [
                        {
                            "id": "decoded_identity",
                            "match": {"segment_id": "decoded_text"},
                            "operator": "identity",
                            "stage": "attention",
                        }
                    ],
                    "runtime": {"plan_only": False, "fail_on_conflict": True},
                }
            ),
            collect_diagnostics=False,
        )
        policy.begin_run("fused-static-h2o")
        policy.set_context(
            phase="text_decode", branch="understanding", step_index=1, total_steps=4
        )
        torch.manual_seed(8)
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        dense_key = torch.randn(3, 2, 8, dtype=torch.bfloat16)
        dense_value = torch.randn_like(dense_key)
        source_key = torch.randn(12, 2, 8, dtype=torch.bfloat16)
        source_value = torch.randn_like(source_key)
        h2o = GQAH2OCache.from_dense(
            source_key,
            source_value,
            logical_token_ids=torch.arange(12),
            token_type_id=2,
            per_kv_head_scores=torch.rand(2, 12),
            target_tokens=4,
            recent_tokens=1,
            selection_frozen=True,
        )
        scores_before = h2o.cumulative_scores.clone()
        policy._execute_stages = lambda *args, **kwargs: self.fail(
            "frozen physical H2O decode should bypass operator execution"
        )
        output = policy.apply_to_attention(
            q_i=query,
            k_i=dense_key,
            v_i=dense_value,
            attn_mask=torch.ones(1, 3, dtype=torch.bool),
            key_type_ids=torch.tensor([5, 6, 6]),
            protected_mask=torch.ones(3, dtype=torch.bool),
            layer_idx=0,
            mode="und",
            sample_idx=0,
            causal=True,
            storage_num_kv_heads=2,
            physical_segments=[h2o],
        )
        combined_key = torch.cat(
            [dense_key, h2o.key.transpose(0, 1)], dim=0
        ).repeat_interleave(2, dim=1)
        combined_value = torch.cat(
            [dense_value, h2o.value.transpose(0, 1)], dim=0
        ).repeat_interleave(2, dim=1)
        expected_scores = torch.matmul(
            query.transpose(0, 1).float(),
            combined_key.transpose(0, 1).float().transpose(-1, -2),
        ) / math.sqrt(query.shape[-1])
        expected = torch.matmul(
            torch.softmax(expected_scores, dim=-1).to(combined_value.dtype),
            combined_value.transpose(0, 1),
        ).transpose(0, 1)
        self.assertTrue(torch.allclose(output, expected, atol=2e-2, rtol=2e-2))
        self.assertTrue(torch.equal(scores_before, h2o.cumulative_scores))
        self.assertEqual(
            1,
            int(policy.realized_storage_totals["fused_h2o_attention_calls"]),
        )

    def test_unmatched_physical_rule_uses_attention_passthrough(self):
        config = _config(
            "h2o_physical_gqa",
            "source_vit",
            {"target_budget_ratio": 0.2, "recent_budget_fraction": 0.25},
        )
        config["rules"][0]["match"]["phases"] = ["text_decode"]
        config["rules"].append(
            {
                "id": "boundary_protect",
                "match": {"segment_id": "boundary"},
                "operator": "protect",
                "stage": "attention",
            }
        )
        policy = UniCacheHookPolicy(
            build_plan_from_config(config), collect_diagnostics=False
        )
        policy.begin_run("prefill-passthrough")
        policy.set_context(phase="prefill", branch="understanding")
        torch.manual_seed(9)
        query = torch.randn(5, 4, 8, dtype=torch.bfloat16)
        key = torch.randn(7, 2, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        type_ids = torch.tensor([2, 2, 2, 2, 2, 5, 5])
        output = policy.apply_to_attention(
            q_i=query,
            k_i=key,
            v_i=value,
            attn_mask=None,
            key_type_ids=type_ids,
            protected_mask=type_ids == 5,
            layer_idx=0,
            mode="und",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=2,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            enable_gqa=True,
        ).squeeze(0).transpose(0, 1)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual({}, dict(policy.realized_storage_totals))

    def test_mixed_physical_attention_matches_explicit_reference(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(
                {
                    "task": {
                        "task_type": "editing",
                        "prompt": "edit",
                        "source_images": ["source.jpg"],
                        "output_modality": "image",
                    },
                    "planners": [{"id": "bagel_typed_segments", "enabled": True}],
                    "rules": [
                        {
                            "id": "identity_current",
                            "match": {"segment_id": "current_vae"},
                            "operator": "identity",
                            "stage": "attention",
                        }
                    ],
                    "runtime": {"plan_only": False, "fail_on_conflict": True},
                }
            )
        )
        policy.begin_run("mixed")
        policy.set_context(phase="denoise", branch="generation", step_index=1, total_steps=2)
        torch.manual_seed(4)
        query = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        dense_key = torch.randn(5, 4, 8, dtype=torch.bfloat16)
        dense_value = torch.randn_like(dense_key)
        source_key = torch.randn(12, 2, 8, dtype=torch.bfloat16)
        source_value = torch.randn_like(source_key)
        packed = PackedKIVICache.from_dense(
            source_key,
            source_value,
            logical_token_ids=torch.arange(12),
            token_type_id=3,
            k_bits=4,
            v_bits=4,
            group_size=4,
            residual_length=4,
            backend="reference",
        )
        output = policy.apply_to_attention(
            q_i=query,
            k_i=dense_key,
            v_i=dense_value,
            attn_mask=None,
            key_type_ids=torch.tensor([4] * 5),
            protected_mask=torch.zeros(5, dtype=torch.bool),
            layer_idx=0,
            mode="gen",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=2,
            physical_segments=[packed],
        )
        restored_key, restored_value = packed.dequantize(dtype=query.dtype)
        restored_key = restored_key.repeat_interleave(2, dim=1)
        restored_value = restored_value.repeat_interleave(2, dim=1)
        all_key = torch.cat([dense_key, restored_key], dim=0)
        all_value = torch.cat([dense_value, restored_value], dim=0)
        scores = torch.matmul(
            query.transpose(0, 1).float(),
            all_key.transpose(0, 1).float().transpose(-1, -2),
        ) / math.sqrt(query.shape[-1])
        expected = torch.matmul(
            torch.softmax(scores, dim=-1).to(all_value.dtype),
            all_value.transpose(0, 1),
        ).transpose(0, 1)
        self.assertTrue(torch.allclose(output, expected, atol=2e-2, rtol=2e-2))


if __name__ == "__main__":
    unittest.main()
