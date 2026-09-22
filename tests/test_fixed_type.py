"""Contract tests for the fixed cache-type operator assignment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from unicache.adapters import UniCacheHookPolicy
from unicache.runtime import build_plan_from_config


CONFIG_ROOT = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


class FixedTypePolicyTests(unittest.TestCase):
    def test_understanding_uses_h2o_for_instruction_and_source_vit(self):
        bundle = build_plan_from_config(_load("fixed_type_understanding.json"))
        operators = {
            step.segment_id: step.operator
            for step in bundle.execution_plan.steps
            if step.operator == "h2o_segment_attention_mask"
        }
        self.assertEqual(
            {
                "instruction": "h2o_segment_attention_mask",
                "source_vit": "h2o_segment_attention_mask",
            },
            operators,
        )
        self.assertEqual([], bundle.execution_plan.summaries["routing_decisions"])

    def test_editing_uses_fixed_h2o_h2o_kivi_mapping(self):
        bundle = build_plan_from_config(_load("fixed_type_editing.json"))
        managed = {
            step.segment_id: step.operator
            for step in bundle.execution_plan.steps
            if step.operator
            in {"h2o_segment_attention_mask", "kivi_quantization"}
        }
        self.assertEqual(
            {
                "instruction": "h2o_segment_attention_mask",
                "source_vit": "h2o_segment_attention_mask",
                "source_vae": "kivi_quantization",
            },
            managed,
        )
        for step in bundle.execution_plan.steps:
            if step.operator == "h2o_segment_attention_mask":
                self.assertEqual(
                    "pre_eviction_attention", step.params["score_source"]
                )
            if step.operator in {
                "h2o_segment_attention_mask",
                "kivi_quantization",
            }:
                self.assertEqual(
                    ["instruction", "source_vit", "source_vae"],
                    step.scheduler.params["budget_segments"],
                )
            if step.operator == "block_attention_metrics":
                self.assertEqual(
                    "pre_eviction_attention", step.params["score_source"]
                )

    def test_generation_only_budgets_existing_instruction_conditioning(self):
        bundle = build_plan_from_config(_load("fixed_type_generation.json"))
        managed = [
            step
            for step in bundle.execution_plan.steps
            if step.operator
            in {"h2o_segment_attention_mask", "kivi_quantization"}
        ]
        self.assertEqual(1, len(managed))
        self.assertEqual("instruction", managed[0].segment_id)
        self.assertEqual(["instruction"], managed[0].scheduler.params["budget_segments"])

    def test_single_route_uses_one_qk_for_pre_eviction_stats_and_output(self):
        policy = UniCacheHookPolicy(
            build_plan_from_config(_load("fixed_type_editing.json"))
        )
        policy.begin_run("fixed-type-single-route")
        policy.set_context(
            phase="denoise", branch="generation", step_index=0, total_steps=10
        )
        torch.manual_seed(23)
        key_types = torch.tensor(
            [1] * 8 + [2] * 2 + [3] * 16 + [4] * 16 + [5] * 8
        )
        q = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        k = torch.randn(len(key_types), 2, 8, dtype=torch.bfloat16)
        v = torch.randn_like(k)

        attention_score_calls = 0
        original_attention_scores = policy._attention_scores

        def counted_attention_scores(*args, **kwargs):
            nonlocal attention_score_calls
            attention_score_calls += 1
            return original_attention_scores(*args, **kwargs)

        policy._attention_scores = counted_attention_scores
        output = policy.apply_to_attention(
            q_i=q,
            k_i=k,
            v_i=v,
            attn_mask=None,
            key_type_ids=key_types,
            protected_mask=torch.zeros(len(key_types), dtype=torch.bool),
            layer_idx=0,
            mode="gen",
            sample_idx=0,
            causal=False,
            storage_num_kv_heads=1,
        )

        self.assertEqual((4, 2, 8), tuple(output.shape))
        self.assertEqual(1, attention_score_calls)
        self.assertEqual(2, len(policy.h2o_states))
        self.assertTrue(policy.block_metric_states)
        for state in policy.h2o_states.values():
            self.assertEqual("pre_eviction_attention", state["score_source"])


if __name__ == "__main__":
    unittest.main()
