"""Conditioning-cache lifetime observation and retirement operators."""

from __future__ import annotations

import torch

from ..core.capabilities import CapabilityDescriptor
from ..core.decisions import SelectionDecision
from ..core.registry import OperatorExecutionContext, ProcessingOperator


def _policy_key(ctx: OperatorExecutionContext) -> tuple[object, ...]:
    policy_id = str(ctx.params.get("lifetime_policy_id", ctx.step.segment_id))
    return (
        "lifetime",
        policy_id,
        ctx.runtime.run_id,
        ctx.runtime.phase,
        ctx.runtime.branch,
        ctx.runtime.cfg_branch,
        int(ctx.runtime.layer_idx),
        int(ctx.runtime.batch_idx),
        ctx.step.segment_id,
    )


class LifetimeRetirementOperator(ProcessingOperator):
    name = "lifetime_retirement"
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

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        state = ctx.state.setdefault(_policy_key(ctx), {})
        offline_retired = bool(ctx.params.get("retired", False))
        if bool(state.get("pending_retire", False)) and not bool(state.get("retired", False)):
            state["retired"] = True
            state["effective_step"] = int(ctx.runtime.step_index or 0)
        retired = bool(state.get("retired", False) or offline_retired)
        if offline_retired and not bool(state.get("retired", False)):
            state["retired"] = True
            state["trigger_step"] = int(ctx.runtime.step_index or 0)
            state["effective_step"] = int(ctx.runtime.step_index or 0)
        if not retired:
            ctx.record({"lifetime_active_calls": 1})
            return ctx

        retired_mask = ctx.segment_mask & ~ctx.protected_mask
        keep_mask = torch.ones(int(ctx.k.shape[0]), dtype=torch.bool, device=ctx.k.device)
        keep_mask[retired_mask] = False
        head_keep = keep_mask.unsqueeze(0).expand(int(ctx.q.shape[1]), -1).clone()
        ctx.attention_keep_mask_override = head_keep
        ctx.decision = SelectionDecision(
            token_count=int(ctx.k.shape[0]),
            protected_mask=ctx.protected_mask & ctx.segment_mask,
            keep_mask=keep_mask,
            budget=int((keep_mask & ctx.segment_mask).sum().item()),
            reason="conditioning_lifetime_retirement",
            metadata={
                "retired_tokens": int(retired_mask.sum().item()),
                "protected_tokens": int((ctx.protected_mask & ctx.segment_mask).sum().item()),
                "trigger_step": state.get("trigger_step"),
                "effective_step": state.get("effective_step"),
                "monotone": True,
                "physical_eviction": True,
            },
        )
        ctx.record(
            {
                "lifetime_retired_calls": 1,
                "lifetime_retired_tokens": int(retired_mask.sum().item()),
                "lifetime_protected_tokens": int((ctx.protected_mask & ctx.segment_mask).sum().item()),
            }
        )
        return ctx


class AttentionMassLifetimeObserver(ProcessingOperator):
    name = "attention_mass_lifetime_observer"
    family = "observation"
    stages = {"post_attention"}
    executable = True
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        devices=frozenset({"cpu", "cuda", "mps"}),
        requires=frozenset({"segment_mask", "attention_probs"}),
        stateful=True,
        mutation_scope="none",
    )

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        for name in ("ema_decay", "threshold", "warmup_fraction"):
            value = float(out.get(name, {"ema_decay": 0.8, "threshold": 0.1, "warmup_fraction": 0.2}[name]))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.name} {name} must be in [0, 1]")
        if int(out.get("patience", 3)) <= 0:
            raise ValueError(f"{self.name} patience must be positive")
        return out

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        if ctx.attention_probs is None:
            raise ValueError(f"{self.name} requires attention probabilities")
        state = ctx.state.setdefault(_policy_key(ctx), {})
        if bool(state.get("retired", False)) or bool(state.get("pending_retire", False)):
            ctx.record({"lifetime_observer_inactive_calls": 1})
            return ctx

        mass = float(ctx.attention_probs[..., ctx.segment_mask].sum(dim=-1).mean().detach().cpu())
        decay = float(ctx.params.get("ema_decay", 0.8))
        previous = state.get("ema_mass")
        ema = mass if previous is None else decay * float(previous) + (1.0 - decay) * mass
        peak = max(float(state.get("peak_ema_mass", ema)), ema)
        ratio = ema / max(peak, 1e-12)
        state.update({"ema_mass": ema, "peak_ema_mass": peak, "mass_ratio": ratio})

        step = int(ctx.runtime.step_index or 0)
        total = max(1, int(ctx.runtime.total_steps or 1))
        warmup_step = int(round(float(ctx.params.get("warmup_fraction", 0.2)) * max(total - 1, 0)))
        below = step >= warmup_step and ratio <= float(ctx.params.get("threshold", 0.1))
        state["below_count"] = int(state.get("below_count", 0)) + 1 if below else 0
        if int(state["below_count"]) >= int(ctx.params.get("patience", 3)):
            state["pending_retire"] = True
            state["trigger_step"] = step

        ctx.record(
            {
                "lifetime_attention_mass": mass,
                "lifetime_ema_mass": ema,
                "lifetime_mass_ratio": ratio,
                "lifetime_below_count": int(state["below_count"]),
                "lifetime_pending_retire": int(bool(state.get("pending_retire", False))),
            }
        )
        return ctx
