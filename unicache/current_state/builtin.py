"""BAGEL current-state policy registrations."""

from __future__ import annotations

from ..core.capabilities import CapabilityDescriptor
from ..core.decisions import CurrentStateDecision
from ..core.registry import CurrentStateOperator
from ..core.schema import RuntimeContext


class _DuCaBase(CurrentStateOperator):
    capabilities = CapabilityDescriptor(
        stages=frozenset({"denoise_step"}),
        phases=frozenset({"denoise", "plan"}),
        devices=frozenset({"cpu", "cuda", "mps", "*"}),
        requires=frozenset({"current_vae_state", "denoise_step", "cfg_branch"}),
        stateful=True,
        mutation_scope="current_state_features",
        physical_storage_mutation=False,
    )

    default_score_type = "attention"

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        score_type = str(out.get("score_type", self.default_score_type))
        if score_type not in {"attention", "random", "age_norm"}:
            raise ValueError(f"{self.name} score_type must be attention, random, or age_norm")
        for name in ("fresh_ratio", "soft_fresh_weight"):
            if name in out and float(out[name]) < 0:
                raise ValueError(f"{self.name} {name} must be non-negative")
        if int(out.get("fresh_threshold", 3)) <= 0:
            raise ValueError(f"{self.name} fresh_threshold must be positive")
        schedule_mode = str(
            out.get(
                "schedule_mode",
                "local_legacy" if self.name == "duca_age_norm_experimental" else "official_toca",
            )
        )
        if schedule_mode not in {"official_toca", "local_legacy"}:
            raise ValueError(f"{self.name} schedule_mode must be official_toca or local_legacy")
        return out

    def begin_run(self, params: dict, ctx: RuntimeContext, state: dict) -> CurrentStateDecision:
        resolved = {
            "fresh_ratio": float(params.get("fresh_ratio", 0.05)),
            "fresh_threshold": int(params.get("fresh_threshold", 3)),
            "soft_fresh_weight": float(params.get("soft_fresh_weight", 0.25)),
            "first_enhance": int(params.get("first_enhance", 1)),
            "min_fresh_tokens": int(params.get("min_fresh_tokens", 8)),
            "score_type": str(params.get("score_type", self.default_score_type)),
            "seed": int(params.get("seed", 0)),
            "schedule_mode": str(
                params.get(
                    "schedule_mode",
                    "local_legacy" if self.name == "duca_age_norm_experimental" else "official_toca",
                )
            ),
        }
        state.update({"operator": self.name, "params": resolved, "run_id": ctx.run_id})
        return CurrentStateDecision(operator=self.name, mode="duca", params=resolved)

    def generation_kwargs(self, decision: CurrentStateDecision) -> dict[str, object]:
        params = decision.params
        return {
            "enable_duca_current_state_cache": True,
            "enable_taylorseer": False,
            "duca_fresh_ratio": params["fresh_ratio"],
            "duca_fresh_threshold": params["fresh_threshold"],
            "duca_soft_fresh_weight": params["soft_fresh_weight"],
            "duca_first_enhance": params["first_enhance"],
            "duca_min_fresh_tokens": params["min_fresh_tokens"],
            "duca_score_type": params["score_type"],
            "duca_seed": params["seed"],
            "duca_schedule_mode": params["schedule_mode"],
        }


class DuCaCurrentStateOperator(_DuCaBase):
    name = "duca_current_state_reuse"
    default_score_type = "attention"


class DuCaAgeNormExperimentalOperator(_DuCaBase):
    name = "duca_age_norm_experimental"
    default_score_type = "age_norm"


class TaylorSeerCurrentStateOperator(CurrentStateOperator):
    name = "taylorseer_current_state_reuse"
    capabilities = CapabilityDescriptor(
        stages=frozenset({"denoise_step"}),
        phases=frozenset({"denoise", "plan"}),
        devices=frozenset({"cpu", "cuda", "mps", "*"}),
        requires=frozenset({"current_vae_state", "denoise_step", "cfg_branch"}),
        stateful=True,
        mutation_scope="current_state_features",
        physical_storage_mutation=False,
    )

    def begin_run(self, params: dict, ctx: RuntimeContext, state: dict) -> CurrentStateDecision:
        resolved = dict(params)
        state.update({"operator": self.name, "params": resolved, "run_id": ctx.run_id})
        return CurrentStateDecision(operator=self.name, mode="taylorseer", params=resolved)

    def generation_kwargs(self, decision: CurrentStateDecision) -> dict[str, object]:
        return {"enable_taylorseer": True, "enable_duca_current_state_cache": False}


def built_in_current_state_operators() -> list[CurrentStateOperator]:
    return [
        DuCaCurrentStateOperator(),
        DuCaAgeNormExperimentalOperator(),
        TaylorSeerCurrentStateOperator(),
    ]
