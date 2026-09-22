"""DuCa-style current-state feature cache utilities for BAGEL inference.

This is a minimal adaptation of DuCa/ToCa's diffusion-transformer feature
caching idea to BAGEL's generation branch.  It targets the active current VAE
state during denoising, not conditioning KV eviction.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def duca_cache_init(
    num_steps: int,
    *,
    fresh_ratio: float = 0.05,
    fresh_threshold: int = 3,
    soft_fresh_weight: float = 0.25,
    first_enhance: int = 1,
    min_fresh_tokens: int = 8,
    score_type: str = "age_norm",
    seed: int = 0,
    schedule_mode: str = "local_legacy",
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_dic: dict[str, Any] = {
        "fresh_ratio": float(fresh_ratio),
        "fresh_threshold": int(fresh_threshold),
        "soft_fresh_weight": float(soft_fresh_weight),
        "first_enhance": int(first_enhance),
        "min_fresh_tokens": int(min_fresh_tokens),
        "score_type": score_type,
        "seed": int(seed),
        "schedule_mode": str(schedule_mode),
        "layers": {},
        "stats": [],
    }
    current: dict[str, Any] = {
        "step": 0,
        "num_steps": int(num_steps),
        "type": "full",
        "layer": 0,
    }
    return cache_dic, current


def duca_prepare_step(cache_dic: dict[str, Any], current: dict[str, Any]) -> None:
    step = int(current.get("step", 0))
    num_steps = max(int(current.get("num_steps", 1)), 1)
    schedule_step = (
        max(num_steps - 1 - step, 0)
        if cache_dic.get("schedule_mode") == "official_toca"
        else step
    )
    threshold = max(int(cache_dic.get("fresh_threshold", 3)), 1)

    if step < int(cache_dic.get("first_enhance", 1)):
        current["type"] = "full"
        cache_dic["cal_threshold"] = threshold
        return

    # DuCa/ToCa reports the 20%-40% region as cache-sensitive.  Refresh more
    # often there; otherwise use a mild force-activation scheduler.
    if int(num_steps * 0.2) <= schedule_step < int(num_steps * 0.4):
        cal_threshold = min(threshold, 2)
    else:
        linear_step_weight = 0.4
        step_factor = 1 + linear_step_weight - 2 * linear_step_weight * schedule_step / num_steps
        cal_threshold = int(torch.round(torch.tensor(threshold / max(step_factor, 1e-6))).item())
        cal_threshold = max(cal_threshold, 1)
    cache_dic["cal_threshold"] = cal_threshold

    if schedule_step % cal_threshold == 0:
        current["type"] = "full"
    elif (schedule_step % cal_threshold) % 2 == 1:
        current["type"] = "ToCa"
    else:
        current["type"] = "aggressive"


def duca_fresh_ratio(cache_dic: dict[str, Any], current: dict[str, Any]) -> float:
    step = float(current.get("step", 0))
    num_steps = max(float(current.get("num_steps", 1)), 1.0)
    schedule_step = (
        max(num_steps - 1.0 - step, 0.0)
        if cache_dic.get("schedule_mode") == "official_toca"
        else step
    )
    layer = float(current.get("layer", 0))
    num_layers = max(float(current.get("num_layers", 28)), 1.0)

    # Same qualitative shape as DuCa/ToCa: more fresh computation early, and a
    # mild layer factor.  BAGEL layer sensitivity differs, so keep this modest.
    step_weight = 2.0
    step_factor = 1 + step_weight - 2 * step_weight * schedule_step / num_steps
    layer_weight = -0.2
    layer_factor = 1 + layer_weight - 2 * layer_weight * layer / max(num_layers - 1, 1.0)
    ratio = float(cache_dic.get("fresh_ratio", 0.05)) * step_factor * layer_factor
    return float(max(0.0, min(1.0, ratio)))


def duca_store_full(
    cache_dic: dict[str, Any],
    layer: int,
    *,
    attn_output: torch.Tensor,
    mlp_output: torch.Tensor,
    vae_token_count: int,
    attention_score: torch.Tensor | None = None,
) -> None:
    cache_dic["layers"][layer] = {
        "attn": attn_output.detach(),
        "mlp": mlp_output.detach(),
        "age": torch.zeros(vae_token_count, dtype=torch.float32, device=mlp_output.device),
        "attention_score": attention_score.detach() if attention_score is not None else None,
    }


def duca_layer_ready(cache_dic: dict[str, Any], layer: int, seq_len: int, vae_token_count: int) -> bool:
    layer_cache = cache_dic.get("layers", {}).get(layer)
    if not layer_cache:
        return False
    attn = layer_cache.get("attn")
    mlp = layer_cache.get("mlp")
    age = layer_cache.get("age")
    return (
        isinstance(attn, torch.Tensor)
        and isinstance(mlp, torch.Tensor)
        and isinstance(age, torch.Tensor)
        and attn.shape[0] == seq_len
        and mlp.shape[0] == seq_len
        and age.numel() == vae_token_count
    )


def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    if score.numel() == 0:
        return score
    score = score.to(torch.float32)
    denom = score.max() - score.min()
    if float(denom.detach().cpu()) < 1e-8:
        return torch.zeros_like(score)
    return (score - score.min()) / denom


def duca_select_fresh(
    cache_dic: dict[str, Any],
    current: dict[str, Any],
    *,
    layer: int,
    vae_states: torch.Tensor,
) -> torch.Tensor:
    layer_cache = cache_dic["layers"][layer]
    age = layer_cache["age"].to(device=vae_states.device)
    ratio = duca_fresh_ratio(cache_dic, current)
    n_tokens = vae_states.shape[0]
    if n_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=vae_states.device)
    keep = int(math.ceil(n_tokens * ratio))
    keep = max(int(cache_dic.get("min_fresh_tokens", 8)), keep)
    keep = min(n_tokens, keep)

    score_type = str(cache_dic.get("score_type", "age_norm"))
    score = torch.zeros(n_tokens, dtype=torch.float32, device=vae_states.device)
    if score_type == "attention":
        attention_score = layer_cache.get("attention_score")
        if isinstance(attention_score, torch.Tensor) and attention_score.numel() == n_tokens:
            score = attention_score.to(device=vae_states.device, dtype=torch.float32)
    elif score_type == "random":
        # CPU generator is deterministic on CUDA and MPS as well; move only
        # the small score vector to the active device.
        generator = torch.Generator(device="cpu")
        seed = int(cache_dic.get("seed", 0))
        seed += 1009 * int(current.get("step", 0)) + int(layer)
        generator.manual_seed(seed)
        score = torch.rand(n_tokens, generator=generator).to(device=vae_states.device)
    if "age" in score_type:
        score = score + float(cache_dic.get("soft_fresh_weight", 0.25)) * _normalize_score(age)
    if "norm" in score_type:
        score = score + _normalize_score(vae_states.detach().to(torch.float32).norm(dim=-1))
    if float(score.abs().sum().detach().cpu()) < 1e-8:
        score = torch.arange(n_tokens, device=vae_states.device, dtype=torch.float32)

    return torch.topk(score, k=keep, largest=True).indices


def duca_update_age(cache_dic: dict[str, Any], layer: int, fresh_local_indices: torch.Tensor) -> None:
    age = cache_dic["layers"][layer]["age"].to(device=fresh_local_indices.device)
    age = age + 1
    if fresh_local_indices.numel() > 0:
        age[fresh_local_indices] = 0
    cache_dic["layers"][layer]["age"] = age


def duca_record(
    cache_dic: dict[str, Any],
    current: dict[str, Any],
    *,
    layer: int,
    vae_token_count: int,
    fresh_count: int,
) -> None:
    cache_dic["stats"].append(
        {
            "step": int(current.get("step", 0)),
            "layer": int(layer),
            "type": str(current.get("type", "unknown")),
            "fresh_ratio_target": float(duca_fresh_ratio(cache_dic, current)),
            "vae_tokens": int(vae_token_count),
            "fresh_tokens": int(fresh_count),
            "stale_tokens": int(max(vae_token_count - fresh_count, 0)),
            "schedule_mode": str(cache_dic.get("schedule_mode", "local_legacy")),
        }
    )
