"""Operator decisions that are independent from physical cache mutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SelectionDecision:
    """Describe token protection or retention without mutating cache storage."""

    token_count: int
    protected_mask: torch.Tensor | None = None
    keep_mask: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    budget: int | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("SelectionDecision token count must be non-negative")
        for name in ("protected_mask", "keep_mask"):
            mask = getattr(self, name)
            if mask is None:
                continue
            if mask.dtype != torch.bool or mask.dim() != 1 or int(mask.numel()) != self.token_count:
                raise ValueError(f"SelectionDecision {name} must be a boolean mask matching token count")
        if self.budget is not None and not 0 <= int(self.budget) <= self.token_count:
            raise ValueError("SelectionDecision budget must be within the token count")

    @property
    def action(self) -> str:
        if self.keep_mask is not None:
            return "keep"
        if self.protected_mask is not None:
            return "protect"
        return "observe"

    def apply_protection(self, base_mask: torch.Tensor) -> torch.Tensor:
        if base_mask.dtype != torch.bool or base_mask.dim() != 1 or int(base_mask.numel()) != self.token_count:
            raise ValueError("Base protection mask must match SelectionDecision token count")
        if self.protected_mask is None:
            return base_mask.clone()
        return base_mask | self.protected_mask.to(device=base_mask.device)


@dataclass
class PhysicalCacheRequest:
    """Request a one-way conversion from dense KV to a physical layout.

    Operators create requests from an attention observation. The BAGEL storage
    adapter consumes them only when it owns the native four-head KV tensors.
    This avoids implementing GQA compaction on the repeated 28-head attention
    view and makes a physical configuration fail explicitly if it cannot be
    realized.
    """

    operator: str
    segment_id: str
    token_count: int
    segment_mask: torch.Tensor
    params: dict[str, Any] = field(default_factory=dict)
    per_kv_head_scores: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("PhysicalCacheRequest token count must be non-negative")
        if (
            self.segment_mask.dtype != torch.bool
            or self.segment_mask.dim() != 1
            or int(self.segment_mask.numel()) != self.token_count
        ):
            raise ValueError(
                "PhysicalCacheRequest segment_mask must be boolean and match token_count"
            )
        if self.per_kv_head_scores is not None:
            if self.per_kv_head_scores.dim() != 2:
                raise ValueError("Physical per-KV-head scores must be [kv_heads, tokens]")
            if int(self.per_kv_head_scores.shape[-1]) != self.token_count:
                raise ValueError("Physical per-KV-head scores must match token_count")


@dataclass
class CurrentStateDecision:
    """Describe one current-state feature-cache policy invocation."""

    operator: str
    mode: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
