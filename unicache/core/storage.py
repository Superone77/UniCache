"""Logical token identity and physical KV-cache layout contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class PhysicalCacheLayout:
    """Describe how a logical cache is represented by the current backend."""

    kind: str
    sequence_length: int
    head_dim: int
    num_query_heads: int
    num_kv_heads: int
    materialized: bool = False
    packed: bool = False
    head_lengths: tuple[int, ...] = ()
    cumulative_lengths: tuple[int, ...] = ()
    dtype: str = "unknown"

    def __post_init__(self) -> None:
        if int(self.sequence_length) < 0:
            raise ValueError("PhysicalCacheLayout.sequence_length must be non-negative")
        for name in ("head_dim", "num_query_heads", "num_kv_heads"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"PhysicalCacheLayout.{name} must be positive")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "PhysicalCacheLayout requires num_query_heads to be divisible by num_kv_heads"
            )
        if self.head_lengths and len(self.head_lengths) != self.num_kv_heads:
            raise ValueError("PhysicalCacheLayout.head_lengths must contain one entry per KV head")

    @property
    def gqa_repeat(self) -> int:
        return self.num_query_heads // self.num_kv_heads


@dataclass
class CacheStorageView:
    """Bind logical token identity to a backend's physical cache metadata.

    The current BAGEL adapter receives repeated tensors used for attention, so
    ``layout.materialized`` is false. ``num_kv_heads`` still records the
    unexpanded storage shape and keeps storage accounting honest.
    """

    key: torch.Tensor
    value: torch.Tensor
    layout: PhysicalCacheLayout
    logical_token_ids: torch.Tensor
    key_type_ids: torch.Tensor | None = None
    quantization_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.key.shape != self.value.shape:
            raise ValueError("CacheStorageView key and value shapes must match")
        if self.key.dim() != 3:
            raise ValueError("CacheStorageView expects [tokens, heads, head_dim] tensors")
        token_count = int(self.key.shape[0])
        if self.logical_token_ids.dim() != 1 or int(self.logical_token_ids.numel()) != token_count:
            raise ValueError("CacheStorageView logical_token_ids must match the token count")
        if self.key_type_ids is not None and (
            self.key_type_ids.dim() != 1 or int(self.key_type_ids.numel()) != token_count
        ):
            raise ValueError("CacheStorageView key_type_ids must match the token count")
        if self.layout.sequence_length != token_count or self.layout.head_dim != int(self.key.shape[-1]):
            raise ValueError("CacheStorageView tensor shape does not match its physical layout")

    @classmethod
    def from_attention_tensors(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        num_query_heads: int | None = None,
        num_kv_heads: int | None = None,
        key_type_ids: torch.Tensor | None = None,
        logical_token_ids: torch.Tensor | None = None,
        physical_kind: str = "attention_view",
        materialized: bool = False,
    ) -> "CacheStorageView":
        if key.dim() != 3:
            raise ValueError("Attention KV tensors must have shape [tokens, heads, head_dim]")
        attention_heads = int(num_query_heads or key.shape[1])
        storage_heads = int(num_kv_heads or key.shape[1])
        token_count = int(key.shape[0])
        ids = logical_token_ids
        if ids is None:
            ids = torch.arange(token_count, device=key.device, dtype=torch.long)
        cumulative_lengths = (
            tuple([0] * (storage_heads + 1))
            if token_count == 0
            else tuple(range(0, (storage_heads + 1) * token_count, token_count))
        )
        layout = PhysicalCacheLayout(
            kind=physical_kind,
            sequence_length=token_count,
            head_dim=int(key.shape[-1]),
            num_query_heads=attention_heads,
            num_kv_heads=storage_heads,
            materialized=materialized,
            packed=False,
            head_lengths=tuple([token_count] * storage_heads),
            cumulative_lengths=cumulative_lengths,
            dtype=str(key.dtype).removeprefix("torch."),
        )
        return cls(
            key=key,
            value=value,
            layout=layout,
            logical_token_ids=ids,
            key_type_ids=key_type_ids,
        )

    @property
    def token_count(self) -> int:
        return int(self.key.shape[0])

    @property
    def original_storage_bytes(self) -> int:
        return (
            self.token_count
            * self.layout.num_kv_heads
            * self.layout.head_dim
            * 2
            * self.key.element_size()
        )
