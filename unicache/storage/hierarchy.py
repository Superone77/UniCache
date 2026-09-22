"""Logical host archive and GPU working-set accounting.

This module models the algorithmic hierarchy without moving tensor payloads.
The BAGEL attention view remains full, while the archive records which KV
head/token entries would be resident on GPU and which entries would be
restored from Host at the next working-set switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class WorkingSetTransition:
    logical_retained_tokens: int
    logical_restored_tokens: int
    logical_evicted_tokens: int
    logical_resident_tokens: int
    kv_head_retained_tokens: int
    kv_head_restored_tokens: int
    kv_head_evicted_tokens: int
    kv_head_resident_tokens: int
    resident_bytes: int
    h2d_bytes: int
    released_bytes: int
    resident_ratio: float


@dataclass(frozen=True)
class QuantizedWorkingSetTransition:
    resident_bytes: int
    h2d_bytes: int
    released_bytes: int
    resident_ratio: float
    transition_kind: str
    descriptor: tuple[object, ...]


@dataclass
class LogicalQuantizedHostArchive:
    """Track full Host KV plus one all-token quantized GPU replica."""

    full_archive_bytes: int
    _current_descriptor: tuple[object, ...] | None = field(
        default=None, init=False, repr=False
    )
    _current_resident_bytes: int = field(default=0, init=False, repr=False)
    _replica_catalog: dict[tuple[object, ...], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _resident_byte_invocations: int = field(default=0, init=False, repr=False)
    _archive_byte_invocations: int = field(default=0, init=False, repr=False)
    _peak_resident_ratio: float = field(default=0.0, init=False, repr=False)
    _materializations: int = field(default=0, init=False, repr=False)
    _precision_transitions: int = field(default=0, init=False, repr=False)
    _cumulative_h2d_bytes: int = field(default=0, init=False, repr=False)
    _cumulative_full_rebuild_h2d_bytes: int = field(
        default=0, init=False, repr=False
    )
    _cumulative_released_bytes: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.full_archive_bytes) <= 0:
            raise ValueError("full_archive_bytes must be positive")
        self.full_archive_bytes = int(self.full_archive_bytes)

    def materialize(
        self,
        *,
        replica_bytes: int,
        descriptor: tuple[object, ...],
    ) -> QuantizedWorkingSetTransition:
        resident_bytes = max(0, int(replica_bytes))
        descriptor = tuple(descriptor)
        previous_descriptor = self._current_descriptor
        previous_bytes = self._current_resident_bytes
        changed = previous_descriptor != descriptor

        if previous_descriptor is None:
            transition_kind = "initial"
            h2d_bytes = 0
            released_bytes = 0
        elif not changed:
            transition_kind = "reuse"
            h2d_bytes = 0
            released_bytes = 0
        else:
            transition_kind = (
                "upgrade"
                if resident_bytes > previous_bytes
                else "downgrade"
                if resident_bytes < previous_bytes
                else "repack"
            )
            # Phase 1 assumes the requested packed replica is prepared in Host
            # memory. Switching precision transfers that replica and releases
            # the old GPU payload at the buffer boundary.
            h2d_bytes = resident_bytes
            released_bytes = previous_bytes

        if previous_descriptor is not None:
            self._cumulative_full_rebuild_h2d_bytes += resident_bytes
        if changed:
            self._precision_transitions += 1
        self._replica_catalog[descriptor] = resident_bytes
        self._current_descriptor = descriptor
        self._current_resident_bytes = resident_bytes
        self._materializations += 1
        self._resident_byte_invocations += resident_bytes
        self._archive_byte_invocations += self.full_archive_bytes
        resident_ratio = float(resident_bytes / self.full_archive_bytes)
        self._peak_resident_ratio = max(self._peak_resident_ratio, resident_ratio)
        self._cumulative_h2d_bytes += h2d_bytes
        self._cumulative_released_bytes += released_bytes
        return QuantizedWorkingSetTransition(
            resident_bytes=resident_bytes,
            h2d_bytes=h2d_bytes,
            released_bytes=released_bytes,
            resident_ratio=resident_ratio,
            transition_kind=transition_kind,
            descriptor=descriptor,
        )

    def summary(self) -> dict[str, int | float | str | list[object]]:
        readable_descriptor = [
            f"mask-bytes:{len(item)}" if isinstance(item, bytes) else item
            for item in (self._current_descriptor or ())
        ]
        return {
            "archive_bytes": self.full_archive_bytes,
            "current_resident_bytes": self._current_resident_bytes,
            "resident_byte_invocations": self._resident_byte_invocations,
            "archive_byte_invocations": self._archive_byte_invocations,
            "average_resident_ratio": float(
                self._resident_byte_invocations
                / max(self._archive_byte_invocations, 1)
            ),
            "peak_resident_ratio": self._peak_resident_ratio,
            "materializations": self._materializations,
            "precision_transitions": self._precision_transitions,
            "current_descriptor": readable_descriptor,
            "host_replica_catalog_bytes": int(sum(self._replica_catalog.values())),
            "host_replica_variants": len(self._replica_catalog),
            "cumulative_h2d_bytes": self._cumulative_h2d_bytes,
            "cumulative_full_rebuild_h2d_bytes": (
                self._cumulative_full_rebuild_h2d_bytes
            ),
            "cumulative_released_bytes": self._cumulative_released_bytes,
            "cumulative_logical_restored_tokens": 0,
            "cumulative_logical_evicted_tokens": 0,
            "cumulative_kv_head_restored_tokens": 0,
            "cumulative_kv_head_evicted_tokens": 0,
        }


@dataclass
class LogicalHostArchive:
    """Track a recoverable GPU working set backed by a complete Host archive."""

    token_mask: torch.Tensor
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    element_size: int
    _resident_kv_mask: torch.Tensor = field(init=False, repr=False)
    _resident_logical_mask: torch.Tensor = field(init=False, repr=False)
    _resident_byte_invocations: int = field(default=0, init=False, repr=False)
    _archive_byte_invocations: int = field(default=0, init=False, repr=False)
    _peak_resident_ratio: float = field(default=0.0, init=False, repr=False)
    _transitions: int = field(default=0, init=False, repr=False)
    _cumulative_h2d_bytes: int = field(default=0, init=False, repr=False)
    _cumulative_full_rebuild_h2d_bytes: int = field(default=0, init=False, repr=False)
    _cumulative_released_bytes: int = field(default=0, init=False, repr=False)
    _cumulative_logical_restored_tokens: int = field(default=0, init=False, repr=False)
    _cumulative_logical_evicted_tokens: int = field(default=0, init=False, repr=False)
    _cumulative_kv_head_restored_tokens: int = field(default=0, init=False, repr=False)
    _cumulative_kv_head_evicted_tokens: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        mask = self.token_mask.detach().to(device="cpu", dtype=torch.bool)
        if mask.dim() != 1:
            raise ValueError("LogicalHostArchive.token_mask must be one-dimensional")
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("LogicalHostArchive head counts must be positive")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("LogicalHostArchive requires query heads divisible by KV heads")
        if self.head_dim <= 0 or self.element_size <= 0:
            raise ValueError("LogicalHostArchive head_dim and element_size must be positive")
        self.token_mask = mask
        self._resident_logical_mask = mask.clone()
        self._resident_kv_mask = mask.unsqueeze(0).expand(self.num_kv_heads, -1).clone()

    @property
    def token_count(self) -> int:
        return int(self.token_mask.sum().item())

    @property
    def bytes_per_kv_head_token(self) -> int:
        return int(self.head_dim * 2 * self.element_size)

    @property
    def archive_bytes(self) -> int:
        return int(self.token_count * self.num_kv_heads * self.bytes_per_kv_head_token)

    def _align_token_axis(self, token_mask: torch.Tensor) -> None:
        current = token_mask.detach().to(device="cpu", dtype=torch.bool)
        if current.dim() != 1:
            raise ValueError("LogicalHostArchive token_mask must be one-dimensional")
        old_length = int(self.token_mask.numel())
        new_length = int(current.numel())
        if new_length < old_length:
            raise ValueError("LogicalHostArchive does not support shrinking the global token axis")
        if not torch.equal(current[:old_length], self.token_mask):
            raise ValueError("LogicalHostArchive token identity changed within the existing prefix")
        if new_length == old_length:
            return
        appended = current[old_length:]
        self.token_mask = current
        self._resident_logical_mask = torch.cat(
            (self._resident_logical_mask, appended.clone()), dim=0
        )
        self._resident_kv_mask = torch.cat(
            (
                self._resident_kv_mask,
                appended.unsqueeze(0).expand(self.num_kv_heads, -1).clone(),
            ),
            dim=1,
        )

    def _to_kv_head_mask(self, query_head_keep_mask: torch.Tensor) -> torch.Tensor:
        keep = query_head_keep_mask.detach().to(device="cpu", dtype=torch.bool)
        expected = (self.num_query_heads, int(self.token_mask.numel()))
        if tuple(keep.shape) != expected:
            raise ValueError(
                f"Working-set mask shape {tuple(keep.shape)} does not match {expected}"
            )
        keep &= self.token_mask.unsqueeze(0)
        repeat = self.num_query_heads // self.num_kv_heads
        return keep.reshape(self.num_kv_heads, repeat, -1).any(dim=1)

    def materialize(
        self,
        query_head_keep_mask: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> WorkingSetTransition:
        if token_mask is not None:
            self._align_token_axis(token_mask)
        next_kv = self._to_kv_head_mask(query_head_keep_mask)
        next_logical = next_kv.any(dim=0) & self.token_mask
        previous_kv = self._resident_kv_mask
        previous_logical = self._resident_logical_mask

        retained_kv = previous_kv & next_kv
        restored_kv = ~previous_kv & next_kv
        evicted_kv = previous_kv & ~next_kv
        retained_logical = previous_logical & next_logical
        restored_logical = ~previous_logical & next_logical
        evicted_logical = previous_logical & ~next_logical

        resident_kv_tokens = int(next_kv.sum().item())
        restored_kv_tokens = int(restored_kv.sum().item())
        evicted_kv_tokens = int(evicted_kv.sum().item())
        payload = self.bytes_per_kv_head_token
        resident_bytes = resident_kv_tokens * payload
        h2d_bytes = restored_kv_tokens * payload
        released_bytes = evicted_kv_tokens * payload
        resident_ratio = float(resident_bytes / max(self.archive_bytes, 1))

        # The initial working set is carved directly from full prefill KV that
        # is already on GPU. Later full-rebuild baselines would reload the
        # entire next resident set from Host at every transition.
        if self._transitions > 0:
            self._cumulative_full_rebuild_h2d_bytes += resident_bytes

        self._resident_kv_mask = next_kv
        self._resident_logical_mask = next_logical
        self._transitions += 1
        self._resident_byte_invocations += resident_bytes
        self._archive_byte_invocations += self.archive_bytes
        self._peak_resident_ratio = max(self._peak_resident_ratio, resident_ratio)
        self._cumulative_h2d_bytes += h2d_bytes
        self._cumulative_released_bytes += released_bytes
        self._cumulative_logical_restored_tokens += int(restored_logical.sum().item())
        self._cumulative_logical_evicted_tokens += int(evicted_logical.sum().item())
        self._cumulative_kv_head_restored_tokens += restored_kv_tokens
        self._cumulative_kv_head_evicted_tokens += evicted_kv_tokens

        return WorkingSetTransition(
            logical_retained_tokens=int(retained_logical.sum().item()),
            logical_restored_tokens=int(restored_logical.sum().item()),
            logical_evicted_tokens=int(evicted_logical.sum().item()),
            logical_resident_tokens=int(next_logical.sum().item()),
            kv_head_retained_tokens=int(retained_kv.sum().item()),
            kv_head_restored_tokens=restored_kv_tokens,
            kv_head_evicted_tokens=evicted_kv_tokens,
            kv_head_resident_tokens=resident_kv_tokens,
            resident_bytes=resident_bytes,
            h2d_bytes=h2d_bytes,
            released_bytes=released_bytes,
            resident_ratio=resident_ratio,
        )

    def summary(self) -> dict[str, int | float]:
        return {
            "archive_tokens": self.token_count,
            "archive_bytes": self.archive_bytes,
            "transitions": self._transitions,
            "current_logical_resident_tokens": int(self._resident_logical_mask.sum().item()),
            "current_kv_head_resident_tokens": int(self._resident_kv_mask.sum().item()),
            "current_resident_bytes": int(
                self._resident_kv_mask.sum().item() * self.bytes_per_kv_head_token
            ),
            "average_resident_ratio": float(
                self._resident_byte_invocations
                / max(self._archive_byte_invocations, 1)
            ),
            "peak_resident_ratio": self._peak_resident_ratio,
            "resident_byte_invocations": self._resident_byte_invocations,
            "archive_byte_invocations": self._archive_byte_invocations,
            "cumulative_h2d_bytes": self._cumulative_h2d_bytes,
            "cumulative_full_rebuild_h2d_bytes": (
                self._cumulative_full_rebuild_h2d_bytes
            ),
            "differential_transfer_savings_ratio": float(
                1.0
                - self._cumulative_h2d_bytes
                / max(self._cumulative_full_rebuild_h2d_bytes, 1)
            )
            if self._cumulative_full_rebuild_h2d_bytes
            else 0.0,
            "cumulative_released_bytes": self._cumulative_released_bytes,
            "cumulative_logical_restored_tokens": self._cumulative_logical_restored_tokens,
            "cumulative_logical_evicted_tokens": self._cumulative_logical_evicted_tokens,
            "cumulative_kv_head_restored_tokens": self._cumulative_kv_head_restored_tokens,
            "cumulative_kv_head_evicted_tokens": self._cumulative_kv_head_evicted_tokens,
        }
