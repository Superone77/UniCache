"""Physical cache layouts used by the UniCache efficiency MVE.

The classes in this module own the tensors counted as resident cache storage.
They deliberately keep logical token identity separate from physical layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else int(tensor.numel() * tensor.element_size())


def gqa_query_key_logits(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute QK for native GQA tensors without materializing repeated KV heads."""

    if query.dim() != 3 or key.dim() != 3:
        raise ValueError("GQA QK expects [tokens, heads, head_dim] tensors")
    if int(query.shape[-1]) != int(key.shape[-1]):
        raise ValueError("GQA QK query/key head dimensions must match")
    query_heads = int(query.shape[1])
    kv_heads = int(key.shape[1])
    if query_heads % kv_heads != 0:
        raise ValueError("GQA query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    grouped_query = query.transpose(0, 1).reshape(
        kv_heads, groups, int(query.shape[0]), int(query.shape[-1])
    )
    grouped_key = key.transpose(0, 1).unsqueeze(1)
    logits = torch.matmul(
        grouped_query.float(), grouped_key.float().transpose(-1, -2)
    )
    return logits.reshape(query_heads, int(query.shape[0]), int(key.shape[0]))


def gqa_probability_value(
    probabilities: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Compute AV for native GQA tensors without materializing repeated KV heads."""

    if probabilities.dim() != 3 or value.dim() != 3:
        raise ValueError("GQA AV expects probabilities [heads, queries, keys] and values [keys, heads, dim]")
    query_heads = int(probabilities.shape[0])
    kv_heads = int(value.shape[1])
    if query_heads % kv_heads != 0:
        raise ValueError("GQA probability heads must be divisible by KV heads")
    if int(probabilities.shape[-1]) != int(value.shape[0]):
        raise ValueError("GQA AV probability key width must match value tokens")
    groups = query_heads // kv_heads
    grouped_probs = probabilities.reshape(
        kv_heads, groups, int(probabilities.shape[1]), int(probabilities.shape[2])
    )
    grouped_value = value.transpose(0, 1).unsqueeze(1)
    output = torch.matmul(grouped_probs.to(value.dtype), grouped_value)
    return output.reshape(
        query_heads, int(probabilities.shape[1]), int(value.shape[-1])
    )


def _pack_codes(codes: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    if bits == 8:
        return codes.to(torch.uint8)
    if bits not in {2, 4}:
        raise ValueError(f"Packed KIVI storage supports 2, 4, or 8 bits, got {bits}")
    dim = dim if dim >= 0 else codes.dim() + dim
    factor = 32 // bits
    length = int(codes.shape[dim])
    padded = int(math.ceil(length / factor) * factor)
    if padded != length:
        shape = list(codes.shape)
        shape[dim] = padded - length
        codes = torch.cat(
            [codes, torch.zeros(shape, dtype=codes.dtype, device=codes.device)], dim=dim
        )
    moved = codes.movedim(dim, -1).to(torch.int32)
    moved = moved.reshape(*moved.shape[:-1], -1, factor)
    shifts = torch.arange(factor, device=codes.device, dtype=torch.int32) * bits
    packed = torch.sum(moved << shifts, dim=-1, dtype=torch.int32)
    return packed.movedim(-1, dim).contiguous()


def _unpack_codes(packed: torch.Tensor, bits: int, dim: int, length: int) -> torch.Tensor:
    if bits == 8:
        index = [slice(None)] * packed.dim()
        index[dim] = slice(0, length)
        return packed[tuple(index)].to(torch.uint8)
    if bits not in {2, 4}:
        raise ValueError(f"Packed KIVI storage supports 2, 4, or 8 bits, got {bits}")
    dim = dim if dim >= 0 else packed.dim() + dim
    factor = 32 // bits
    moved = packed.movedim(dim, -1).to(torch.int32)
    shifts = torch.arange(factor, device=packed.device, dtype=torch.int32) * bits
    mask = (1 << bits) - 1
    unpacked = ((moved.unsqueeze(-1) >> shifts) & mask).reshape(
        *moved.shape[:-1], -1
    )
    unpacked = unpacked[..., :length]
    return unpacked.movedim(-1, dim).to(torch.uint8).contiguous()


def _quantize_grouped_reference(
    data: torch.Tensor,
    *,
    bits: int,
    group_dim: int,
    group_size: int,
    pack_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    group_dim = group_dim if group_dim >= 0 else data.dim() + group_dim
    length = int(data.shape[group_dim])
    groups = []
    scales = []
    minima = []
    max_code = float((1 << bits) - 1)
    eps = torch.finfo(torch.float32).eps
    for start in range(0, length, group_size):
        stop = min(length, start + group_size)
        index = [slice(None)] * data.dim()
        index[group_dim] = slice(start, stop)
        chunk = data[tuple(index)].float()
        mn = chunk.amin(dim=group_dim, keepdim=True)
        mx = chunk.amax(dim=group_dim, keepdim=True)
        scale = ((mx - mn) / max_code).clamp_min(eps)
        groups.append(torch.round((chunk - mn) / scale).clamp_(0, max_code).to(torch.uint8))
        scales.append(scale.squeeze(group_dim).to(data.dtype))
        minima.append(mn.squeeze(group_dim).to(data.dtype))
    codes = torch.cat(groups, dim=group_dim)
    scale = torch.stack(scales, dim=group_dim)
    minimum = torch.stack(minima, dim=group_dim)
    return _pack_codes(codes, bits, pack_dim), scale, minimum, length


def _quantize_grouped(
    data: torch.Tensor,
    *,
    bits: int,
    group_dim: int,
    group_size: int,
    pack_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Vectorize complete groups while preserving the reference KIVI layout."""

    group_dim = group_dim if group_dim >= 0 else data.dim() + group_dim
    length = int(data.shape[group_dim])
    if length == 0 or length % group_size:
        return _quantize_grouped_reference(
            data,
            bits=bits,
            group_dim=group_dim,
            group_size=group_size,
            pack_dim=pack_dim,
        )

    max_code = float((1 << bits) - 1)
    eps = torch.finfo(torch.float32).eps
    moved = data.movedim(group_dim, 0)
    grouped = moved.reshape(length // group_size, group_size, *moved.shape[1:]).float()
    minimum_grouped = grouped.amin(dim=1, keepdim=True)
    maximum_grouped = grouped.amax(dim=1, keepdim=True)
    scale_grouped = ((maximum_grouped - minimum_grouped) / max_code).clamp_min(eps)
    codes_grouped = (
        torch.round((grouped - minimum_grouped) / scale_grouped)
        .clamp_(0, max_code)
        .to(torch.uint8)
    )
    codes = codes_grouped.reshape(length, *moved.shape[1:]).movedim(0, group_dim)
    scale = scale_grouped.squeeze(1).movedim(0, group_dim).to(data.dtype)
    minimum = minimum_grouped.squeeze(1).movedim(0, group_dim).to(data.dtype)
    return _pack_codes(codes, bits, pack_dim), scale, minimum, length


def _dequantize_grouped(
    packed: torch.Tensor,
    scale: torch.Tensor,
    minimum: torch.Tensor,
    *,
    bits: int,
    group_dim: int,
    group_size: int,
    pack_dim: int,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    codes = _unpack_codes(packed, bits, pack_dim, length).float()
    chunks = []
    for group_index, start in enumerate(range(0, length, group_size)):
        stop = min(length, start + group_size)
        index = [slice(None)] * codes.dim()
        index[group_dim] = slice(start, stop)
        stat_index = [slice(None)] * scale.dim()
        stat_index[group_dim] = group_index
        group_scale = scale[tuple(stat_index)].unsqueeze(group_dim).float()
        group_minimum = minimum[tuple(stat_index)].unsqueeze(group_dim).float()
        chunks.append(codes[tuple(index)] * group_scale + group_minimum)
    return torch.cat(chunks, dim=group_dim).to(dtype)


@dataclass
class PackedKIVICache:
    """A KIVI-style packed KV segment with a BF16 residual suffix."""

    key_code: torch.Tensor | None
    key_scale: torch.Tensor | None
    key_minimum: torch.Tensor | None
    value_code: torch.Tensor | None
    value_scale: torch.Tensor | None
    value_minimum: torch.Tensor | None
    residual_key: torch.Tensor
    residual_value: torch.Tensor
    logical_token_ids: torch.Tensor
    token_type_id: int
    k_bits: int
    v_bits: int
    group_size: int
    quantized_tokens: int
    head_dim: int
    num_kv_heads: int
    backend: str = "reference"
    kernel_native_layout: bool = False
    budget_shortfall_events: int = 0

    @classmethod
    def from_dense(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        logical_token_ids: torch.Tensor,
        token_type_id: int,
        k_bits: int,
        v_bits: int,
        group_size: int,
        residual_length: int,
        backend: str = "reference",
    ) -> "PackedKIVICache":
        if key.shape != value.shape or key.dim() != 3:
            raise ValueError("KIVI physical cache expects matching [tokens, kv_heads, head_dim] tensors")
        if k_bits not in {2, 4, 8} or v_bits not in {2, 4, 8}:
            raise ValueError("KIVI physical cache supports 2, 4, or 8 bit payloads")
        tokens, heads, head_dim = map(int, key.shape)
        if group_size <= 0:
            raise ValueError("KIVI physical cache group_size must be positive")
        if backend == "cuda" and (k_bits in {2, 4} or v_bits in {2, 4}):
            if group_size not in {64, 128}:
                raise ValueError(
                    "The pinned KIVI CUDA kernel supports group_size 64 or 128"
                )
            if head_dim % group_size != 0:
                raise ValueError(
                    "The pinned KIVI CUDA value kernel requires head_dim divisible by group_size"
                )
        requested_residual = min(tokens, max(0, int(residual_length)))
        # The official outer-dimension kernel reconstructs its output width as
        # num_groups * group_size. Keep any incomplete prefix group in BF16 so
        # the packed prefix has an exact logical length without padding tokens.
        quantized = ((tokens - requested_residual) // group_size) * group_size
        residual = tokens - quantized
        q_key = key[:quantized]
        q_value = value[:quantized]
        kernel_native_layout = bool(
            backend == "cuda"
            and quantized
            and k_bits in {2, 4}
            and v_bits in {2, 4}
        )
        if quantized:
            if kernel_native_layout:
                from .kivi_cuda import quantize_kivi_native

                (
                    key_code,
                    key_scale,
                    key_minimum,
                    value_code,
                    value_scale,
                    value_minimum,
                ) = quantize_kivi_native(
                    q_key,
                    q_value,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    group_size=group_size,
                )
            else:
                key_code, key_scale, key_minimum, _ = _quantize_grouped(
                    q_key,
                    bits=k_bits,
                    group_dim=0,
                    group_size=group_size,
                    pack_dim=0,
                )
                value_code, value_scale, value_minimum, _ = _quantize_grouped(
                    q_value,
                    bits=v_bits,
                    group_dim=2,
                    group_size=group_size,
                    pack_dim=2,
                )
        else:
            key_code = key_scale = key_minimum = None
            value_code = value_scale = value_minimum = None
        if kernel_native_layout and quantized and key_code.dim() != 3:
            raise RuntimeError("KIVI native packer returned an invalid layout")
        return cls(
            key_code=key_code,
            key_scale=key_scale,
            key_minimum=key_minimum,
            value_code=value_code,
            value_scale=value_scale,
            value_minimum=value_minimum,
            # A contiguous suffix view can still retain the entire dense
            # segment's backing allocation. Clone so resident bytes match the
            # physical cache that remains live after materialization.
            residual_key=key[quantized:].clone(),
            residual_value=value[quantized:].clone(),
            logical_token_ids=logical_token_ids.detach().clone(),
            token_type_id=int(token_type_id),
            k_bits=int(k_bits),
            v_bits=int(v_bits),
            group_size=int(group_size),
            quantized_tokens=quantized,
            head_dim=head_dim,
            num_kv_heads=heads,
            backend=str(backend),
            kernel_native_layout=kernel_native_layout,
        )

    @property
    def token_count(self) -> int:
        return int(self.logical_token_ids.numel())

    @property
    def resident_bytes(self) -> int:
        tensors = (
            self.key_code,
            self.key_scale,
            self.key_minimum,
            self.value_code,
            self.value_scale,
            self.value_minimum,
            self.residual_key,
            self.residual_value,
            self.logical_token_ids,
        )
        return sum(tensor_bytes(item) for item in tensors)

    @property
    def byte_breakdown(self) -> dict[str, int]:
        return {
            "payload_bytes": sum(
                tensor_bytes(item)
                for item in (self.key_code, self.value_code)
            ),
            "scale_bytes": sum(
                tensor_bytes(item)
                for item in (self.key_scale, self.value_scale)
            ),
            "minimum_bytes": sum(
                tensor_bytes(item)
                for item in (self.key_minimum, self.value_minimum)
            ),
            "residual_bytes": tensor_bytes(self.residual_key)
            + tensor_bytes(self.residual_value),
            "indices_bytes": tensor_bytes(self.logical_token_ids),
            "score_bytes": 0,
        }

    @property
    def full_precision_bytes(self) -> int:
        element_size = int(self.residual_key.element_size())
        return self.token_count * self.num_kv_heads * self.head_dim * 2 * element_size

    def dequantize(self, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = dtype or self.residual_key.dtype
        if self.quantized_tokens:
            if self.kernel_native_layout:
                key_code = self.key_code.permute(2, 0, 1)
                key_scale = self.key_scale.permute(2, 0, 1)
                key_minimum = self.key_minimum.permute(2, 0, 1)
                value_code = self.value_code.permute(1, 0, 2)
                value_scale = self.value_scale.permute(1, 0, 2)
                value_minimum = self.value_minimum.permute(1, 0, 2)
            else:
                key_code = self.key_code
                key_scale = self.key_scale
                key_minimum = self.key_minimum
                value_code = self.value_code
                value_scale = self.value_scale
                value_minimum = self.value_minimum
            q_key = _dequantize_grouped(
                key_code,
                key_scale,
                key_minimum,
                bits=self.k_bits,
                group_dim=0,
                group_size=self.group_size,
                pack_dim=0,
                length=self.quantized_tokens,
                dtype=dtype,
            )
            q_value = _dequantize_grouped(
                value_code,
                value_scale,
                value_minimum,
                bits=self.v_bits,
                group_dim=2,
                group_size=self.group_size,
                pack_dim=2,
                length=self.head_dim,
                dtype=dtype,
            )
            key = torch.cat([q_key, self.residual_key.to(dtype)], dim=0)
            value = torch.cat([q_value, self.residual_value.to(dtype)], dim=0)
        else:
            key = self.residual_key.to(dtype)
            value = self.residual_value.to(dtype)
        return key, value

    def repack_monotonic(self, *, bits: int, group_size: int | None = None) -> bool:
        """Lower precision without permitting a deleted precision level to return."""

        target_bits = int(bits)
        target_group = int(group_size or self.group_size)
        if target_bits > self.k_bits or target_bits > self.v_bits or target_group < self.group_size:
            self.budget_shortfall_events += 1
            return False
        if target_bits == self.k_bits and target_bits == self.v_bits and target_group == self.group_size:
            return True
        key, value = self.dequantize(dtype=self.residual_key.dtype)
        replacement = PackedKIVICache.from_dense(
            key,
            value,
            logical_token_ids=self.logical_token_ids,
            token_type_id=self.token_type_id,
            k_bits=target_bits,
            v_bits=target_bits,
            group_size=target_group,
            residual_length=int(self.residual_key.shape[0]),
            backend=self.backend,
        )
        shortfalls = self.budget_shortfall_events
        self.__dict__.update(replacement.__dict__)
        self.budget_shortfall_events = shortfalls
        return True

    def attention_parts(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """Return logits and an AV closure without retaining a BF16 replica.

        This reference backend is used by CPU tests and as a correctness oracle.
        Raven physical configs require the CUDA backend and do not silently use
        this path.
        """

        if self.backend == "cuda":
            from .kivi_cuda import packed_kivi_attention_parts

            return packed_kivi_attention_parts(self, query)
        if self.backend != "reference":
            raise ValueError(f"Unknown KIVI physical backend: {self.backend}")
        key, value = self.dequantize(dtype=query.dtype)
        logits = gqa_query_key_logits(query, key) / math.sqrt(query.shape[-1])

        def apply_value(probabilities: torch.Tensor) -> torch.Tensor:
            return gqa_probability_value(probabilities, value)

        return logits, apply_value


@dataclass
class GQAH2OCache:
    """Per-KV-head physical H2O cache with fixed-width selected storage."""

    key: torch.Tensor  # [kv_heads, retained_tokens, head_dim]
    value: torch.Tensor
    logical_token_ids: torch.Tensor  # [kv_heads, retained_tokens]
    cumulative_scores: torch.Tensor
    token_type_id: int
    original_tokens: int
    heavy_tokens: int
    recent_tokens: int
    selection_frozen: bool = False
    budget_shortfall_tokens: int = 0

    @classmethod
    def from_dense(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        logical_token_ids: torch.Tensor,
        token_type_id: int,
        per_kv_head_scores: torch.Tensor,
        target_tokens: int,
        recent_tokens: int,
        selection_frozen: bool = False,
    ) -> "GQAH2OCache":
        if key.shape != value.shape or key.dim() != 3:
            raise ValueError("Physical H2O expects matching [tokens, kv_heads, head_dim] tensors")
        tokens, kv_heads, head_dim = map(int, key.shape)
        if tuple(per_kv_head_scores.shape) != (kv_heads, tokens):
            raise ValueError("Physical H2O scores must be [kv_heads, segment_tokens]")
        target = min(tokens, max(1, int(target_tokens)))
        recent = min(target, max(0, int(recent_tokens)))
        heavy = target - recent
        history_end = tokens - recent
        heavy_k = min(heavy, history_end)
        if heavy_k:
            heavy_idx = torch.topk(
                per_kv_head_scores[:, :history_end], k=heavy_k, dim=1
            ).indices.sort(dim=1).values
        else:
            heavy_idx = torch.empty(
                (kv_heads, 0), device=key.device, dtype=torch.long
            )
        if recent:
            recent_idx = torch.arange(
                tokens - recent, tokens, device=key.device
            ).unsqueeze(0).expand(kv_heads, -1)
        else:
            recent_idx = torch.empty(
                (kv_heads, 0), device=key.device, dtype=torch.long
            )
        indexes = torch.cat([heavy_idx, recent_idx], dim=1)
        source_key = key.transpose(0, 1)
        source_value = value.transpose(0, 1)
        gather = indexes.unsqueeze(-1).expand(-1, -1, head_dim)
        logical = logical_token_ids.to(device=key.device).unsqueeze(0).expand(kv_heads, -1)
        return cls(
            key=torch.gather(source_key, 1, gather).contiguous(),
            value=torch.gather(source_value, 1, gather).contiguous(),
            logical_token_ids=torch.gather(logical, 1, indexes).contiguous(),
            cumulative_scores=torch.gather(per_kv_head_scores, 1, indexes).float().contiguous(),
            token_type_id=int(token_type_id),
            original_tokens=tokens,
            heavy_tokens=heavy,
            recent_tokens=recent,
            selection_frozen=bool(selection_frozen),
        )

    @property
    def num_kv_heads(self) -> int:
        return int(self.key.shape[0])

    @property
    def retained_tokens(self) -> int:
        return int(self.key.shape[1])

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor_bytes(item)
            for item in (self.key, self.value, self.logical_token_ids, self.cumulative_scores)
        )

    @property
    def byte_breakdown(self) -> dict[str, int]:
        return {
            "payload_bytes": tensor_bytes(self.key) + tensor_bytes(self.value),
            "scale_bytes": 0,
            "minimum_bytes": 0,
            "residual_bytes": 0,
            "indices_bytes": tensor_bytes(self.logical_token_ids),
            "score_bytes": tensor_bytes(self.cumulative_scores),
        }

    @property
    def full_precision_bytes(self) -> int:
        return (
            self.original_tokens
            * self.num_kv_heads
            * int(self.key.shape[-1])
            * 2
            * self.key.element_size()
        )

    def attention_parts(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        key = self.key.transpose(0, 1)
        value = self.value.transpose(0, 1)
        logits = gqa_query_key_logits(query, key) / math.sqrt(query.shape[-1])

        def apply_value(probabilities: torch.Tensor) -> torch.Tensor:
            return gqa_probability_value(probabilities, value)

        return logits, apply_value

    def update_scores(self, probabilities: torch.Tensor) -> None:
        """Accumulate attention received by each surviving token per KV head."""

        if probabilities.dim() != 3:
            raise ValueError("H2O probabilities must be [query_heads, queries, tokens]")
        query_heads = int(probabilities.shape[0])
        if query_heads % self.num_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads for GQA H2O")
        groups = query_heads // self.num_kv_heads
        received = probabilities.float().sum(dim=1).reshape(
            self.num_kv_heads, groups, self.retained_tokens
        ).sum(dim=1)
        self.cumulative_scores = self.cumulative_scores + received

    def enforce_budget(self) -> None:
        self.trim(self.heavy_tokens + self.recent_tokens, self.recent_tokens)

    def request_budget(self, target_tokens: int, recent_tokens: int) -> int:
        requested = max(1, int(target_tokens))
        effective = min(requested, self.retained_tokens)
        self.budget_shortfall_tokens += requested - effective
        self.recent_tokens = min(effective, max(0, int(recent_tokens)))
        self.heavy_tokens = effective - self.recent_tokens
        self.enforce_budget()
        return effective

    def trim(self, target_tokens: int, recent_tokens: int) -> None:
        target = min(self.retained_tokens, max(1, int(target_tokens)))
        if target >= self.retained_tokens:
            return
        recent = min(target, max(0, int(recent_tokens)))
        heavy = target - recent
        keep_per_head = []
        for head in range(self.num_kv_heads):
            if recent:
                recent_idx = torch.arange(
                    self.retained_tokens - recent,
                    self.retained_tokens,
                    device=self.key.device,
                )
                history_end = self.retained_tokens - recent
            else:
                recent_idx = torch.empty(0, device=self.key.device, dtype=torch.long)
                history_end = self.retained_tokens
            if heavy:
                heavy_idx = torch.topk(
                    self.cumulative_scores[head, :history_end], k=min(heavy, history_end)
                ).indices.sort().values
            else:
                heavy_idx = torch.empty(0, device=self.key.device, dtype=torch.long)
            keep_per_head.append(torch.cat([heavy_idx, recent_idx]))
        indexes = torch.stack(keep_per_head, dim=0)
        gather = indexes.unsqueeze(-1).expand(-1, -1, self.key.shape[-1])
        self.key = torch.gather(self.key, 1, gather).contiguous()
        self.value = torch.gather(self.value, 1, gather).contiguous()
        self.logical_token_ids = torch.gather(self.logical_token_ids, 1, indexes).contiguous()
        self.cumulative_scores = torch.gather(self.cumulative_scores, 1, indexes).contiguous()


PhysicalSegment = PackedKIVICache | GQAH2OCache


@dataclass
class PhysicalLayerCache:
    segments: list[PhysicalSegment]

    @property
    def resident_bytes(self) -> int:
        return sum(segment.resident_bytes for segment in self.segments)

    @property
    def full_precision_bytes(self) -> int:
        return sum(segment.full_precision_bytes for segment in self.segments)

    def add(self, segment: PhysicalSegment) -> None:
        self.segments.append(segment)
