"""Dense tensor cache compaction with packed multi-sample support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..core.decisions import PhysicalCacheRequest
from .physical import GQAH2OCache, PackedKIVICache


@dataclass
class CacheMutationResult:
    key: torch.Tensor
    value: torch.Tensor
    key_type_ids: torch.Tensor | None
    sample_lens: torch.Tensor
    original_bytes: int
    compacted_bytes: int
    physical_segments_by_sample: list[list[Any]] | None = None

    @property
    def bytes_saved(self) -> int:
        return self.original_bytes - self.compacted_bytes


class TensorCacheStorageBackend:
    """Compact a packed dense cache according to one keep mask per sample."""

    @staticmethod
    def compact(
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        key_type_ids: torch.Tensor | None,
        sample_lens: torch.Tensor,
        keep_masks: list[torch.Tensor],
        sample_lengths: list[int] | None = None,
    ) -> CacheMutationResult:
        if key.shape != value.shape or key.dim() != 3:
            raise ValueError("Tensor cache backend expects matching [tokens, heads, head_dim] KV tensors")
        lengths = sample_lengths
        if lengths is None:
            lengths = [int(item) for item in sample_lens.to("cpu").tolist()]
        if sum(lengths) != int(key.shape[0]):
            raise ValueError("Packed sample lengths do not match the cache token count")
        if len(keep_masks) != len(lengths):
            raise ValueError("Tensor cache backend requires one keep mask per packed sample")
        if key_type_ids is not None and int(key_type_ids.numel()) != int(key.shape[0]):
            raise ValueError("key_type_ids do not match the cache token count")

        key_chunks = []
        value_chunks = []
        type_chunks = []
        normalized_masks = []
        offset = 0
        for length, keep_mask in zip(lengths, keep_masks):
            mask = keep_mask.to(device=key.device, dtype=torch.bool)
            if mask.dim() != 1 or int(mask.numel()) != length:
                raise ValueError("Keep mask does not match its packed sample length")
            sample_slice = slice(offset, offset + length)
            key_chunks.append(key[sample_slice][mask])
            value_chunks.append(value[sample_slice][mask])
            if key_type_ids is not None:
                type_chunks.append(key_type_ids[sample_slice][mask.to(key_type_ids.device)])
            normalized_masks.append(mask)
            offset += length

        def _cat_or_empty(chunks: list[torch.Tensor], source: torch.Tensor) -> torch.Tensor:
            if chunks:
                return torch.cat(chunks, dim=0)
            return source[:0]

        compacted_key = _cat_or_empty(key_chunks, key)
        compacted_value = _cat_or_empty(value_chunks, value)
        compacted_types = None
        if key_type_ids is not None:
            compacted_types = _cat_or_empty(type_chunks, key_type_ids)
        original_bytes = int(key.numel() * key.element_size() + value.numel() * value.element_size())
        compacted_bytes = int(
            compacted_key.numel() * compacted_key.element_size()
            + compacted_value.numel() * compacted_value.element_size()
        )
        compacted_lens = (
            torch.stack(
                [mask.sum(dtype=sample_lens.dtype) for mask in normalized_masks]
            ).to(device=sample_lens.device, dtype=sample_lens.dtype)
            if normalized_masks
            else sample_lens[:0]
        )
        return CacheMutationResult(
            key=compacted_key,
            value=compacted_value,
            key_type_ids=compacted_types,
            sample_lens=compacted_lens,
            original_bytes=original_bytes,
            compacted_bytes=compacted_bytes,
        )

    @staticmethod
    def compact_and_materialize(
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        key_type_ids: torch.Tensor | None,
        sample_lens: torch.Tensor,
        keep_masks: list[torch.Tensor],
        physical_requests: list[list[PhysicalCacheRequest]],
        sample_lengths: list[int] | None = None,
    ) -> CacheMutationResult:
        """Remove requested segments from dense KV and build physical layouts."""

        lengths = sample_lengths
        if lengths is None:
            lengths = [int(item) for item in sample_lens.to("cpu").tolist()]
        if len(physical_requests) != len(lengths):
            raise ValueError("Physical requests require one list per packed sample")
        physical_by_sample: list[list[Any]] = []
        offset = 0
        for sample_idx, (length, requests) in enumerate(zip(lengths, physical_requests)):
            sample_key = key[offset : offset + length]
            sample_value = value[offset : offset + length]
            occupied = torch.zeros(length, device=key.device, dtype=torch.bool)
            segments: list[Any] = []
            normalized_requests = []
            for request in requests:
                if request.token_count < length:
                    raise ValueError(
                        "PhysicalCacheRequest is shorter than the persistent cache sample"
                    )
                segment_mask = request.segment_mask[:length].to(
                    device=key.device, dtype=torch.bool
                )
                normalized_requests.append((request, segment_mask))
            if len(normalized_requests) > 1:
                claims = torch.stack(
                    [mask for _, mask in normalized_requests], dim=0
                ).sum(dim=0)
                if bool((claims > 1).any()):
                    raise ValueError(
                        f"Overlapping physical requests in sample {sample_idx} are not allowed"
                    )
            for request, segment_mask in normalized_requests:
                occupied |= segment_mask
                token_span = request.metadata.get("token_span")
                segment_slice = None
                if token_span is not None:
                    if not isinstance(token_span, (tuple, list)) or len(token_span) != 2:
                        raise ValueError("Physical token_span must be a (start, stop) pair")
                    span_start, span_stop = map(int, token_span)
                    if not 0 <= span_start < span_stop <= length:
                        raise ValueError("Physical token_span is outside its packed sample")
                    segment_slice = slice(span_start, span_stop)
                    segment_idx = torch.arange(
                        span_start, span_stop, device=key.device, dtype=torch.long
                    )
                else:
                    segment_idx = torch.nonzero(
                        segment_mask, as_tuple=False
                    ).flatten()
                    if int(segment_idx.numel()) == 0:
                        continue
                metadata_type_id = request.metadata.get("token_type_id")
                type_id = int(metadata_type_id) if metadata_type_id is not None else -1
                if type_id < 0 and key_type_ids is not None:
                    segment_types = torch.unique(
                        key_type_ids[offset : offset + length][segment_idx]
                    )
                    if int(segment_types.numel()) == 1:
                        type_id = int(segment_types.item())
                logical_ids = segment_idx.to(dtype=torch.int64)
                if segment_slice is None:
                    dense_key = sample_key[segment_idx].contiguous()
                    dense_value = sample_value[segment_idx].contiguous()
                else:
                    dense_key = sample_key[segment_slice]
                    dense_value = sample_value[segment_slice]
                if request.operator == "kivi_packed_quantization":
                    bits = int(request.params.get("bits", 4))
                    segment = PackedKIVICache.from_dense(
                        dense_key,
                        dense_value,
                        logical_token_ids=logical_ids,
                        token_type_id=type_id,
                        k_bits=int(request.params.get("k_bits", bits)),
                        v_bits=int(request.params.get("v_bits", bits)),
                        group_size=int(request.params.get("group_size", 64)),
                        residual_length=int(request.params.get("residual_length", 32)),
                        backend=str(request.params.get("backend", "reference")),
                    )
                elif request.operator == "h2o_physical_gqa":
                    scores = request.per_kv_head_scores
                    if scores is None:
                        raise ValueError("Physical H2O request is missing per-KV-head scores")
                    scores = scores[:, :length].to(device=key.device, dtype=torch.float32)
                    segment_scores = scores[:, segment_idx]
                    target = request.params.get("budget", request.params.get("keep_k"))
                    if target is None:
                        ratio = float(request.params.get("target_budget_ratio", 1.0))
                        target = max(1, int(round(int(segment_idx.numel()) * ratio)))
                    recent = request.params.get("recent_size")
                    if recent is None:
                        recent = int(
                            round(
                                int(target)
                                * float(request.params.get("recent_budget_fraction", 0.25))
                            )
                        )
                    segment = GQAH2OCache.from_dense(
                        dense_key,
                        dense_value,
                        logical_token_ids=logical_ids,
                        token_type_id=type_id,
                        per_kv_head_scores=segment_scores,
                        target_tokens=int(target),
                        recent_tokens=int(recent),
                        selection_frozen=bool(
                            request.params.get("selection_frozen", False)
                        ),
                    )
                else:
                    raise ValueError(f"Unknown physical cache request: {request.operator}")
                segments.append(segment)
            keep_masks[sample_idx] &= ~occupied
            physical_by_sample.append(segments)
            offset += length

        mutation = TensorCacheStorageBackend.compact(
            key=key,
            value=value,
            key_type_ids=key_type_ids,
            sample_lens=sample_lens,
            keep_masks=keep_masks,
            sample_lengths=lengths,
        )
        mutation.physical_segments_by_sample = physical_by_sample
        return mutation
