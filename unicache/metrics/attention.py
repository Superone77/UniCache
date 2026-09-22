"""Attention metrics over one logical KV segment."""

from __future__ import annotations

import math
from typing import Any

import torch


DEFAULT_STEP_CHUNK_SIZE = 5
DEFAULT_LAYER_CHUNK_SIZE = 4


def _summary_groups(
    segment_indices: torch.Tensor,
    *,
    block_size: int,
    representatives: int,
) -> list[torch.Tensor]:
    groups: list[torch.Tensor] = []
    for start in range(0, int(segment_indices.numel()), block_size):
        block = segment_indices[
            start : min(start + block_size, int(segment_indices.numel()))
        ]
        groups.extend(
            part
            for part in torch.tensor_split(
                block, min(representatives, int(block.numel()))
            )
            if int(part.numel()) > 0
        )
    return groups


def _segment_identity_matches(
    archived_mask: torch.Tensor,
    current_mask: torch.Tensor,
) -> bool:
    archived = archived_mask.to(device="cpu", dtype=torch.bool)
    current = current_mask.detach().to(device="cpu", dtype=torch.bool)
    if int(current.numel()) < int(archived.numel()):
        return False
    if not torch.equal(current[: archived.numel()], archived):
        return False
    return not bool(current[archived.numel() :].any())


def build_block_summary_metadata(
    *,
    k: torch.Tensor,
    segment_mask: torch.Tensor,
    block_size: int = 8,
    representatives: int = 4,
    active_keep_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build reusable centroid metadata while the full segment is resident.

    The returned centroids are independent of the current query. Later mass
    estimates may subtract the still-resident keys from each cached group, so
    they do not need to read keys that have already been offloaded.
    """

    if k.ndim != 3:
        raise ValueError("k must have shape [tokens, heads, head_dim]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if representatives <= 0 or representatives > block_size:
        raise ValueError("representatives must be in [1, block_size]")
    segment = segment_mask.to(device=k.device, dtype=torch.bool)
    if segment.ndim != 1 or int(segment.numel()) != int(k.shape[0]):
        raise ValueError("segment_mask must be one-dimensional and match K")
    if active_keep_mask is not None:
        keep = active_keep_mask.to(device=k.device, dtype=torch.bool)
        expected = (int(k.shape[1]), int(k.shape[0]))
        if tuple(keep.shape) != expected:
            raise ValueError("active_keep_mask must have shape [heads, keys]")
        if not bool(keep[:, segment].all()):
            raise ValueError(
                "Block summary metadata requires the full segment working set"
            )

    segment_indices = torch.nonzero(segment, as_tuple=False).flatten()
    groups = _summary_groups(
        segment_indices,
        block_size=block_size,
        representatives=representatives,
    )
    if not groups:
        raise ValueError("Block summary metadata requires a non-empty segment")

    centroids = torch.stack(
        [k.detach()[group].float().mean(dim=0) for group in groups], dim=1
    ).to(dtype=k.dtype)
    counts = torch.tensor(
        [int(group.numel()) for group in groups],
        device=k.device,
        dtype=torch.int32,
    )
    group_indices = [
        group.detach().to(device="cpu", dtype=torch.long) for group in groups
    ]
    metadata_bytes = int(centroids.numel() * centroids.element_size())
    metadata_bytes += int(counts.numel() * counts.element_size())
    metadata_bytes += sum(int(group.numel() * 4) for group in group_indices)
    return {
        "segment_mask": segment.detach().to(device="cpu", dtype=torch.bool),
        "group_indices": group_indices,
        "centroids": centroids,
        "counts": counts,
        "block_size": int(block_size),
        "representatives": int(representatives),
        "num_heads": int(k.shape[1]),
        "head_dim": int(k.shape[2]),
        "metadata_bytes": metadata_bytes,
        "source": "cached_prefill",
    }


def block_summary_metadata_key(
    *,
    run_id: str,
    phase: str,
    branch: str | None,
    cfg_branch: str,
    batch_idx: int,
    layer_idx: int,
    segment_id: str,
    operator_id: str,
) -> tuple[Any, ...]:
    return (
        "block_summary_metadata",
        str(run_id),
        str(phase),
        branch,
        str(cfg_branch),
        int(batch_idx),
        int(layer_idx),
        str(segment_id),
        str(operator_id),
    )


def attention_mass_and_k90(
    attention_probs: torch.Tensor,
    segment_mask: torch.Tensor,
) -> tuple[float, float]:
    """Return segment attention mass and the fraction of tokens covering 90% of it.

    Attention probabilities are averaged over query heads and query positions.
    K90 is normalized by the number of tokens in the segment. A segment with no
    attention mass is treated conservatively as dense (K90=1).
    """

    if attention_probs.ndim != 3:
        raise ValueError(
            "attention_probs must have shape [heads, queries, keys], got "
            f"{tuple(attention_probs.shape)}"
        )
    mask = segment_mask.to(device=attention_probs.device, dtype=torch.bool)
    if mask.ndim != 1 or int(mask.numel()) != int(attention_probs.shape[-1]):
        raise ValueError("segment_mask must be one-dimensional and match the key length")
    token_scores = attention_probs.float().mean(dim=(0, 1))[mask]
    token_count = int(token_scores.numel())
    if token_count == 0:
        return 0.0, 1.0
    mass = float(token_scores.sum().item())
    if mass <= 0.0:
        return 0.0, 1.0
    normalized = token_scores / mass
    cumulative = torch.cumsum(torch.sort(normalized, descending=True).values, dim=0)
    k = int(torch.searchsorted(cumulative, torch.tensor(0.9, device=cumulative.device)).item()) + 1
    return mass, min(1.0, float(k) / float(token_count))


def block_summary_partition_components(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    attention_probs: torch.Tensor,
    segment_mask: torch.Tensor,
    active_keep_mask: torch.Tensor,
    attn_mask: torch.Tensor | None,
    block_size: int = 8,
    representatives: int = 4,
    calibration_factor: float = 1.0,
    summary_metadata: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return active segment mass and estimated offloaded/active partition ratio.

    The active attention probabilities are already normalized over the GPU
    working set. Offloaded keys are summarized by contiguous subblock
    centroids. The summary estimates ``Z_off / Z_active`` independently for
    every query/head pair before reconstructing the segment mass.

    When ``summary_metadata`` is provided, only currently active keys are read
    from ``k``. Calling without metadata remains an explicit full-K oracle path
    for estimator calibration tests.
    """

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must have shape [tokens, heads, head_dim]")
    if attention_probs.ndim != 3:
        raise ValueError("attention_probs must have shape [heads, queries, keys]")
    if int(q.shape[1]) != int(k.shape[1]):
        raise ValueError("block summary estimator requires repeated K heads")
    if tuple(attention_probs.shape) != (
        int(q.shape[1]),
        int(q.shape[0]),
        int(k.shape[0]),
    ):
        raise ValueError("attention_probs shape does not match q/k")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if representatives <= 0 or representatives > block_size:
        raise ValueError("representatives must be in [1, block_size]")
    if calibration_factor <= 0.0:
        raise ValueError("calibration_factor must be positive")

    device = attention_probs.device
    segment = segment_mask.to(device=device, dtype=torch.bool)
    keep = active_keep_mask.to(device=device, dtype=torch.bool)
    num_heads = int(q.shape[1])
    query_len = int(q.shape[0])
    key_len = int(k.shape[0])
    if tuple(keep.shape) != (num_heads, key_len):
        raise ValueError("active_keep_mask must have shape [heads, keys]")
    if segment.ndim != 1 or int(segment.numel()) != key_len:
        raise ValueError("segment_mask must be one-dimensional and match K")

    model_visible = torch.ones(
        (query_len, key_len), device=device, dtype=torch.bool
    )
    if attn_mask is not None:
        model_visible &= attn_mask.to(device=device, dtype=torch.bool)
    active = keep[:, None, :] & model_visible[None, :, :]
    offloaded = (
        segment[None, None, :]
        & ~keep[:, None, :]
        & model_visible[None, :, :]
    )
    active_segment_mass = attention_probs.float()[..., segment].sum(dim=-1)
    metadata_source = "cached_prefill"
    if summary_metadata is None:
        summary_metadata = build_block_summary_metadata(
            k=k,
            segment_mask=segment,
            block_size=block_size,
            representatives=representatives,
        )
        metadata_source = "full_k_oracle"
    if int(summary_metadata.get("block_size", -1)) != int(block_size):
        raise ValueError("Block summary metadata block_size mismatch")
    if int(summary_metadata.get("representatives", -1)) != int(representatives):
        raise ValueError("Block summary metadata representatives mismatch")
    if int(summary_metadata.get("num_heads", -1)) != num_heads:
        raise ValueError("Block summary metadata head count mismatch")
    if int(summary_metadata.get("head_dim", -1)) != int(k.shape[-1]):
        raise ValueError("Block summary metadata head_dim mismatch")
    if not _segment_identity_matches(summary_metadata["segment_mask"], segment):
        raise ValueError("Block summary metadata segment identity changed")
    if not bool(offloaded.any()):
        mass = float(active_segment_mass.mean().item())
        ratio = torch.zeros_like(active_segment_mass)
        return active_segment_mass, ratio, {
            "active_attention_mass": mass,
            "estimated_offloaded_partition_ratio": 0.0,
            "corrected_attention_mass": mass,
            "offloaded_head_query_tokens": 0.0,
            "summary_metadata_bytes": float(summary_metadata["metadata_bytes"]),
            "summary_metadata_source": metadata_source,
        }

    q_h = q.transpose(0, 1).float()
    k_h = k.transpose(0, 1).float()
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    log_z_active = torch.full(
        (num_heads, query_len), float("-inf"), device=device, dtype=torch.float32
    )
    for head_idx in range(num_heads):
        active_indices = torch.nonzero(keep[head_idx], as_tuple=False).flatten()
        if int(active_indices.numel()) == 0:
            continue
        active_scores = torch.matmul(
            q_h[head_idx], k_h[head_idx, active_indices].transpose(0, 1)
        ) / math.sqrt(q.shape[-1])
        active_visible = model_visible[:, active_indices]
        log_z_active[head_idx] = torch.logsumexp(
            active_scores.masked_fill(~active_visible, neg_inf), dim=-1
        )

    cached_centroids = summary_metadata["centroids"].to(
        device=device, dtype=torch.float32
    )
    cached_counts = summary_metadata["counts"].to(
        device=device, dtype=torch.int64
    )
    group_terms: list[torch.Tensor] = []
    for group_idx, group_cpu in enumerate(summary_metadata["group_indices"]):
        group = group_cpu.to(device=device, dtype=torch.long)
        visible = model_visible[:, group]
        visible_counts = visible.sum(dim=-1)
        full_count = int(cached_counts[group_idx].item())
        if bool(((visible_counts != 0) & (visible_counts != full_count)).any()):
            raise ValueError(
                "Cached block summaries require group-uniform attention visibility"
            )
        per_head_terms = []
        for head_idx in range(num_heads):
            active_group = group[keep[head_idx, group]]
            offloaded_count = full_count - int(active_group.numel())
            if offloaded_count <= 0:
                logits = torch.full(
                    (query_len,), float("-inf"), device=device, dtype=torch.float32
                )
            else:
                full_sum = cached_centroids[head_idx, group_idx] * float(full_count)
                active_sum = (
                    k_h[head_idx, active_group].sum(dim=0)
                    if int(active_group.numel()) > 0
                    else torch.zeros_like(full_sum)
                )
                offloaded_centroid = (full_sum - active_sum) / float(offloaded_count)
                logits = torch.matmul(q_h[head_idx], offloaded_centroid) / math.sqrt(
                    q.shape[-1]
                )
                logits = logits + math.log(offloaded_count)
                logits = logits.masked_fill(visible_counts == 0, neg_inf)
            per_head_terms.append(logits)
        group_terms.append(torch.stack(per_head_terms, dim=0))
    log_z_off = torch.logsumexp(torch.stack(group_terms, dim=-1), dim=-1)

    partition_ratio = calibration_factor * torch.exp(log_z_off - log_z_active)
    partition_ratio = torch.nan_to_num(
        partition_ratio,
        nan=0.0,
        posinf=torch.finfo(torch.float32).max,
        neginf=0.0,
    )
    corrected = (active_segment_mass + partition_ratio) / (1.0 + partition_ratio)
    return active_segment_mass, partition_ratio, {
        "active_attention_mass": float(active_segment_mass.mean().item()),
        "estimated_offloaded_partition_ratio": float(partition_ratio.mean().item()),
        "corrected_attention_mass": float(corrected.mean().item()),
        "offloaded_head_query_tokens": float(offloaded.sum().item()),
        "summary_metadata_bytes": float(summary_metadata["metadata_bytes"]),
        "summary_metadata_source": metadata_source,
    }


def block_summary_corrected_attention_mass(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    attention_probs: torch.Tensor,
    segment_mask: torch.Tensor,
    active_keep_mask: torch.Tensor,
    attn_mask: torch.Tensor | None,
    block_size: int = 8,
    representatives: int = 4,
    calibration_factor: float = 1.0,
    summary_metadata: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Correct one segment when it is the only offloaded partition."""

    active_mass, partition_ratio, stats = block_summary_partition_components(
        q=q,
        k=k,
        attention_probs=attention_probs,
        segment_mask=segment_mask,
        active_keep_mask=active_keep_mask,
        attn_mask=attn_mask,
        block_size=block_size,
        representatives=representatives,
        calibration_factor=calibration_factor,
        summary_metadata=summary_metadata,
    )
    corrected = (active_mass + partition_ratio) / (1.0 + partition_ratio)
    stats = dict(stats)
    stats["corrected_attention_mass"] = float(corrected.mean().item())
    return stats["corrected_attention_mass"], stats


def chunk_indices(
    *,
    layer_idx: int,
    step_index: int | None,
    decode_index: int | None,
    layer_chunk_size: int,
    step_chunk_size: int,
    first_layer_separate: bool = False,
    first_step_separate: bool = False,
) -> tuple[int, int]:
    if layer_chunk_size <= 0 or step_chunk_size <= 0:
        raise ValueError("layer_chunk_size and step_chunk_size must be positive")
    progress_index = step_index if step_index is not None else decode_index
    layer = int(layer_idx)
    if first_layer_separate:
        layer_chunk = 0 if layer == 0 else 1 + (layer - 1) // layer_chunk_size
    else:
        layer_chunk = layer // layer_chunk_size
    progress = int(progress_index or 0)
    if first_step_separate:
        step_chunk = 0 if progress == 0 else 1 + (progress - 1) // step_chunk_size
    else:
        step_chunk = progress // step_chunk_size
    return layer_chunk, step_chunk


def block_metric_key(
    *,
    run_id: str,
    phase: str,
    branch: str | None,
    cfg_branch: str,
    batch_idx: int,
    segment_id: str,
    layer_chunk: int,
    step_chunk: int,
) -> tuple[Any, ...]:
    return (
        str(run_id),
        str(phase),
        branch,
        str(cfg_branch),
        int(batch_idx),
        str(segment_id),
        int(layer_chunk),
        int(step_chunk),
    )


def block_metric_stream_key(
    *,
    run_id: str,
    phase: str,
    branch: str | None,
    cfg_branch: str,
    batch_idx: int,
    segment_id: str,
    layer_chunk: int,
) -> tuple[Any, ...]:
    """Key an EMA stream that persists across step-chunk boundaries."""

    return (
        "ema_stream",
        str(run_id),
        str(phase),
        branch,
        str(cfg_branch),
        int(batch_idx),
        str(segment_id),
        int(layer_chunk),
    )
