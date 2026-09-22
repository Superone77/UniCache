"""CUDA dispatch for the pinned KIVI packed QK/AV kernels.

Set ``UNICACHE_KIVI_ROOT`` to the KIVI checkout at commit
876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6 and install its ``quant`` extension.
Physical CUDA configs fail closed when that kernel is unavailable.
"""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING, Callable

import torch

from .physical import gqa_probability_value, gqa_query_key_logits

if TYPE_CHECKING:
    from .physical import PackedKIVICache


KIVI_COMMIT = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"


def select_kivi_kernel_backend(query_length: int) -> str:
    """Select GEMV for decode and a tiled kernel for multi-query attention."""

    requested = os.environ.get("UNICACHE_KIVI_MULTI_QUERY_BACKEND", "auto").lower()
    if requested not in {"auto", "official_gemv", "triton_tiled"}:
        raise ValueError(f"Unknown KIVI multi-query backend: {requested}")
    if requested != "auto":
        return requested
    return "official_gemv" if int(query_length) == 1 else "triton_tiled"


def _load_tiled_kernels():
    try:
        from .kivi_triton import (
            dequantize_kv_chunk_tiled,
            packed_attention_tiled,
            packed_av_tiled,
            packed_qk_tiled,
        )
    except Exception as exc:  # pragma: no cover - exercised on Raven
        raise RuntimeError(
            "Multi-query physical KIVI requires Triton. Install it in the CUDA environment "
            "or set UNICACHE_KIVI_MULTI_QUERY_BACKEND=official_gemv for diagnosis."
        ) from exc
    return (
        packed_qk_tiled,
        packed_av_tiled,
        packed_attention_tiled,
        dequantize_kv_chunk_tiled,
    )


def dequantize_kivi_chunk(
    cache: "PackedKIVICache",
    *,
    start: int,
    stop: int,
    dtype: torch.dtype,
    output_key: torch.Tensor | None = None,
    output_value: torch.Tensor | None = None,
    output_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore one bounded tile without creating a full BF16 cache replica."""

    start = max(0, int(start))
    stop = min(int(stop), int(cache.quantized_tokens))
    if stop <= start:
        empty = cache.residual_key[:0].to(dtype)
        return empty, empty
    if not cache.kernel_native_layout or cache.k_bits != cache.v_bits:
        raise RuntimeError("Chunked KIVI dequantization requires native equal-bit KV")
    _, _, _, dequantize_kv_chunk_tiled = _load_tiled_kernels()
    return dequantize_kv_chunk_tiled(
        cache.key_code,
        cache.key_scale,
        cache.key_minimum,
        cache.value_code,
        cache.value_scale,
        cache.value_minimum,
        start=start,
        stop=stop,
        bits=cache.k_bits,
        group_size=cache.group_size,
        dtype=dtype,
        output_key=output_key,
        output_value=output_value,
        output_offset=output_offset,
    )


def packed_kivi_quantized_attention_summary(
    cache: "PackedKIVICache", query: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized output and LSE for the packed quantized prefix."""

    if not cache.quantized_tokens:
        raise ValueError("Packed KIVI summary requires a non-empty quantized prefix")
    if cache.k_bits != cache.v_bits or cache.k_bits not in {2, 4}:
        raise ValueError("Packed KIVI fused summary supports equal 2-bit or 4-bit KV")
    if not cache.kernel_native_layout:
        raise RuntimeError("Packed KIVI fused summary requires kernel-native layout")
    _, _, packed_attention_tiled, _ = _load_tiled_kernels()
    q = query.transpose(0, 1).contiguous().to(torch.float16)
    output, logsumexp = packed_attention_tiled(
        q,
        cache.key_code,
        cache.key_scale,
        cache.key_minimum,
        cache.value_code,
        cache.value_scale,
        cache.value_minimum,
        bits=cache.k_bits,
        group_size=cache.group_size,
    )
    return output.transpose(0, 1).to(query.dtype), logsumexp


def _load_official_kernel():
    root = os.environ.get("UNICACHE_KIVI_ROOT")
    if root:
        root = os.path.abspath(root)
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        from quant.matmul import cuda_bmm_fA_qB_outer
    except Exception as exc:  # pragma: no cover - exercised on Raven
        raise RuntimeError(
            "Physical KIVI requires the official KIVI CUDA extension. "
            "Set UNICACHE_KIVI_ROOT to the pinned checkout and install quant/."
        ) from exc
    return cuda_bmm_fA_qB_outer


def _load_official_packer():
    """Load KIVI's fused Triton quantize-and-pack implementation."""

    root = os.environ.get("UNICACHE_KIVI_ROOT")
    if root:
        root = os.path.abspath(root)
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        from quant.new_pack import triton_quantize_and_pack_along_last_dim
    except Exception as exc:  # pragma: no cover - exercised on Raven
        raise RuntimeError(
            "Physical KIVI materialization requires the official KIVI Triton packer. "
            "Set UNICACHE_KIVI_ROOT to the pinned checkout."
        ) from exc
    return triton_quantize_and_pack_along_last_dim


def quantize_kivi_native(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    k_bits: int,
    v_bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create KIVI's kernel-native packed layout without an intermediate code tensor.

    KIVI groups Key along the token dimension and Value along head_dim.  Its
    official packer operates on the final dimension, so the two transposes
    below express exactly those two layouts.
    """

    if key.shape != value.shape or key.dim() != 3:
        raise ValueError("KIVI native packing expects matching [tokens, heads, dim] KV")
    if key.device.type != "cuda":
        raise RuntimeError("KIVI native packing requires CUDA tensors")
    pack = _load_official_packer()
    key_native = key.permute(1, 2, 0).unsqueeze(0).contiguous()
    value_native = value.permute(1, 0, 2).unsqueeze(0).contiguous()
    key_code, key_scale, key_minimum = pack(key_native, group_size, k_bits)
    value_code, value_scale, value_minimum = pack(
        value_native, group_size, v_bits
    )
    return (
        key_code.squeeze(0),
        key_scale.squeeze(0).to(torch.float16),
        key_minimum.squeeze(0).to(torch.float16),
        value_code.squeeze(0),
        value_scale.squeeze(0).to(torch.float16),
        value_minimum.squeeze(0).to(torch.float16),
    )


def _uint8_qk(cache: "PackedKIVICache", query: torch.Tensor) -> torch.Tensor:
    outputs = []
    repeat = int(query.shape[1]) // cache.num_kv_heads
    for start in range(0, cache.quantized_tokens, cache.group_size):
        stop = min(cache.quantized_tokens, start + cache.group_size)
        code = cache.key_code[start:stop].float()
        group_index = start // cache.group_size
        scale = cache.key_scale[group_index].unsqueeze(0).float()
        minimum = cache.key_minimum[group_index].unsqueeze(0).float()
        key = (code * scale + minimum).to(query.dtype).repeat_interleave(repeat, dim=1)
        outputs.append(
            torch.matmul(
                query.transpose(0, 1).float(),
                key.transpose(0, 1).float().transpose(-1, -2),
            )
        )
    return torch.cat(outputs, dim=-1) if outputs else query.new_empty(
        (query.shape[1], query.shape[0], 0), dtype=torch.float32
    )


def _uint8_av(cache: "PackedKIVICache", probabilities: torch.Tensor) -> torch.Tensor:
    repeat = int(probabilities.shape[0]) // cache.num_kv_heads
    output = None
    for start in range(0, cache.quantized_tokens, cache.group_size):
        stop = min(cache.quantized_tokens, start + cache.group_size)
        code = cache.value_code[start:stop].float()
        pieces = []
        for group_index, dim_start in enumerate(range(0, cache.head_dim, cache.group_size)):
            dim_stop = min(cache.head_dim, dim_start + cache.group_size)
            scale = cache.value_scale[start:stop, :, group_index].unsqueeze(-1).float()
            minimum = cache.value_minimum[start:stop, :, group_index].unsqueeze(-1).float()
            pieces.append(code[:, :, dim_start:dim_stop] * scale + minimum)
        value = torch.cat(pieces, dim=-1).to(probabilities.dtype).repeat_interleave(repeat, dim=1)
        contribution = torch.matmul(
            probabilities[:, :, start:stop].to(value.dtype), value.transpose(0, 1)
        )
        output = contribution if output is None else output + contribution
    if output is None:
        output = probabilities.new_zeros(
            (probabilities.shape[0], probabilities.shape[1], cache.head_dim)
        )
    return output


def packed_kivi_attention_parts(
    cache: "PackedKIVICache",
    query: torch.Tensor,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    if query.device.type != "cuda":
        raise RuntimeError("Physical KIVI CUDA backend received a non-CUDA query")
    if cache.k_bits != cache.v_bits:
        raise ValueError("Official KIVI kernel path requires equal key/value bit widths")
    repeat = int(query.shape[1]) // cache.num_kv_heads
    if repeat * cache.num_kv_heads != int(query.shape[1]):
        raise ValueError("Query heads must be divisible by KIVI KV heads")

    quantized_logits = None
    if cache.quantized_tokens:
        if cache.k_bits in {2, 4}:
            if not cache.kernel_native_layout:
                raise RuntimeError(
                    "Physical KIVI CUDA cache is not stored in kernel-native layout"
                )
            backend = select_kivi_kernel_backend(query.shape[0])
            if backend == "official_gemv":
                kernel = _load_official_kernel()
                q = query.transpose(0, 1).unsqueeze(0).to(torch.float16)
                code = cache.key_code.unsqueeze(0)
                scale = cache.key_scale.unsqueeze(0)
                minimum = cache.key_minimum.unsqueeze(0)
                quantized_logits = kernel(
                    cache.group_size, q, code, scale, minimum, cache.k_bits
                ).squeeze(0).float()
            else:
                packed_qk_tiled, _, _, _ = _load_tiled_kernels()
                q = query.transpose(0, 1).contiguous().to(torch.float16)
                quantized_logits = packed_qk_tiled(
                    q,
                    cache.key_code,
                    cache.key_scale,
                    cache.key_minimum,
                    bits=cache.k_bits,
                    group_size=cache.group_size,
                ).float()
        elif cache.k_bits == 8:
            quantized_logits = _uint8_qk(cache, query)
        else:  # pragma: no cover - constructor validates this
            raise ValueError(f"Unsupported physical KIVI bits: {cache.k_bits}")

    residual_logits = gqa_query_key_logits(query, cache.residual_key)
    logits = residual_logits if quantized_logits is None else torch.cat(
        [quantized_logits, residual_logits], dim=-1
    )
    logits = logits / math.sqrt(query.shape[-1])

    def apply_value(probabilities: torch.Tensor) -> torch.Tensor:
        quant_probs = probabilities[:, :, : cache.quantized_tokens]
        residual_probs = probabilities[:, :, cache.quantized_tokens :]
        if cache.quantized_tokens:
            if cache.v_bits in {2, 4}:
                backend = select_kivi_kernel_backend(query.shape[0])
                if backend == "official_gemv":
                    kernel = _load_official_kernel()
                    code = cache.value_code.unsqueeze(0)
                    scale = cache.value_scale.unsqueeze(0)
                    minimum = cache.value_minimum.unsqueeze(0)
                    quant_out = kernel(
                        cache.group_size,
                        quant_probs.unsqueeze(0).to(torch.float16),
                        code,
                        scale,
                        minimum,
                        cache.v_bits,
                    ).squeeze(0)
                else:
                    _, packed_av_tiled, _, _ = _load_tiled_kernels()
                    quant_out = packed_av_tiled(
                        quant_probs.contiguous().to(torch.float16),
                        cache.value_code,
                        cache.value_scale,
                        cache.value_minimum,
                        bits=cache.v_bits,
                        group_size=cache.group_size,
                    )
            else:
                quant_out = _uint8_av(cache, quant_probs)
        else:
            quant_out = probabilities.new_zeros(
                (probabilities.shape[0], probabilities.shape[1], cache.head_dim)
            )
        residual_out = gqa_probability_value(
            residual_probs, cache.residual_value
        )
        return quant_out.to(residual_out.dtype) + residual_out

    return logits, apply_value
