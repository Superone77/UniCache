"""Tiled multi-query kernels for KIVI's packed asymmetric KV layout."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_qk_kernel(
    query,
    code,
    scale,
    minimum,
    output,
    m_size,
    n_size,
    d_size,
    q_heads,
    kv_heads,
    q_stride_h,
    q_stride_m,
    q_stride_d,
    code_stride_h,
    code_stride_d,
    code_stride_n,
    stat_stride_h,
    stat_stride_d,
    stat_stride_g,
    out_stride_h,
    out_stride_m,
    out_stride_n,
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_h = tl.program_id(1)
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(n_size, BLOCK_N)
    pid_m = pid // n_blocks
    pid_n = pid - pid_m * n_blocks
    kv_h = pid_h * kv_heads // q_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    value_mask = (1 << BITS) - 1
    values_per_word = 32 // BITS

    for d_start in range(0, d_size, BLOCK_D):
        d = d_start + offs_d
        q_ptrs = (
            query
            + pid_h * q_stride_h
            + offs_m[:, None] * q_stride_m
            + d[None, :] * q_stride_d
        )
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < m_size) & (d[None, :] < d_size),
            other=0.0,
        )
        packed_ptrs = (
            code
            + kv_h * code_stride_h
            + d[:, None] * code_stride_d
            + (offs_n[None, :] // values_per_word) * code_stride_n
        )
        packed = tl.load(
            packed_ptrs,
            mask=(d[:, None] < d_size) & (offs_n[None, :] < n_size),
            other=0,
        )
        shifts = (offs_n % values_per_word) * BITS
        quantized = (packed >> shifts[None, :]) & value_mask
        group_index = (pid_n * BLOCK_N) // GROUP_SIZE
        stat_ptrs = (
            kv_h * stat_stride_h
            + d * stat_stride_d
            + group_index * stat_stride_g
        )
        scales = tl.load(
            scale + stat_ptrs,
            mask=d < d_size,
            other=0.0,
        )
        minima = tl.load(
            minimum + stat_ptrs,
            mask=d < d_size,
            other=0.0,
        )
        dequantized = quantized.to(tl.float32) * scales[:, None] + minima[:, None]
        accumulator += tl.dot(q, dequantized.to(tl.float16))

    out_ptrs = (
        output
        + pid_h * out_stride_h
        + offs_m[:, None] * out_stride_m
        + offs_n[None, :] * out_stride_n
    )
    tl.store(
        out_ptrs,
        accumulator,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
    )


@triton.jit
def _packed_av_kernel(
    probabilities,
    code,
    scale,
    minimum,
    output,
    m_size,
    n_size,
    d_size,
    q_heads,
    kv_heads,
    p_stride_h,
    p_stride_m,
    p_stride_n,
    code_stride_h,
    code_stride_n,
    code_stride_d,
    stat_stride_h,
    stat_stride_n,
    stat_stride_g,
    out_stride_h,
    out_stride_m,
    out_stride_d,
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_h = tl.program_id(1)
    pid = tl.program_id(0)
    d_blocks = tl.cdiv(d_size, BLOCK_D)
    pid_m = pid // d_blocks
    pid_d = pid - pid_m * d_blocks
    kv_h = pid_h * kv_heads // q_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    value_mask = (1 << BITS) - 1
    values_per_word = 32 // BITS

    for n_start in range(0, n_size, BLOCK_N):
        n = n_start + offs_n
        p_ptrs = (
            probabilities
            + pid_h * p_stride_h
            + offs_m[:, None] * p_stride_m
            + n[None, :] * p_stride_n
        )
        probs = tl.load(
            p_ptrs,
            mask=(offs_m[:, None] < m_size) & (n[None, :] < n_size),
            other=0.0,
        )
        packed_ptrs = (
            code
            + kv_h * code_stride_h
            + n[:, None] * code_stride_n
            + (offs_d[None, :] // values_per_word) * code_stride_d
        )
        packed = tl.load(
            packed_ptrs,
            mask=(n[:, None] < n_size) & (offs_d[None, :] < d_size),
            other=0,
        )
        shifts = (offs_d % values_per_word) * BITS
        quantized = (packed >> shifts[None, :]) & value_mask
        group_index = (pid_d * BLOCK_D) // GROUP_SIZE
        stat_ptrs = (
            kv_h * stat_stride_h
            + n * stat_stride_n
            + group_index * stat_stride_g
        )
        scales = tl.load(
            scale + stat_ptrs,
            mask=n < n_size,
            other=0.0,
        )
        minima = tl.load(
            minimum + stat_ptrs,
            mask=n < n_size,
            other=0.0,
        )
        dequantized = quantized.to(tl.float32) * scales[:, None] + minima[:, None]
        accumulator += tl.dot(probs, dequantized.to(tl.float16))

    out_ptrs = (
        output
        + pid_h * out_stride_h
        + offs_m[:, None] * out_stride_m
        + offs_d[None, :] * out_stride_d
    )
    tl.store(
        out_ptrs,
        accumulator,
        mask=(offs_m[:, None] < m_size) & (offs_d[None, :] < d_size),
    )


@triton.jit
def _packed_attention_kernel(
    query,
    key_code,
    key_scale,
    key_minimum,
    value_code,
    value_scale,
    value_minimum,
    output,
    logsumexp,
    m_size,
    n_size,
    d_size,
    q_heads,
    kv_heads,
    q_stride_h,
    q_stride_m,
    q_stride_d,
    kc_stride_h,
    kc_stride_d,
    kc_stride_n,
    ks_stride_h,
    ks_stride_d,
    ks_stride_g,
    vc_stride_h,
    vc_stride_n,
    vc_stride_d,
    vs_stride_h,
    vs_stride_n,
    vs_stride_g,
    out_stride_h,
    out_stride_m,
    out_stride_d,
    lse_stride_h,
    lse_stride_m,
    SM_SCALE: tl.constexpr,
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h * kv_heads // q_heads
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (
        query
        + pid_h * q_stride_h
        + offs_m[:, None] * q_stride_m
        + offs_d[None, :] * q_stride_d
    )
    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < m_size) & (offs_d[None, :] < d_size),
        other=0.0,
    )
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    code_mask = (1 << BITS) - 1
    values_per_word = 32 // BITS

    for n_start in range(0, n_size, BLOCK_N):
        n = n_start + offs_n
        packed_key = tl.load(
            key_code
            + kv_h * kc_stride_h
            + offs_d[:, None] * kc_stride_d
            + (n[None, :] // values_per_word) * kc_stride_n,
            mask=(offs_d[:, None] < d_size) & (n[None, :] < n_size),
            other=0,
        )
        key_quantized = (
            packed_key >> (((n % values_per_word) * BITS)[None, :])
        ) & code_mask
        key_group = n_start // GROUP_SIZE
        key_stats = (
            kv_h * ks_stride_h
            + offs_d * ks_stride_d
            + key_group * ks_stride_g
        )
        key_scales = tl.load(
            key_scale + key_stats, mask=offs_d < d_size, other=0.0
        )
        key_minima = tl.load(
            key_minimum + key_stats, mask=offs_d < d_size, other=0.0
        )
        key = key_quantized.to(tl.float32) * key_scales[:, None] + key_minima[:, None]
        scores = tl.dot(q, key.to(tl.float16)) * SM_SCALE
        scores = tl.where(n[None, :] < n_size, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        next_max = tl.maximum(row_max, tile_max)
        correction = tl.exp(row_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        next_sum = row_sum * correction + tl.sum(probabilities, axis=1)

        packed_value = tl.load(
            value_code
            + kv_h * vc_stride_h
            + n[:, None] * vc_stride_n
            + (offs_d[None, :] // values_per_word) * vc_stride_d,
            mask=(n[:, None] < n_size) & (offs_d[None, :] < d_size),
            other=0,
        )
        value_quantized = (
            packed_value >> (((offs_d % values_per_word) * BITS)[None, :])
        ) & code_mask
        if GROUP_SIZE == BLOCK_D:
            value_stats = kv_h * vs_stride_h + n * vs_stride_n
            value_scales = tl.load(
                value_scale + value_stats, mask=n < n_size, other=0.0
            )
            value_minima = tl.load(
                value_minimum + value_stats, mask=n < n_size, other=0.0
            )
            value = (
                value_quantized.to(tl.float32) * value_scales[:, None]
                + value_minima[:, None]
            ).to(tl.float16)
        else:
            value_stats = (
                kv_h * vs_stride_h
                + n[:, None] * vs_stride_n
                + (offs_d[None, :] // GROUP_SIZE) * vs_stride_g
            )
            value_scales = tl.load(
                value_scale + value_stats,
                mask=(n[:, None] < n_size) & (offs_d[None, :] < d_size),
                other=0.0,
            )
            value_minima = tl.load(
                value_minimum + value_stats,
                mask=(n[:, None] < n_size) & (offs_d[None, :] < d_size),
                other=0.0,
            )
            value = (
                value_quantized.to(tl.float32) * value_scales + value_minima
            ).to(tl.float16)
        accumulator = (
            accumulator * correction[:, None]
            + tl.dot(probabilities.to(tl.float16), value)
        )
        row_max = next_max
        row_sum = next_sum

    normalized = accumulator / row_sum[:, None]
    out_ptrs = (
        output
        + pid_h * out_stride_h
        + offs_m[:, None] * out_stride_m
        + offs_d[None, :] * out_stride_d
    )
    tl.store(
        out_ptrs,
        normalized,
        mask=(offs_m[:, None] < m_size) & (offs_d[None, :] < d_size),
    )
    tl.store(
        logsumexp + pid_h * lse_stride_h + offs_m * lse_stride_m,
        row_max + tl.log(row_sum),
        mask=offs_m < m_size,
    )


@triton.jit
def _dequantize_kv_chunk_kernel(
    key_code,
    key_scale,
    key_minimum,
    value_code,
    value_scale,
    value_minimum,
    output_key,
    output_value,
    token_start,
    token_count,
    output_offset,
    d_size,
    kc_stride_h,
    kc_stride_d,
    kc_stride_n,
    ks_stride_h,
    ks_stride_d,
    ks_stride_g,
    vc_stride_h,
    vc_stride_n,
    vc_stride_d,
    vs_stride_h,
    vs_stride_n,
    vs_stride_g,
    out_stride_n,
    out_stride_h,
    out_stride_d,
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n = token_start + local_n
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    values_per_word = 32 // BITS
    code_mask = (1 << BITS) - 1

    packed_key = tl.load(
        key_code
        + pid_h * kc_stride_h
        + d[:, None] * kc_stride_d
        + (n[None, :] // values_per_word) * kc_stride_n,
        mask=(d[:, None] < d_size) & (local_n[None, :] < token_count),
        other=0,
    )
    key_quantized = (
        packed_key >> (((n % values_per_word) * BITS)[None, :])
    ) & code_mask
    key_stats = (
        pid_h * ks_stride_h
        + d[:, None] * ks_stride_d
        + (n[None, :] // GROUP_SIZE) * ks_stride_g
    )
    key = (
        key_quantized.to(tl.float32)
        * tl.load(
            key_scale + key_stats,
            mask=(d[:, None] < d_size) & (local_n[None, :] < token_count),
            other=0.0,
        )
        + tl.load(
            key_minimum + key_stats,
            mask=(d[:, None] < d_size) & (local_n[None, :] < token_count),
            other=0.0,
        )
    )

    packed_value = tl.load(
        value_code
        + pid_h * vc_stride_h
        + n[:, None] * vc_stride_n
        + (d[None, :] // values_per_word) * vc_stride_d,
        mask=(local_n[:, None] < token_count) & (d[None, :] < d_size),
        other=0,
    )
    value_quantized = (
        packed_value >> (((d % values_per_word) * BITS)[None, :])
    ) & code_mask
    value_stats = (
        pid_h * vs_stride_h
        + n[:, None] * vs_stride_n
        + (d[None, :] // GROUP_SIZE) * vs_stride_g
    )
    value = (
        value_quantized.to(tl.float32)
        * tl.load(
            value_scale + value_stats,
            mask=(local_n[:, None] < token_count) & (d[None, :] < d_size),
            other=0.0,
        )
        + tl.load(
            value_minimum + value_stats,
            mask=(local_n[:, None] < token_count) & (d[None, :] < d_size),
            other=0.0,
        )
    )
    tl.store(
        output_key
        + (output_offset + local_n[None, :]) * out_stride_n
        + pid_h * out_stride_h
        + d[:, None] * out_stride_d,
        key,
        mask=(local_n[None, :] < token_count) & (d[:, None] < d_size),
    )
    tl.store(
        output_value
        + (output_offset + local_n[:, None]) * out_stride_n
        + pid_h * out_stride_h
        + d[None, :] * out_stride_d,
        value,
        mask=(local_n[:, None] < token_count) & (d[None, :] < d_size),
    )


def packed_qk_tiled(
    query: torch.Tensor,
    code: torch.Tensor,
    scale: torch.Tensor,
    minimum: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Compute QK from ``[kv_heads, dim, packed_tokens]`` Key storage."""

    if query.dim() != 3 or code.dim() != 3:
        raise ValueError("Tiled KIVI QK expects query [q_heads, q_len, dim]")
    q_heads, query_length, d_size = map(int, query.shape)
    kv_heads = int(code.shape[0])
    if q_heads % kv_heads:
        raise ValueError("Tiled KIVI QK requires query heads divisible by KV heads")
    queries_per_kv_head = q_heads // kv_heads
    grouped_query = query.view(kv_heads, queries_per_kv_head * query_length, d_size)
    m_size = int(grouped_query.shape[1])
    n_size = int(code.shape[-1]) * (32 // int(bits))
    output = torch.empty(
        (kv_heads, m_size, n_size), device=query.device, dtype=torch.float16
    )
    # A denoising query has hundreds of rows. A wider M tile amortizes packed
    # Key decode and scale/minimum loads across substantially more queries.
    block_m, block_n, block_d = 512, 16, 64
    grid = (triton.cdiv(m_size, block_m) * triton.cdiv(n_size, block_n), kv_heads)
    _packed_qk_kernel[grid](
        grouped_query,
        code,
        scale,
        minimum,
        output,
        m_size,
        n_size,
        d_size,
        kv_heads,
        kv_heads,
        *grouped_query.stride(),
        *code.stride(),
        *scale.stride(),
        *output.stride(),
        BITS=int(bits),
        GROUP_SIZE=int(group_size),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
        num_stages=3,
    )
    return output.view(q_heads, query_length, n_size)


def packed_av_tiled(
    probabilities: torch.Tensor,
    code: torch.Tensor,
    scale: torch.Tensor,
    minimum: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Compute AV from ``[kv_heads, tokens, packed_dim]`` Value storage."""

    if probabilities.dim() != 3 or code.dim() != 3:
        raise ValueError("Tiled KIVI AV expects probabilities [q_heads, q_len, tokens]")
    q_heads, query_length, n_size = map(int, probabilities.shape)
    kv_heads = int(code.shape[0])
    if q_heads % kv_heads:
        raise ValueError("Tiled KIVI AV requires query heads divisible by KV heads")
    queries_per_kv_head = q_heads // kv_heads
    grouped_probabilities = probabilities.view(
        kv_heads, queries_per_kv_head * query_length, n_size
    )
    m_size = int(grouped_probabilities.shape[1])
    d_size = int(code.shape[-1]) * (32 // int(bits))
    output = torch.empty(
        (kv_heads, m_size, d_size), device=probabilities.device, dtype=torch.float16
    )
    # Value codes are shared by every query row in the tile, so use the same
    # wider M tile and a 64-token reduction tile for denoising attention.
    block_m, block_d, block_n = 512, 16, 64
    grid = (triton.cdiv(m_size, block_m) * triton.cdiv(d_size, block_d), kv_heads)
    _packed_av_kernel[grid](
        grouped_probabilities,
        code,
        scale,
        minimum,
        output,
        m_size,
        n_size,
        d_size,
        kv_heads,
        kv_heads,
        *grouped_probabilities.stride(),
        *code.stride(),
        *scale.stride(),
        *output.stride(),
        BITS=int(bits),
        GROUP_SIZE=int(group_size),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=3,
    )
    return output.view(q_heads, query_length, d_size)


def packed_attention_tiled(
    query: torch.Tensor,
    key_code: torch.Tensor,
    key_scale: torch.Tensor,
    key_minimum: torch.Tensor,
    value_code: torch.Tensor,
    value_scale: torch.Tensor,
    value_minimum: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused online-softmax attention over a packed KIVI segment."""

    q_heads, query_length, d_size = map(int, query.shape)
    kv_heads = int(key_code.shape[0])
    if q_heads % kv_heads:
        raise ValueError("Packed KIVI attention requires query heads divisible by KV heads")
    n_size = int(key_code.shape[-1]) * (32 // int(bits))
    queries_per_kv_head = q_heads // kv_heads
    grouped_query = query.view(
        kv_heads, queries_per_kv_head * query_length, d_size
    )
    m_size = int(grouped_query.shape[1])
    grouped_output = torch.empty_like(grouped_query, dtype=torch.float16)
    grouped_logsumexp = torch.empty(
        (kv_heads, m_size), device=query.device, dtype=torch.float32
    )
    block_m = int(os.environ.get("UNICACHE_KIVI_PACKED_ATTN_BLOCK_M", "128"))
    block_n = int(os.environ.get("UNICACHE_KIVI_PACKED_ATTN_BLOCK_N", "64"))
    num_warps = int(os.environ.get("UNICACHE_KIVI_PACKED_ATTN_NUM_WARPS", "8"))
    num_stages = int(os.environ.get("UNICACHE_KIVI_PACKED_ATTN_NUM_STAGES", "1"))
    if block_m not in {64, 128, 256} or block_n not in {32, 64, 128}:
        raise ValueError("Packed KIVI attention blocks are unsupported")
    if num_warps not in {4, 8} or num_stages not in {1, 2, 3}:
        raise ValueError("Packed KIVI attention launch parameters are unsupported")
    block_d = 128
    grid = (triton.cdiv(m_size, block_m), kv_heads)
    _packed_attention_kernel[grid](
        grouped_query,
        key_code,
        key_scale,
        key_minimum,
        value_code,
        value_scale,
        value_minimum,
        grouped_output,
        grouped_logsumexp,
        m_size,
        n_size,
        d_size,
        kv_heads,
        kv_heads,
        *grouped_query.stride(),
        *key_code.stride(),
        *key_scale.stride(),
        *value_code.stride(),
        *value_scale.stride(),
        *grouped_output.stride(),
        *grouped_logsumexp.stride(),
        SM_SCALE=d_size**-0.5,
        BITS=int(bits),
        GROUP_SIZE=int(group_size),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        grouped_output.view(q_heads, query_length, d_size),
        grouped_logsumexp.view(q_heads, query_length),
    )


def dequantize_kv_chunk_tiled(
    key_code: torch.Tensor,
    key_scale: torch.Tensor,
    key_minimum: torch.Tensor,
    value_code: torch.Tensor,
    value_scale: torch.Tensor,
    value_minimum: torch.Tensor,
    *,
    start: int,
    stop: int,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
    output_key: torch.Tensor | None = None,
    output_value: torch.Tensor | None = None,
    output_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a bounded token tile from kernel-native KIVI storage."""

    token_count = int(stop) - int(start)
    kv_heads = int(key_code.shape[0])
    d_size = int(key_code.shape[1])
    output_offset = int(output_offset)
    if output_key is None or output_value is None:
        if output_key is not None or output_value is not None:
            raise ValueError("Both KIVI output buffers must be provided together")
        output_offset = 0
        output_key = torch.empty(
            (token_count, kv_heads, d_size), device=key_code.device, dtype=dtype
        )
        output_value = torch.empty_like(output_key)
    expected_tail = (kv_heads, d_size)
    if (
        output_key.dim() != 3
        or output_value.shape != output_key.shape
        or tuple(output_key.shape[1:]) != expected_tail
        or output_key.device != key_code.device
        or output_value.device != key_code.device
        or output_key.dtype != dtype
        or output_value.dtype != dtype
        or output_offset < 0
        or output_offset + token_count > int(output_key.shape[0])
    ):
        raise ValueError("KIVI dequantization output workspace has an invalid layout")
    block_n = int(os.environ.get("UNICACHE_KIVI_DEQUANT_BLOCK_N", "64"))
    block_d = int(os.environ.get("UNICACHE_KIVI_DEQUANT_BLOCK_D", "64"))
    num_warps = int(os.environ.get("UNICACHE_KIVI_DEQUANT_NUM_WARPS", "4"))
    if block_n not in {32, 64, 128} or block_d not in {32, 64, 128}:
        raise ValueError(
            "KIVI dequant blocks must be one of 32, 64, or 128"
        )
    if num_warps not in {4, 8}:
        raise ValueError("KIVI dequant num_warps must be 4 or 8")
    grid = (triton.cdiv(token_count, block_n), triton.cdiv(d_size, block_d), kv_heads)
    _dequantize_kv_chunk_kernel[grid](
        key_code,
        key_scale,
        key_minimum,
        value_code,
        value_scale,
        value_minimum,
        output_key,
        output_value,
        int(start),
        token_count,
        output_offset,
        d_size,
        *key_code.stride(),
        *key_scale.stride(),
        *value_code.stride(),
        *value_scale.stride(),
        *output_key.stride(),
        BITS=int(bits),
        GROUP_SIZE=int(group_size),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    chunk = slice(output_offset, output_offset + token_count)
    return output_key[chunk], output_value[chunk]
