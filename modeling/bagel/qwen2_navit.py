# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.


import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
from modeling.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    Qwen2MLP, 
    Qwen2PreTrainedModel, 
    Qwen2RMSNorm, 
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from modeling.qwen2.configuration_qwen2 import Qwen2Config as _Qwen2Config
from modeling.cache_utils.taylorseer import (
    cal_type, taylor_cache_init, derivative_approximation, taylor_formula,
)
from modeling.cache_utils.duca import (
    duca_layer_ready,
    duca_prepare_step,
    duca_record,
    duca_select_fresh,
    duca_store_full,
    duca_update_age,
)


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
# flex_attention = torch.compile(flex_attention) # , dynamic=True, mode='max-autotune'
flex_attention = torch.compile(flex_attention)


ATTENTION_RECORDER = None
TOPK_KV_POLICY = None

KV_TYPE_UNKNOWN = 0
KV_TYPE_INSTRUCTION = 1
KV_TYPE_SOURCE_VIT = 2
KV_TYPE_SOURCE_VAE = 3
KV_TYPE_CURRENT_VAE = 4
KV_TYPE_BOUNDARY = 5
KV_TYPE_DECODED_TEXT = 6

KV_TYPE_NAMES = {
    KV_TYPE_UNKNOWN: "unknown",
    KV_TYPE_INSTRUCTION: "instruction",
    KV_TYPE_SOURCE_VIT: "source_vit",
    KV_TYPE_SOURCE_VAE: "source_vae",
    KV_TYPE_CURRENT_VAE: "current_vae",
    KV_TYPE_BOUNDARY: "boundary",
    KV_TYPE_DECODED_TEXT: "decoded_text",
}


def set_attention_recorder(recorder):
    global ATTENTION_RECORDER
    ATTENTION_RECORDER = recorder


def set_topk_kv_policy(policy):
    global TOPK_KV_POLICY
    TOPK_KV_POLICY = policy


def clear_topk_kv_policy():
    set_topk_kv_policy(None)


def get_current_state_attention_scores(layer_idx, sample_idx=0):
    if TOPK_KV_POLICY is None:
        return None
    getter = getattr(TOPK_KV_POLICY, "get_current_state_attention_scores", None)
    if getter is None:
        return None
    return getter(layer_idx=layer_idx, sample_idx=sample_idx)


def get_current_state_runtime_params(layer_idx):
    if TOPK_KV_POLICY is None:
        return {}
    getter = getattr(TOPK_KV_POLICY, "current_state_runtime_params", None)
    if getter is None:
        return {}
    return getter(layer_idx=layer_idx)


def set_topk_kv_context(
    *,
    phase=None,
    step_index=None,
    total_steps=None,
    branch=None,
    cfg_branch=None,
    cfg_branches=None,
    decode_index=None,
    run_id=None,
):
    if TOPK_KV_POLICY is None:
        return
    setter = getattr(TOPK_KV_POLICY, "set_context", None)
    if setter is not None:
        setter(
            phase=phase,
            step_index=step_index,
            total_steps=total_steps,
            branch=branch,
            cfg_branch=cfg_branch,
            cfg_branches=cfg_branches,
            decode_index=decode_index,
            run_id=run_id,
        )


def _record_token_attention(
    layer_idx,
    packed_query_states,
    merged_key_states,
    cu_seqlens_q,
    cu_seqlens_k,
    causal,
):
    if ATTENTION_RECORDER is None or not getattr(ATTENTION_RECORDER, "enabled", False):
        return
    if layer_idx not in ATTENTION_RECORDER.layers:
        return

    cu_q = cu_seqlens_q.to("cpu").tolist()
    cu_k = cu_seqlens_k.to("cpu").tolist()
    scale = float(packed_query_states.shape[-1]) ** -0.5
    with torch.no_grad():
        for sample_idx in range(len(cu_q) - 1):
            q_i = packed_query_states[cu_q[sample_idx]:cu_q[sample_idx + 1]]
            k_i = merged_key_states[cu_k[sample_idx]:cu_k[sample_idx + 1]]
            if q_i.numel() == 0 or k_i.numel() == 0:
                continue
            query_labels = ATTENTION_RECORDER.query_labels
            selected_query_labels = query_labels
            query_types_to_record = getattr(ATTENTION_RECORDER, "query_types_to_record", None)
            if query_types_to_record and query_labels is not None and len(query_labels) == q_i.shape[0]:
                query_mask = torch.tensor(
                    [label in query_types_to_record for label in query_labels],
                    device=q_i.device,
                    dtype=torch.bool,
                )
                if not bool(query_mask.any()):
                    continue
                q_i = q_i[query_mask]
                selected_query_labels = [label for label, keep in zip(query_labels, query_mask.tolist()) if keep]
            if q_i.shape[1] != k_i.shape[1]:
                repeat = q_i.shape[1] // k_i.shape[1]
                k_i = k_i.repeat_interleave(repeat, dim=1)
            q_i = q_i.detach().float().transpose(0, 1)
            k_i = k_i.detach().float().transpose(0, 1)
            scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * scale
            if causal:
                attn_mask = _bottom_right_causal_mask(q_i.shape[1], k_i.shape[1], q_i.device)
                scores = scores.masked_fill(~attn_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            key_received = probs.mean(dim=(0, 1)).detach().to("cpu")
            ATTENTION_RECORDER.record(
                layer_idx=layer_idx,
                sample_idx=sample_idx,
                query_len=int(q_i.shape[1]),
                key_len=int(k_i.shape[1]),
                query_type="all",
                key_received=key_received,
            )
            record_block_recall = getattr(ATTENTION_RECORDER, "record_block_recall", None)
            if callable(record_block_recall):
                record_block_recall(
                    layer_idx=layer_idx,
                    sample_idx=sample_idx,
                    query_states=q_i,
                    key_states=k_i,
                    attention_probs=probs,
                )
            query_labels = selected_query_labels
            if query_labels is not None and len(query_labels) == q_i.shape[1]:
                for query_type in sorted(set(query_labels)):
                    mask = torch.tensor(
                        [label == query_type for label in query_labels],
                        device=probs.device,
                        dtype=torch.bool,
                    )
                    if not bool(mask.any()):
                        continue
                    grouped = probs[:, mask, :].mean(dim=(0, 1)).detach().to("cpu")
                    ATTENTION_RECORDER.record(
                        layer_idx=layer_idx,
                        sample_idx=sample_idx,
                        query_len=int(mask.sum().item()),
                        key_len=int(k_i.shape[1]),
                        query_type=query_type,
                        key_received=grouped,
                    )


class Qwen2Config(_Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        _attn_implementation="flash_attention_2",
        qk_norm=True,
        layer_module="Qwen2DecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}
        self.token_type_cache = {k: None for k in range(num_layers)}
        self.layer_sample_lens = {k: None for k in range(num_layers)}
        self.physical_cache = {k: None for k in range(num_layers)}
        self.layer_layout_cache = {}
        self.next_token_type_ids = None

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0

    def set_layer_lens(self, layer_idx, sample_lens):
        layer_idx = int(layer_idx)
        self.layer_sample_lens[layer_idx] = sample_lens.detach().to(dtype=torch.int32)
        stale = [key for key in self.layer_layout_cache if key[0] == layer_idx]
        for key in stale:
            self.layer_layout_cache.pop(key, None)

    def get_layer_lens(self, layer_idx, fallback=None):
        lengths = self.layer_sample_lens.get(int(layer_idx))
        return fallback if lengths is None else lengths

    def set_physical_segments(self, layer_idx, segments_by_sample):
        if segments_by_sample is None:
            return
        current = self.physical_cache.get(int(layer_idx))
        if current is None:
            self.physical_cache[int(layer_idx)] = segments_by_sample
            return
        if len(current) != len(segments_by_sample):
            raise ValueError("Physical cache sample count changed within a run")
        for existing, new_segments in zip(current, segments_by_sample):
            existing.extend(new_segments)

    def get_physical_segments(self, layer_idx):
        return self.physical_cache.get(int(layer_idx))

    def release_dense_storage(self):
        """Drop dense tensors after another cache becomes their packed owner."""

        for layer_idx in range(self.num_layers):
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
            token_types = self.token_type_cache[layer_idx]
            if key is not None:
                self.key_cache[layer_idx] = key.new_empty((0, *key.shape[1:]))
            if value is not None:
                self.value_cache[layer_idx] = value.new_empty((0, *value.shape[1:]))
            if token_types is not None:
                self.token_type_cache[layer_idx] = token_types.new_empty((0,))


def pack_naive_caches(caches):
    """Pack independent CFG caches as separate varlen-attention samples."""

    caches = tuple(caches)
    if not caches:
        raise ValueError("At least one cache is required for CFG packing")
    num_layers = caches[0].num_layers
    if any(cache.num_layers != num_layers for cache in caches):
        raise ValueError("CFG caches must have the same layer count")
    packed = NaiveCache(num_layers)
    for layer_idx in range(num_layers):
        keys = [cache.key_cache[layer_idx] for cache in caches]
        values = [cache.value_cache[layer_idx] for cache in caches]
        token_types = [cache.token_type_cache[layer_idx] for cache in caches]
        populated = [item is not None for item in keys]
        if any(populated) and not all(populated):
            raise ValueError("CFG caches must populate the same layers")
        if all(populated):
            if any(value is None for value in values):
                raise ValueError("CFG cache keys and values must be populated together")
            packed.key_cache[layer_idx] = torch.cat(keys, dim=0)
            packed.value_cache[layer_idx] = torch.cat(values, dim=0)
            if any(item is not None for item in token_types):
                if not all(item is not None for item in token_types):
                    raise ValueError("CFG token-type caches must be populated consistently")
                packed.token_type_cache[layer_idx] = torch.cat(token_types, dim=0)
        lengths = [cache.get_layer_lens(layer_idx) for cache in caches]
        if any(item is not None for item in lengths):
            if not all(item is not None for item in lengths):
                raise ValueError("CFG per-layer lengths must be populated consistently")
            packed.set_layer_lens(layer_idx, torch.cat(lengths, dim=0))
        physical = [cache.get_physical_segments(layer_idx) for cache in caches]
        if any(item is not None for item in physical):
            if not all(item is not None for item in physical):
                raise ValueError("CFG physical caches must be populated consistently")
            packed.set_physical_segments(
                layer_idx,
                [segments for branch in physical for segments in branch],
            )
    next_types = [cache.next_token_type_ids for cache in caches]
    if any(item is not None for item in next_types):
        if not all(item is not None for item in next_types):
            raise ValueError("CFG next-token types must be populated consistently")
        packed.next_token_type_ids = torch.cat(next_types, dim=0)
    return packed


@dataclass(frozen=True)
class LayerCacheLayout:
    key_value_lens: torch.Tensor
    packed_key_value_indexes: torch.Tensor
    packed_query_indexes: torch.Tensor


def resolve_layer_cache_layout(
    past_key_values,
    *,
    layer_idx,
    query_lens,
    fallback_key_value_lens,
    query_token_count=None,
):
    """Build packed indexes from the physical cache length of one layer."""

    stored_key_value_lens = past_key_values.get_layer_lens(layer_idx)
    key_value_lens = (
        fallback_key_value_lens
        if stored_key_value_lens is None
        else stored_key_value_lens
    )
    key_value_lens = key_value_lens.to(device=query_lens.device, dtype=query_lens.dtype)
    if key_value_lens.numel() != query_lens.numel():
        raise ValueError("Per-layer cache lengths must match the packed query batch")
    if query_token_count is None:
        query_token_count = int(query_lens.sum().item())
    cache_key = (
        int(layer_idx),
        int(query_lens.numel()),
        int(query_token_count),
        query_lens.device.type,
        query_lens.device.index,
        query_lens.dtype,
    )
    cache_reuse_enabled = TOPK_KV_POLICY is not None and getattr(
        TOPK_KV_POLICY, "enabled", False
    )
    if cache_reuse_enabled and stored_key_value_lens is not None:
        cached = past_key_values.layer_layout_cache.get(cache_key)
        if cached is not None:
            return cached
    packed_keys = []
    packed_queries = []
    offset = 0
    for key_len, query_len in zip(key_value_lens.to("cpu").tolist(), query_lens.to("cpu").tolist()):
        packed_keys.extend(range(offset, offset + int(key_len)))
        packed_queries.extend(range(offset + int(key_len), offset + int(key_len) + int(query_len)))
        offset += int(key_len) + int(query_len)
    layout = LayerCacheLayout(
        key_value_lens=key_value_lens,
        packed_key_value_indexes=torch.tensor(packed_keys, device=query_lens.device, dtype=torch.long),
        packed_query_indexes=torch.tensor(packed_queries, device=query_lens.device, dtype=torch.long),
    )
    if cache_reuse_enabled and stored_key_value_lens is not None:
        past_key_values.layer_layout_cache[cache_key] = layout
    return layout


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


def _bottom_right_causal_mask(query_len, key_len, device):
    query_positions = torch.arange(query_len, device=device)[:, None]
    key_positions = torch.arange(key_len, device=device)[None, :]
    return key_positions <= query_positions + key_len - query_len


def _topk_policy_enabled(layer_idx, mode):
    return (
        TOPK_KV_POLICY is not None
        and getattr(TOPK_KV_POLICY, "enabled", False)
        and getattr(TOPK_KV_POLICY, "phase", None)
        in {"prefill", "text_decode", "denoise", "understanding", "generation", "editing"}
        and (not hasattr(TOPK_KV_POLICY, "should_apply") or TOPK_KV_POLICY.should_apply(layer_idx, mode))
    )


def _topk_keep_ratio(layer_idx, mode):
    if TOPK_KV_POLICY is None:
        return 1.0
    getter = getattr(TOPK_KV_POLICY, "keep_ratio", None)
    if getter is None:
        return 1.0
    return float(getter(layer_idx, mode))


def _topk_candidate_budget(**kwargs):
    if TOPK_KV_POLICY is None:
        return None
    getter = getattr(TOPK_KV_POLICY, "candidate_budget", None)
    if getter is None:
        return None
    return getter(**kwargs)


def _topk_record(**kwargs):
    if TOPK_KV_POLICY is None:
        return
    recorder = getattr(TOPK_KV_POLICY, "record", None)
    if recorder is not None:
        recorder(**kwargs)


def _apply_topk_cache_update(*, k, v, key_type_ids, sample_lens, layer_idx, mode):
    if TOPK_KV_POLICY is None or not getattr(TOPK_KV_POLICY, "enabled", False):
        return None
    updater = getattr(TOPK_KV_POLICY, "apply_to_cache_update", None)
    if updater is None:
        return None
    has_pending = getattr(TOPK_KV_POLICY, "has_pending_cache_update", None)
    if has_pending is not None and not has_pending(layer_idx=int(layer_idx), mode=mode or ""):
        return None
    return updater(
        k=k,
        v=v,
        key_type_ids=key_type_ids,
        sample_lens=sample_lens,
        layer_idx=int(layer_idx),
        mode=mode or "",
    )


def varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal,
    layer_idx=None,
    mode=None,
    protected_key_indexes=None,
    key_type_ids=None,
    physical_cache_by_sample=None,
):
    use_topk = layer_idx is not None and _topk_policy_enabled(layer_idx, mode)
    if flash_attn_varlen_func is not None and q.device.type == "cuda" and not use_topk:
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

    batch_static_apply = (
        getattr(TOPK_KV_POLICY, "apply_static_physical_attention_batch", None)
        if use_topk
        else None
    )
    if batch_static_apply is not None and physical_cache_by_sample:
        batch_output = batch_static_apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            layer_idx=int(layer_idx),
            mode=mode or "",
            causal=causal,
            physical_cache_by_sample=physical_cache_by_sample,
        )
        if batch_output is not None:
            return batch_output

    outputs = []
    if int(cu_seqlens_q.numel()) == 2 and int(cu_seqlens_k.numel()) == 2:
        sample_spans = [(0, int(q.shape[0]), 0, int(k.shape[0]))]
    else:
        cu_q = cu_seqlens_q.to("cpu").tolist()
        cu_k = cu_seqlens_k.to("cpu").tolist()
        sample_spans = [
            (cu_q[idx], cu_q[idx + 1], cu_k[idx], cu_k[idx + 1])
            for idx in range(len(cu_q) - 1)
        ]
    for idx, (q_start, q_end, k_start, k_end) in enumerate(sample_spans):
        q_i = q[q_start:q_end]
        k_i = k[k_start:k_end]
        v_i = v[k_start:k_end]
        sample_physical_segments = (
            physical_cache_by_sample[idx]
            if physical_cache_by_sample is not None
            else None
        )
        static_physical_apply = (
            getattr(TOPK_KV_POLICY, "apply_static_physical_attention", None)
            if use_topk
            else None
        )
        if static_physical_apply is not None and sample_physical_segments:
            static_output = static_physical_apply(
                q_i=q_i,
                k_i=k_i,
                v_i=v_i,
                layer_idx=int(layer_idx),
                mode=mode or "",
                sample_idx=int(idx),
                causal=causal,
                physical_segments=sample_physical_segments,
            )
            if static_output is not None:
                outputs.append(static_output)
                continue
        storage_num_kv_heads = int(k_i.shape[1])

        type_aware_apply = (
            getattr(TOPK_KV_POLICY, "apply_to_attention", None)
            if use_topk
            else None
        )
        native_gqa = bool(
            type_aware_apply is not None
            and getattr(TOPK_KV_POLICY, "accepts_native_gqa", False)
        )
        if q_i.shape[1] != k_i.shape[1] and not native_gqa:
            repeat = q_i.shape[1] // k_i.shape[1]
            k_i = k_i.repeat_interleave(repeat, dim=1)
            v_i = v_i.repeat_interleave(repeat, dim=1)

        attn_mask = None
        if causal and int(q_i.shape[0]) != 1:
            attn_mask = _bottom_right_causal_mask(q_i.shape[0], k_i.shape[0], q_i.device)

        if use_topk:
            key_len = int(k_i.shape[0])
            protected_mask = torch.zeros(key_len, device=k_i.device, dtype=torch.bool)
            if protected_key_indexes is not None:
                protected = protected_key_indexes.to(device=k_i.device)
                start = int(k_start)
                end = int(k_end)
                protected = protected[(protected >= start) & (protected < end)] - start
                if protected.numel() > 0:
                    protected_mask[protected.long()] = True
            extra_protected = getattr(TOPK_KV_POLICY, "extra_protected_key_mask", None)
            if extra_protected is not None:
                extra_mask = extra_protected(
                    key_len=key_len,
                    protected_mask=protected_mask,
                    layer_idx=layer_idx,
                    mode=mode,
                    device=k_i.device,
                )
                if extra_mask is not None:
                    protected_mask |= extra_mask.to(device=k_i.device, dtype=torch.bool)

            sample_key_type_ids = None
            if key_type_ids is not None:
                sample_key_type_ids = key_type_ids[k_start:k_end].to(device=k_i.device)
            if type_aware_apply is not None:
                type_aware_kwargs = dict(
                    q_i=q_i,
                    k_i=k_i,
                    v_i=v_i,
                    attn_mask=attn_mask,
                    key_type_ids=sample_key_type_ids,
                    protected_mask=protected_mask,
                    layer_idx=int(layer_idx),
                    mode=mode or "",
                    sample_idx=int(idx),
                    causal=causal,
                )
                if getattr(TOPK_KV_POLICY, "accepts_storage_kv_metadata", False):
                    type_aware_kwargs["storage_num_kv_heads"] = storage_num_kv_heads
                if getattr(TOPK_KV_POLICY, "accepts_physical_cache_metadata", False):
                    type_aware_kwargs["physical_segments"] = sample_physical_segments
                out_i = type_aware_apply(**type_aware_kwargs)
                outputs.append(out_i)
                continue

            q_h = q_i.transpose(0, 1).float()
            k_h = k_i.transpose(0, 1).float()
            scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(q_i.shape[-1])
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask.unsqueeze(0), torch.finfo(scores.dtype).min)

            score_query_count = int(getattr(TOPK_KV_POLICY, "score_query_count", 0) or 0)
            if score_query_count > 0 and score_query_count < scores.shape[-2]:
                score_logits = scores[:, -score_query_count:, :]
            else:
                score_logits = scores
            probs = torch.softmax(score_logits, dim=-1)
            key_scores = probs.mean(dim=(0, 1))
            kernel_size = int(getattr(TOPK_KV_POLICY, "kernel_size", 1) or 1)
            pooling = getattr(TOPK_KV_POLICY, "pooling", "none")
            if kernel_size > 1:
                pooled = key_scores[None, None, :]
                if pooling == "avgpool":
                    pooled = torch.nn.functional.avg_pool1d(
                        pooled,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        stride=1,
                    )
                elif pooling == "maxpool":
                    pooled = torch.nn.functional.max_pool1d(
                        pooled,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        stride=1,
                    )
                key_scores = pooled[0, 0, : key_scores.shape[0]]

            candidate_mask = ~protected_mask
            candidate_len = int(candidate_mask.sum().item())
            keep_ratio = max(0.0, min(1.0, _topk_keep_ratio(layer_idx, mode)))
            min_keep = int(getattr(TOPK_KV_POLICY, "min_keep", 8))
            keep_candidate_k = 0
            target_candidate_budget = 0
            if candidate_len > 0:
                budget = _topk_candidate_budget(
                    layer_idx=int(layer_idx),
                    mode=mode,
                    sample_idx=int(idx),
                    query_len=int(q_i.shape[0]),
                    key_len=key_len,
                    protected_key_len=int(protected_mask.sum().item()),
                    candidate_key_len=candidate_len,
                )
                if budget is None:
                    budget = int(math.ceil(candidate_len * keep_ratio))
                target_candidate_budget = int(budget)
                keep_candidate_k = min(candidate_len, max(min_keep, int(budget)))

            keep_mask = protected_mask.clone()
            if keep_candidate_k >= candidate_len:
                keep_mask[candidate_mask] = True
            elif keep_candidate_k > 0:
                candidate_scores = key_scores.masked_fill(~candidate_mask, torch.finfo(key_scores.dtype).min)
                topk_idx = torch.topk(candidate_scores, k=keep_candidate_k, largest=True).indices
                keep_mask[topk_idx] = True
            keep_k = int(keep_mask.sum().item())

            masked_scores = scores.masked_fill(~keep_mask[None, None, :], torch.finfo(scores.dtype).min)
            masked_probs = torch.softmax(masked_scores, dim=-1).to(v_i.dtype)
            out_i = torch.matmul(masked_probs, v_i.transpose(0, 1)).transpose(0, 1)
            _topk_record(
                layer_idx=int(layer_idx),
                mode=mode or "",
                sample_idx=int(idx),
                query_len=int(q_i.shape[0]),
                key_len=key_len,
                protected_key_len=int(protected_mask.sum().item()),
                candidate_key_len=candidate_len,
                keep_candidate_k=int(keep_candidate_k),
                keep_k=int(keep_k),
                keep_ratio=float(keep_k / max(key_len, 1)),
                candidate_keep_ratio=float(keep_candidate_k / max(candidate_len, 1)),
                target_keep_ratio=float(keep_ratio),
                target_candidate_budget=int(target_candidate_budget),
                skipped_tokens=int(candidate_len - keep_candidate_k),
            )
        else:
            out_i = scaled_dot_product_attention(
                q_i.transpose(0, 1).unsqueeze(0),
                k_i.transpose(0, 1).unsqueeze(0),
                v_i.transpose(0, 1).unsqueeze(0),
                attn_mask=attn_mask,
            )
            out_i = out_i.squeeze(0).transpose(0, 1)
        outputs.append(out_i)
    return torch.cat(outputs, dim=0)


class PackedAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, List):
            packed_key_states = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states = packed_key_states.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = pad_sequence(packed_query_states.permute(1, 0, 2), pad_size)
            packed_key_states = pad_sequence(packed_key_states.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states.unsqueeze(0), 
                packed_key_states.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)
        had_past_cache = past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None
        cache_input_lens = key_values_lens
        if had_past_cache:
            layer_layout = resolve_layer_cache_layout(
                past_key_values,
                layer_idx=self.layer_idx,
                query_lens=query_lens,
                fallback_key_value_lens=key_values_lens,
                query_token_count=packed_query_states.shape[0],
            )
            key_values_lens = layer_layout.key_value_lens
            cache_input_lens = key_values_lens
            packed_key_value_indexes = layer_layout.packed_key_value_indexes
            packed_query_indexes = layer_layout.packed_query_indexes
        query_token_type_ids = None
        if past_key_values is not None:
            query_token_type_ids = getattr(past_key_values, "next_token_type_ids", None)
            if query_token_type_ids is not None:
                query_token_type_ids = query_token_type_ids.to(device=packed_query_indexes.device, dtype=torch.long)
        merged_key_type_ids = None

        if had_past_cache:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            token_type_cache = getattr(past_key_values, "token_type_cache", None)
            if isinstance(token_type_cache, dict):
                past_token_type_ids = token_type_cache.get(self.layer_idx)
            else:
                past_token_type_ids = token_type_cache
            if past_token_type_ids is not None or query_token_type_ids is not None:
                merged_key_type_ids = packed_query_indexes.new_full((seqlens,), KV_TYPE_UNKNOWN)
                if past_token_type_ids is not None and past_token_type_ids.numel() > 0:
                    merged_key_type_ids[packed_key_value_indexes] = past_token_type_ids.to(
                        device=packed_query_indexes.device,
                        dtype=torch.long,
                    )
                if query_token_type_ids is not None:
                    merged_key_type_ids[packed_query_indexes] = query_token_type_ids
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            if query_token_type_ids is not None:
                merged_key_type_ids = query_token_type_ids
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        _record_token_attention(
            layer_idx=self.layer_idx,
            packed_query_states=packed_query_states,
            merged_key_states=merged_key_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            causal=is_causal,
        )

        packed_attn_output = varlen_attention(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
            layer_idx=self.layer_idx,
            mode="und",
            protected_key_indexes=packed_query_indexes,
            key_type_ids=merged_key_type_ids,
            physical_cache_by_sample=(
                past_key_values.get_physical_segments(self.layer_idx)
                if had_past_cache
                else None
            ),
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            mutation = _apply_topk_cache_update(
                k=merged_key_states,
                v=merged_value_states,
                key_type_ids=merged_key_type_ids,
                sample_lens=key_values_lens,
                layer_idx=self.layer_idx,
                mode="und",
            )
            if mutation is not None:
                merged_key_states = mutation.key
                merged_value_states = mutation.value
                merged_key_type_ids = mutation.key_type_ids
                key_values_lens = mutation.sample_lens
                past_key_values.set_physical_segments(
                    self.layer_idx, mutation.physical_segments_by_sample
                )
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states
            past_key_values.set_layer_lens(self.layer_idx, key_values_lens)
            if merged_key_type_ids is not None:
                past_key_values.token_type_cache[self.layer_idx] = merged_key_type_ids.detach()
        elif had_past_cache:
            past_token_type_ids = past_key_values.token_type_cache.get(self.layer_idx)
            mutation = _apply_topk_cache_update(
                k=past_key_values.key_cache[self.layer_idx],
                v=past_key_values.value_cache[self.layer_idx],
                key_type_ids=past_token_type_ids,
                sample_lens=cache_input_lens,
                layer_idx=self.layer_idx,
                mode="und",
            )
            if mutation is not None and mutation.bytes_saved > 0:
                past_key_values.key_cache[self.layer_idx] = mutation.key
                past_key_values.value_cache[self.layer_idx] = mutation.value
                past_key_values.token_type_cache[self.layer_idx] = mutation.key_type_ids
                past_key_values.set_layer_lens(self.layer_idx, mutation.sample_lens)
                past_key_values.set_physical_segments(
                    self.layer_idx, mutation.physical_segments_by_sample
                )

        return packed_attn_output, past_key_values


class PackedAttentionMoT(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)

        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(packed_sequence_gen)

        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(packed_sequence_gen)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        if self.config.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])

        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, List):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        packed_attn_output_[packed_und_token_indexes] = self.o_proj(packed_attn_output[packed_und_token_indexes])
        packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes])

        return packed_attn_output_

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        if mode == 'und':
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == 'gen':
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
            packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
            packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)

            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)

            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_vae_token_indexes])

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)
        had_past_cache = past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None
        cache_input_lens = key_values_lens
        if had_past_cache:
            layer_layout = resolve_layer_cache_layout(
                past_key_values,
                layer_idx=self.layer_idx,
                query_lens=query_lens,
                fallback_key_value_lens=key_values_lens,
                query_token_count=packed_query_states.shape[0],
            )
            key_values_lens = layer_layout.key_value_lens
            cache_input_lens = key_values_lens
            packed_key_value_indexes = layer_layout.packed_key_value_indexes
            packed_query_indexes = layer_layout.packed_query_indexes
        query_token_type_ids = None
        if past_key_values is not None:
            query_token_type_ids = getattr(past_key_values, "next_token_type_ids", None)
            if query_token_type_ids is not None:
                query_token_type_ids = query_token_type_ids.to(device=packed_query_indexes.device, dtype=torch.long)
        merged_key_type_ids = None

        if had_past_cache:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            token_type_cache = getattr(past_key_values, "token_type_cache", None)
            if isinstance(token_type_cache, dict):
                past_token_type_ids = token_type_cache.get(self.layer_idx)
            else:
                past_token_type_ids = token_type_cache
            if past_token_type_ids is not None or query_token_type_ids is not None:
                merged_key_type_ids = packed_query_indexes.new_full((seqlens,), KV_TYPE_UNKNOWN)
                if past_token_type_ids is not None and past_token_type_ids.numel() > 0:
                    merged_key_type_ids[packed_key_value_indexes] = past_token_type_ids.to(
                        device=packed_query_indexes.device,
                        dtype=torch.long,
                    )
                if query_token_type_ids is not None:
                    merged_key_type_ids[packed_query_indexes] = query_token_type_ids
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            if query_token_type_ids is not None:
                merged_key_type_ids = query_token_type_ids
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        _record_token_attention(
            layer_idx=self.layer_idx,
            packed_query_states=packed_query_states,
            merged_key_states=merged_key_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            causal=is_causal,
        )

        packed_attn_output = varlen_attention(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
            layer_idx=self.layer_idx,
            mode=mode,
            protected_key_indexes=packed_query_indexes,
            key_type_ids=merged_key_type_ids,
            physical_cache_by_sample=(
                past_key_values.get_physical_segments(self.layer_idx)
                if had_past_cache
                else None
            ),
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if mode == 'und':
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == 'gen':
            packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])

        if update_past_key_values:
            mutation = _apply_topk_cache_update(
                k=merged_key_states,
                v=merged_value_states,
                key_type_ids=merged_key_type_ids,
                sample_lens=key_values_lens,
                layer_idx=self.layer_idx,
                mode=mode,
            )
            if mutation is not None:
                merged_key_states = mutation.key
                merged_value_states = mutation.value
                merged_key_type_ids = mutation.key_type_ids
                key_values_lens = mutation.sample_lens
                past_key_values.set_physical_segments(
                    self.layer_idx, mutation.physical_segments_by_sample
                )
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states
            past_key_values.set_layer_lens(self.layer_idx, key_values_lens)
            if merged_key_type_ids is not None:
                past_key_values.token_type_cache[self.layer_idx] = merged_key_type_ids.detach()
        elif had_past_cache:
            past_token_type_ids = past_key_values.token_type_cache.get(self.layer_idx)
            mutation = _apply_topk_cache_update(
                k=past_key_values.key_cache[self.layer_idx],
                v=past_key_values.value_cache[self.layer_idx],
                key_type_ids=past_token_type_ids,
                sample_lens=cache_input_lens,
                layer_idx=self.layer_idx,
                mode=mode,
            )
            if mutation is not None and mutation.bytes_saved > 0:
                past_key_values.key_cache[self.layer_idx] = mutation.key
                past_key_values.value_cache[self.layer_idx] = mutation.value
                past_key_values.token_type_cache[self.layer_idx] = mutation.key_type_ids
                past_key_values.set_layer_lens(self.layer_idx, mutation.sample_lens)
                past_key_values.set_physical_segments(
                    self.layer_idx, mutation.physical_segments_by_sample
                )

        return packed_attn_output, past_key_values


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self, 
        config, 
        layer_idx: Optional[int] = None, 
        attn_module: Optional[Qwen2Attention] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.input_layernorm(packed_sequence[packed_und_token_indexes])
        packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])

        # Self Attention
        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
        packed_sequence = residual + packed_sequence_

        # Fully Connected
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
    
        packed_sequence_[packed_gen_token_indexes] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        )
        packed_sequence = residual + packed_sequence_

        return packed_sequence

    def _forward_inference_duca_current_state(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        layer = int(self.current["layer"])
        runtime_params = get_current_state_runtime_params(layer)
        for name in (
            "fresh_ratio",
            "fresh_threshold",
            "soft_fresh_weight",
            "first_enhance",
            "min_fresh_tokens",
            "score_type",
            "seed",
            "schedule_mode",
        ):
            if name in runtime_params:
                self.cache_dic[name] = runtime_params[name]
        vae_token_count = int(packed_vae_token_indexes.numel())
        seq_len = int(packed_query_sequence.shape[0])
        step_type = str(self.current.get("type", "full"))
        if not duca_layer_ready(self.cache_dic, layer, seq_len, vae_token_count):
            step_type = "full"

        if step_type == "full":
            residual = packed_query_sequence
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                packed_query_sequence[packed_text_indexes]
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )

            packed_query_sequence_, past_key_values = self.self_attn(
                packed_query_sequence=packed_query_sequence_,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )
            attn_output = packed_query_sequence_
            packed_query_sequence = residual + attn_output

            residual = packed_query_sequence
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(torch.bfloat16)

            mlp_output = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            mlp_output[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            mlp_output[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
            duca_store_full(
                self.cache_dic,
                layer,
                attn_output=attn_output,
                mlp_output=mlp_output,
                vae_token_count=vae_token_count,
                attention_score=get_current_state_attention_scores(layer),
            )
            duca_record(
                self.cache_dic,
                self.current,
                layer=layer,
                vae_token_count=vae_token_count,
                fresh_count=vae_token_count,
            )
            packed_query_sequence = residual + mlp_output
            return packed_query_sequence, past_key_values

        layer_cache = self.cache_dic["layers"][layer]
        residual = packed_query_sequence
        attn_output = layer_cache["attn"].to(device=packed_query_sequence.device, dtype=packed_query_sequence.dtype)
        packed_query_sequence = residual + attn_output

        residual = packed_query_sequence
        mlp_output = layer_cache["mlp"].to(device=packed_query_sequence.device, dtype=torch.bfloat16).clone()
        if packed_text_indexes.numel() > 0:
            packed_text_query_sequence = self.post_attention_layernorm(
                packed_query_sequence[packed_text_indexes]
            ).to(torch.bfloat16)
            mlp_output[packed_text_indexes] = self.mlp(packed_text_query_sequence)

        fresh_count = 0
        if step_type == "ToCa" and vae_token_count > 0:
            fresh_local_indexes = duca_select_fresh(
                self.cache_dic,
                self.current,
                layer=layer,
                vae_states=packed_query_sequence[packed_vae_token_indexes],
            )
            if fresh_local_indexes.numel() > 0:
                fresh_abs_indexes = packed_vae_token_indexes[fresh_local_indexes]
                fresh_query_sequence = self.post_attention_layernorm_moe_gen(
                    packed_query_sequence[fresh_abs_indexes]
                ).to(torch.bfloat16)
                mlp_output[fresh_abs_indexes] = self.mlp_moe_gen(fresh_query_sequence)
                fresh_count = int(fresh_local_indexes.numel())
            duca_update_age(self.cache_dic, layer, fresh_local_indexes)
        else:
            duca_update_age(
                self.cache_dic,
                layer,
                torch.empty(0, dtype=torch.long, device=packed_query_sequence.device),
            )

        layer_cache["mlp"] = mlp_output.detach()
        duca_record(
            self.cache_dic,
            self.current,
            layer=layer,
            vae_token_count=vae_token_count,
            fresh_count=fresh_count,
        )
        packed_query_sequence = residual + mlp_output
        return packed_query_sequence, past_key_values

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        
        enable_duca = getattr(self, "enable_duca_current_state_cache", False)
        if (
            enable_duca
            and mode == "gen"
            and packed_vae_token_indexes is not None
            and packed_text_indexes is not None
        ):
            return self._forward_inference_duca_current_state(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )

        enable_taylorseer = getattr(self, 'enable_taylorseer', False)

        if enable_taylorseer and self.current['type'] == 'full':
            self.current['module'] = 'total'
            taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

        if not enable_taylorseer or (enable_taylorseer and self.current['type'] == 'full'):
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = self.input_layernorm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_

            # Self Attention
            packed_query_sequence, past_key_values = self.self_attn(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )
            packed_query_sequence = residual + packed_query_sequence

            # Fully Connected
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
                packed_query_sequence = self.mlp(packed_query_sequence)
            elif mode == "gen":
                packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
                packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
                packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
                packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(torch.bfloat16)

                packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
                packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
                packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
                packed_query_sequence = packed_query_sequence_

            packed_query_sequence = residual + packed_query_sequence
        
        if enable_taylorseer:
            if self.current['type'] == 'full':
                derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=packed_query_sequence)
            elif self.current['type'] == 'Taylor':
                self.current['module'] = 'total'
                packed_query_sequence = taylor_formula(cache_dic=self.cache_dic, current=self.current)

        return packed_query_sequence, past_key_values


class Qwen2MoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)

        packed_sequence_new = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_und = self.mlp(packed_sequence[packed_und_token_indexes])
        packed_sequence_gen = self.mlp_moe_gen(packed_sequence[packed_gen_token_indexes])
        packed_sequence_new[packed_und_token_indexes] = packed_sequence_und
        packed_sequence_new[packed_gen_token_indexes] = packed_sequence_gen

        packed_sequence = residual + packed_sequence_new

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        if mode == "und":
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_query_sequence[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoEDecoderLayer": Qwen2MoEDecoderLayer,
    "Qwen2MoTDecoderLayer": partial(Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT),
}


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = 'Mo' in config.layer_module

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        if self.config.freeze_und:
            packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        for decoder_layer in self.layers:
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs
            )

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
            if self.config.freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
            packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
            return packed_sequence_
        else:
            return self.norm(packed_sequence)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        
        enable_taylorseer = getattr(self, 'enable_taylorseer', False)
        if enable_taylorseer:
            cal_type(self.cache_dic, self.current)
            self.current['stream'] = 'layers_stream'

        enable_duca = getattr(self, "enable_duca_current_state_cache", False) and mode == "gen"
        if enable_duca:
            duca_prepare_step(self.duca_cache_dic, self.duca_current)
            self.duca_current["num_layers"] = len(self.layers)

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == 'gen':
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for layer_idx, decoder_layer in enumerate(self.layers):
            decoder_layer.enable_taylorseer = False
            decoder_layer.enable_duca_current_state_cache = False
            if enable_taylorseer:
                decoder_layer.current = self.current
                decoder_layer.cache_dic = self.cache_dic
                decoder_layer.enable_taylorseer = True
                self.current['layer'] = layer_idx
            if enable_duca:
                decoder_layer.current = self.duca_current
                decoder_layer.cache_dic = self.duca_cache_dic
                decoder_layer.enable_taylorseer = False
                decoder_layer.enable_duca_current_state_cache = True
                self.duca_current["layer"] = layer_idx
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)
        
        if enable_taylorseer:
            self.current['step'] += 1
        if enable_duca:
            self.duca_current["step"] += 1

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self):
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        outputs = self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

        return outputs
