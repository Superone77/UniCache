"""Dynamic budget schedulers for UniCache."""

from __future__ import annotations

import math

from ..core.constants import SUPPORTED_FAKE_QUANT_BITS
from ..core.registry import BudgetScheduler
from ..core.schema import KVSegment, RuntimeContext
from ..metrics import (
    DEFAULT_LAYER_CHUNK_SIZE,
    DEFAULT_STEP_CHUNK_SIZE,
    block_metric_key,
    chunk_indices,
)


# Nominal retained-storage budget -> KIVI configuration. Larger groups reduce
# quantization metadata, while bit width is the coarse precision control. The
# scheduler always chooses the highest row that does not exceed its assigned
# type-level budget. These nominal ratios are policy levels; runtime also
# reports a shape-aware storage estimate for later calibration.
KIVI_RETAINED_STORAGE_LOOKUP = (
    (0.125, 2, 128),
    (0.200, 2, 64),
    (0.250, 2, 32),
    (0.375, 4, 128),
    (0.450, 4, 64),
    (0.500, 4, 32),
    (0.625, 8, 128),
    (0.700, 8, 64),
    (0.800, 8, 32),
    (1.000, 16, 128),
)


def _progress(index: int | None, total: int | None) -> float:
    if index is None or total is None or total <= 1:
        return 0.0
    return min(1.0, max(0.0, float(index) / float(total - 1)))


def _lerp(start: float, end: float, progress: float) -> float:
    return float(start) + (float(end) - float(start)) * float(progress)


class ConstantBudgetScheduler(BudgetScheduler):
    name = "constant_budget"

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        return dict(params)


class LayerPyramidBudgetScheduler(BudgetScheduler):
    name = "layer_pyramid_budget"

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        capacity = int(params.get("max_capacity_prompt", params.get("selected_capacity", 0)))
        beta = max(float(params.get("beta", 20.0)), 1.0)
        if capacity <= 0:
            return {"max_capacity_prompt": 0, "keep_k": 0, "beta": beta}
        min_num = max(1, int(capacity // beta))
        max_num = max(min_num, int(capacity * 2 - min_num))
        total_layers = max(1, int(ctx.total_layers))
        if total_layers == 1:
            keep_k = capacity
        else:
            layer_progress = min(1.0, max(0.0, ctx.layer_idx / float(total_layers - 1)))
            keep_k = max(1, int(round(max_num - (max_num - min_num) * layer_progress)))
        return {
            "max_capacity_prompt": capacity,
            "keep_k": keep_k,
            "beta": beta,
            "layer_idx": int(ctx.layer_idx),
            "total_layers": total_layers,
        }


class PyramidKVLayerBudgetScheduler(BudgetScheduler):
    """Official PyramidKV layer budget after reserving the recent window."""

    name = "pyramidkv_layer_budget"

    def validate_params(self, params: dict) -> None:
        capacity = int(params.get("max_capacity_prompt", 0))
        window = int(params.get("window_size", params.get("recent_size", 64)))
        if capacity <= 0:
            raise ValueError("pyramidkv_layer_budget max_capacity_prompt must be positive")
        if window < 0 or window >= capacity:
            raise ValueError("pyramidkv_layer_budget window_size must be in [0, max_capacity_prompt)")
        if float(params.get("beta", 20.0)) < 1.0:
            raise ValueError("pyramidkv_layer_budget beta must be at least 1")

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        self.validate_params(params)
        capacity = int(params["max_capacity_prompt"])
        window = int(params.get("window_size", params.get("recent_size", 64)))
        beta = float(params.get("beta", 20.0))
        compressible = capacity - window
        min_budget = max(1, int(compressible // beta))
        max_budget = max(min_budget, int(compressible * 2 - min_budget))
        total_layers = max(1, int(ctx.total_layers))
        if total_layers == 1:
            selected = compressible
        else:
            progress = min(1.0, max(0.0, float(ctx.layer_idx) / float(total_layers - 1)))
            selected = int(round(max_budget - (max_budget - min_budget) * progress))
        selected = max(1, selected)
        return {
            "keep_k": int(window + selected),
            "selected_capacity": selected,
            "window_size": window,
            "recent_size": window,
            "max_capacity_prompt": capacity,
            "beta": beta,
            "layer_idx": int(ctx.layer_idx),
            "total_layers": total_layers,
        }


class RetireAfterScheduler(BudgetScheduler):
    name = "retire_after"

    _SYMBOLS = {
        "0": 0.0,
        "T/4": 0.25,
        "T/3": 1.0 / 3.0,
        "T/2": 0.5,
        "2T/3": 2.0 / 3.0,
        "3T/4": 0.75,
        "T+1": None,
    }

    def validate_params(self, params: dict) -> None:
        fields = [name for name in ("cutoff", "cutoff_step", "cutoff_fraction") if name in params]
        if len(fields) != 1:
            raise ValueError("retire_after requires exactly one of cutoff, cutoff_step, or cutoff_fraction")
        if "cutoff" in params and str(params["cutoff"]) not in self._SYMBOLS:
            raise ValueError(f"retire_after unsupported symbolic cutoff: {params['cutoff']}")
        if "cutoff_step" in params and int(params["cutoff_step"]) < 0:
            raise ValueError("retire_after cutoff_step must be non-negative")
        if "cutoff_fraction" in params and not 0.0 <= float(params["cutoff_fraction"]) <= 1.0:
            raise ValueError("retire_after cutoff_fraction must be in [0, 1]")

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        self.validate_params(params)
        total = max(1, int(ctx.total_steps or 1))
        step = int(ctx.step_index or 0)
        if "cutoff_step" in params:
            cutoff_step = int(params["cutoff_step"])
        elif "cutoff_fraction" in params:
            cutoff_step = int(round(float(params["cutoff_fraction"]) * max(total - 1, 0)))
        else:
            symbol = str(params["cutoff"])
            fraction = self._SYMBOLS[symbol]
            cutoff_step = total if fraction is None else int(round(fraction * max(total - 1, 0)))
        return {
            "retired": bool(step >= cutoff_step and cutoff_step < total),
            "cutoff_step": cutoff_step,
            "current_step": step,
            "total_steps": total,
        }


class DenoiseLinearDecayScheduler(BudgetScheduler):
    name = "denoise_linear_decay"

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        p = _progress(ctx.step_index, ctx.total_steps)
        keep_ratio = _lerp(params.get("start_keep_ratio", 1.0), params.get("end_keep_ratio", 1.0), p)
        out = {"keep_ratio": keep_ratio, "step_progress": p}
        if params.get("layer_shape") == "pyramid":
            capacity = int(round(float(params.get("base_capacity", params.get("max_capacity_prompt", 1000))) * keep_ratio))
            layer_params = dict(params)
            layer_params["max_capacity_prompt"] = max(int(params.get("min_keep", 1)), capacity)
            out.update(LayerPyramidBudgetScheduler().resolve(layer_params, ctx, segment))
            out["keep_ratio"] = keep_ratio
        return out


class DecodeLinearDecayScheduler(BudgetScheduler):
    name = "decode_linear_decay"

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        total = params.get("total_decode_steps", ctx.total_steps)
        p = _progress(ctx.decode_index, total)
        return {"keep_ratio": _lerp(params.get("start_keep_ratio", 1.0), params.get("end_keep_ratio", 1.0), p)}


class BitWidthDecayScheduler(BudgetScheduler):
    name = "bit_width_decay"

    def validate_params(self, params: dict) -> None:
        supported = {int(bits) for bits in params.get("supported_bits", SUPPORTED_FAKE_QUANT_BITS)}
        invalid = supported - SUPPORTED_FAKE_QUANT_BITS
        if not supported or invalid:
            raise ValueError(
                "bit_width_decay supported_bits must be a non-empty subset of "
                f"{sorted(SUPPORTED_FAKE_QUANT_BITS)}; got {sorted(supported)}"
            )
        for name in ("start_bits", "end_bits", "start_k_bits", "end_k_bits", "start_v_bits", "end_v_bits"):
            if name in params and int(params[name]) not in supported:
                raise ValueError(f"bit_width_decay {name}={params[name]} is not present in supported_bits")

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        self.validate_params(params)
        p = _progress(ctx.step_index, ctx.total_steps)
        supported = sorted({int(bits) for bits in params.get("supported_bits", SUPPORTED_FAKE_QUANT_BITS)})

        def resolve_bits(start_name: str, end_name: str) -> tuple[int, float]:
            target = _lerp(
                params.get(start_name, params.get("start_bits", 8)),
                params.get(end_name, params.get("end_bits", 8)),
                p,
            )
            # Prefer the higher precision level on a tie.
            selected = min(supported, key=lambda bits: (abs(bits - target), -bits))
            return int(selected), float(target)

        k_bits, k_target = resolve_bits("start_k_bits", "end_k_bits")
        v_bits, v_target = resolve_bits("start_v_bits", "end_v_bits")
        return {
            "bits": k_bits if k_bits == v_bits else min(k_bits, v_bits),
            "k_bits": k_bits,
            "v_bits": v_bits,
            "target_k_bits": k_target,
            "target_v_bits": v_target,
            "supported_bits": supported,
            "step_progress": p,
        }


class FreshRatioSchedule(BudgetScheduler):
    name = "fresh_ratio_schedule"

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        p = _progress(ctx.step_index, ctx.total_steps)
        return {
            "fresh_ratio": _lerp(params.get("start_fresh_ratio", 0.2), params.get("end_fresh_ratio", 0.2), p),
            "step_progress": p,
        }


class ChunkedEMAAttentionMassBudgetScheduler(BudgetScheduler):
    """Allocate a shared type-level budget from the previous chunk's EMA mass."""

    name = "chunked_ema_attention_mass_budget"
    _SUPPORTED_OPERATORS = {"h2o_segment_attention_mask", "kivi_quantization"}

    def validate_params(self, params: dict) -> None:
        operator = str(params.get("operator", ""))
        if operator not in self._SUPPORTED_OPERATORS:
            raise ValueError(
                "chunked_ema_attention_mass_budget operator must be one of "
                f"{sorted(self._SUPPORTED_OPERATORS)}"
            )
        total_ratio = float(params.get("total_budget_ratio", 1.0))
        if not 0.0 <= total_ratio <= 1.0:
            raise ValueError("total_budget_ratio must be in [0, 1]")
        initial_ratio = float(params.get("initial_total_budget_ratio", total_ratio))
        if not 0.0 <= initial_ratio <= 1.0:
            raise ValueError("initial_total_budget_ratio must be in [0, 1]")
        if bool(params.get("monotonic_retirement", False)) and initial_ratio < total_ratio:
            raise ValueError(
                "monotonic_retirement requires initial_total_budget_ratio >= total_budget_ratio"
            )
        allocation_mode = str(params.get("allocation_mode", "ema_attention_mass"))
        if allocation_mode not in {"ema_attention_mass", "static_capacity"}:
            raise ValueError(
                "chunked_ema_attention_mass_budget allocation_mode must be "
                "'ema_attention_mass' or 'static_capacity'"
            )
        min_ratio = float(params.get("min_total_budget_ratio", 0.0))
        if not 0.0 <= min_ratio <= initial_ratio:
            raise ValueError(
                "min_total_budget_ratio must be in [0, initial_total_budget_ratio]"
            )
        gamma = float(params.get("conditional_decay_gamma", 1.0))
        if gamma <= 0.0:
            raise ValueError("conditional_decay_gamma must be positive")
        for name in ("step_chunk_size", "layer_chunk_size"):
            default = DEFAULT_STEP_CHUNK_SIZE if name == "step_chunk_size" else DEFAULT_LAYER_CHUNK_SIZE
            if int(params.get(name, default)) <= 0:
                raise ValueError(f"{name} must be positive")
        if operator == "kivi_quantization":
            invalid = {bits for _, bits, _ in KIVI_RETAINED_STORAGE_LOOKUP} - SUPPORTED_FAKE_QUANT_BITS
            if invalid:
                raise ValueError(f"KIVI lookup contains unsupported precision: {sorted(invalid)}")

    @staticmethod
    def _weighted_capped_allocation(
        capacities: dict[str, float],
        weights: dict[str, float],
        total_budget: float,
    ) -> dict[str, float]:
        allocation = {name: 0.0 for name in capacities}
        active = set(capacities)
        remaining = min(float(total_budget), sum(capacities.values()))
        while active and remaining > 1e-9:
            weight_sum = sum(max(0.0, weights[name]) for name in active)
            if weight_sum <= 0.0:
                local_weights = {name: 1.0 for name in active}
                weight_sum = float(len(active))
            else:
                local_weights = {name: max(0.0, weights[name]) for name in active}
            proposals = {
                name: remaining * local_weights[name] / weight_sum for name in active
            }
            capped = [
                name
                for name in active
                if proposals[name] >= capacities[name] - allocation[name]
            ]
            if not capped:
                for name, amount in proposals.items():
                    allocation[name] += amount
                break
            for name in capped:
                amount = capacities[name] - allocation[name]
                allocation[name] += amount
                remaining -= amount
                active.remove(name)
        return allocation

    @staticmethod
    def _estimate_kivi_ratio(
        *,
        segment_tokens: int,
        head_dim: int,
        bits: int,
        group_size: int,
        residual_length: int,
    ) -> float:
        tokens = max(1, int(segment_tokens))
        dim = max(1, int(head_dim))
        full_tokens = min(tokens, max(0, int(residual_length)))
        quant_tokens = tokens - full_tokens
        original = float(tokens * dim * 4)
        full_bytes = float(full_tokens * dim * 4)
        payload = float(quant_tokens * dim * (2 * bits) / 8.0)
        key_groups = math.ceil(quant_tokens / group_size) if quant_tokens else 0
        value_groups = math.ceil(dim / group_size) if quant_tokens else 0
        metadata = float((key_groups * dim + quant_tokens * value_groups) * 8)
        return (full_bytes + payload + metadata) / original

    def _resolve_kivi_precision(
        self,
        params: dict,
        *,
        target: float,
        segment_tokens: int | None,
        head_dim: int | None,
    ) -> dict:
        eligible = [row for row in KIVI_RETAINED_STORAGE_LOOKUP if row[0] <= target + 1e-9]
        nominal_ratio, bits, group_size = eligible[-1] if eligible else KIVI_RETAINED_STORAGE_LOOKUP[0]
        estimated = None
        if segment_tokens is not None and head_dim is not None:
            estimated = self._estimate_kivi_ratio(
                segment_tokens=segment_tokens,
                head_dim=head_dim,
                bits=bits,
                group_size=group_size,
                residual_length=int(params.get("residual_length", 32)),
            )
        return {
            "bits": bits,
            "k_bits": bits,
            "v_bits": bits,
            "group_size": group_size,
            "target_bits": 16.0 * target,
            "kivi_lookup_storage_ratio": nominal_ratio,
            "kivi_budget_gap": target - nominal_ratio,
            "estimated_storage_ratio": estimated,
            "kivi_lookup_source": "KIVI_RETAINED_STORAGE_LOOKUP",
        }

    @staticmethod
    def _scheduled_total_ratio(
        params: dict,
        ctx: RuntimeContext,
        *,
        step_chunk: int,
    ) -> float:
        target = float(params.get("total_budget_ratio", 1.0))
        initial = float(params.get("initial_total_budget_ratio", target))
        if initial == target:
            return target
        step_chunk_size = int(params.get("step_chunk_size", DEFAULT_STEP_CHUNK_SIZE))
        total_steps = max(1, int(ctx.total_steps or 1))
        total_chunks = max(1, math.ceil(total_steps / step_chunk_size))
        if total_chunks == 1:
            progress = 0.0
        else:
            progress = min(1.0, max(0.0, float(step_chunk) / float(total_chunks - 1)))
        schedule = str(params.get("budget_schedule", "linear"))
        if schedule != "linear":
            raise ValueError(f"unsupported chunked EMA budget_schedule {schedule!r}")
        return initial + (target - initial) * progress

    def resolve(self, params: dict, ctx: RuntimeContext, segment: KVSegment | None = None) -> dict:
        self.validate_params(params)
        segment_id = segment.id if segment is not None else str(params.get("segment_id", ""))
        if not segment_id:
            raise ValueError("chunked EMA budget resolution requires a segment id")
        layer_chunk, step_chunk = chunk_indices(
            layer_idx=ctx.layer_idx,
            step_index=ctx.step_index,
            decode_index=ctx.decode_index,
            layer_chunk_size=int(params.get("layer_chunk_size", DEFAULT_LAYER_CHUNK_SIZE)),
            step_chunk_size=int(params.get("step_chunk_size", DEFAULT_STEP_CHUNK_SIZE)),
            first_layer_separate=bool(params.get("first_layer_separate", False)),
            first_step_separate=bool(params.get("first_step_separate", False)),
        )
        previous_step_chunk = step_chunk - 1
        metrics = dict(ctx.operator_state.get("block_metrics", {}) or {})
        configured_budget_segments = [
            str(item) for item in params.get("budget_segments", [segment_id])
        ]
        active_cache_types = {
            str(item) for item in ctx.kv_metadata.get("active_cache_types", [])
        }
        active_budget_segments = [
            name for name in configured_budget_segments if name in active_cache_types
        ]
        if not active_cache_types or segment_id not in active_budget_segments:
            active_budget_segments = list(configured_budget_segments)
        inactive_budget_segments = [
            name for name in configured_budget_segments if name not in active_budget_segments
        ]
        budget_segments = active_budget_segments
        previous_metrics = {}
        metric_layer_chunk = layer_chunk
        metric_step_chunk = previous_step_chunk
        if previous_step_chunk >= 0:
            for candidate in budget_segments:
                value = metrics.get(
                    block_metric_key(
                        run_id=ctx.run_id,
                        phase=ctx.phase,
                        branch=ctx.branch,
                        cfg_branch=ctx.cfg_branch,
                        batch_idx=ctx.batch_idx,
                        segment_id=candidate,
                        layer_chunk=layer_chunk,
                        step_chunk=previous_step_chunk,
                    )
                )
                if value is not None:
                    previous_metrics[candidate] = value
        if (
            len(previous_metrics) != len(budget_segments)
            and bool(params.get("bootstrap_from_previous_layer_chunk", False))
            and layer_chunk > 0
        ):
            spatial_metrics = {}
            for candidate in budget_segments:
                value = metrics.get(
                    block_metric_key(
                        run_id=ctx.run_id,
                        phase=ctx.phase,
                        branch=ctx.branch,
                        cfg_branch=ctx.cfg_branch,
                        batch_idx=ctx.batch_idx,
                        segment_id=candidate,
                        layer_chunk=layer_chunk - 1,
                        step_chunk=step_chunk,
                    )
                )
                if value is not None:
                    spatial_metrics[candidate] = value
            if len(spatial_metrics) == len(budget_segments):
                previous_metrics = spatial_metrics
                metric_layer_chunk = layer_chunk - 1
                metric_step_chunk = step_chunk
        total_ratio = float(params.get("total_budget_ratio", 1.0))
        initial_total_ratio = float(params.get("initial_total_budget_ratio", total_ratio))
        progress_index = ctx.step_index if ctx.step_index is not None else ctx.decode_index
        full_kv_calibration = bool(params.get("first_step_full", False)) and int(
            progress_index or 0
        ) == 0
        conditional_decay = bool(params.get("conditional_attention_budget_decay", False))
        reference_conditional_mass = None
        previous_conditional_mass = None
        conditional_mass_ratio = None
        total_budget_source = "configured_schedule"
        if full_kv_calibration:
            scheduled_total_ratio = 1.0
            total_budget_source = "full_kv_calibration"
        elif conditional_decay:
            def chunk_mass(candidate_step_chunk: int) -> float | None:
                values = []
                for candidate in budget_segments:
                    value = metrics.get(
                        block_metric_key(
                            run_id=ctx.run_id,
                            phase=ctx.phase,
                            branch=ctx.branch,
                            cfg_branch=ctx.cfg_branch,
                            batch_idx=ctx.batch_idx,
                            segment_id=candidate,
                            layer_chunk=layer_chunk,
                            step_chunk=candidate_step_chunk,
                        )
                    )
                    if value is None:
                        return None
                    values.append(float(value["ema_attention_mass"]))
                return sum(values)

            reference_conditional_mass = chunk_mass(0)
            previous_conditional_mass = chunk_mass(previous_step_chunk)
            scheduled_total_ratio = initial_total_ratio
            if reference_conditional_mass is not None and reference_conditional_mass > 1e-12:
                gamma = float(params.get("conditional_decay_gamma", 1.0))
                min_ratio = float(params.get("min_total_budget_ratio", 0.0))
                historical_ratios = [initial_total_ratio]
                for candidate_step_chunk in range(1, max(1, previous_step_chunk + 1)):
                    mass = chunk_mass(candidate_step_chunk)
                    if mass is None:
                        continue
                    share_ratio = max(0.0, mass / reference_conditional_mass)
                    historical_ratios.append(
                        min(
                            initial_total_ratio,
                            max(min_ratio, initial_total_ratio * (share_ratio ** gamma)),
                        )
                    )
                scheduled_total_ratio = min(historical_ratios)
                if previous_conditional_mass is not None:
                    conditional_mass_ratio = (
                        previous_conditional_mass / reference_conditional_mass
                    )
                total_budget_source = "conditional_attention_share_decay"
            else:
                total_budget_source = "conditional_attention_calibration_fallback"
        else:
            scheduled_total_ratio = self._scheduled_total_ratio(
                params,
                ctx,
                step_chunk=step_chunk,
            )
        if full_kv_calibration:
            target = 1.0
            source = "full_kv_calibration"
            ema_mass = None
            total_budget = None
            weights = {}
            allocation = {}
            segment_tokens = ctx.kv_metadata.get("current_segment_tokens")
        elif len(previous_metrics) != len(budget_segments):
            target = float(params.get("fallback_budget_ratio", scheduled_total_ratio))
            if "initial_total_budget_ratio" in params:
                target = min(target, scheduled_total_ratio)
            source = "fallback"
            ema_mass = None
            total_budget = None
            weights = {}
            allocation = {}
            segment_tokens = ctx.kv_metadata.get("current_segment_tokens")
        else:
            capacities = {
                name: float(max(1, int(previous_metrics[name]["segment_tokens"])))
                for name in budget_segments
            }
            allocation_mode = str(params.get("allocation_mode", "ema_attention_mass"))
            if allocation_mode == "static_capacity":
                weights = dict(capacities)
            else:
                weights = {
                    name: float(previous_metrics[name]["ema_attention_mass"])
                    for name in budget_segments
                }
            total_budget = scheduled_total_ratio * sum(capacities.values())
            allocation = self._weighted_capped_allocation(capacities, weights, total_budget)
            segment_tokens = int(capacities[segment_id])
            target = allocation[segment_id] / capacities[segment_id]
            ema_mass = weights[segment_id]
            source = (
                "shared_budget_static_capacity"
                if allocation_mode == "static_capacity"
                else (
                    "shared_budget_from_previous_layer_chunk_ema_attention_mass"
                    if metric_step_chunk == step_chunk and metric_layer_chunk == layer_chunk - 1
                    else "shared_budget_from_previous_step_chunk_ema_attention_mass"
                )
            )
        target = min(1.0, max(0.0, target))
        resolved = {
            "dynamic_budget": True,
            "target_budget_ratio": target,
            "budget_source": source,
            "observed_ema_attention_mass": ema_mass,
            "metric_layer_chunk": metric_layer_chunk if previous_metrics else layer_chunk,
            "metric_step_chunk": metric_step_chunk if previous_metrics else None,
            "current_layer_chunk": layer_chunk,
            "current_step_chunk": step_chunk,
            "total_budget_ratio": total_ratio,
            "initial_total_budget_ratio": initial_total_ratio,
            "scheduled_total_budget_ratio": scheduled_total_ratio,
            "total_budget_source": total_budget_source,
            "full_kv_calibration": full_kv_calibration,
            "reference_conditional_attention_mass": reference_conditional_mass,
            "previous_conditional_attention_mass": previous_conditional_mass,
            "conditional_attention_mass_ratio": conditional_mass_ratio,
            "conditional_attention_budget_decay": conditional_decay,
            "allocation_mode": str(params.get("allocation_mode", "ema_attention_mass")),
            "budget_schedule": str(params.get("budget_schedule", "linear")),
            "bootstrap_from_previous_layer_chunk": bool(
                params.get("bootstrap_from_previous_layer_chunk", False)
            ),
            "monotonic_retirement": bool(params.get("monotonic_retirement", False)),
            "total_budget_token_equivalent": total_budget,
            "budget_weight_by_segment": dict(weights),
            "budget_allocation_token_equivalent_by_segment": dict(allocation),
            "budget_active_ema_mass_by_segment": {
                name: float(
                    previous_metrics[name].get(
                        "ema_active_attention_mass",
                        previous_metrics[name]["ema_attention_mass"],
                    )
                )
                for name in previous_metrics
            },
            "budget_mass_estimator_by_segment": {
                name: str(previous_metrics[name].get("mass_estimator", "active_only"))
                for name in previous_metrics
            },
            "budget_segments": budget_segments,
            "configured_budget_segments": configured_budget_segments,
            "active_budget_segments": active_budget_segments,
            "inactive_budget_segments": inactive_budget_segments,
            "first_layer_separate": bool(params.get("first_layer_separate", False)),
            "first_step_separate": bool(params.get("first_step_separate", False)),
        }
        operator = str(params["operator"])
        if operator == "h2o_segment_attention_mask":
            recent_fraction = float(params.get("recent_fraction", 0.25))
            if not 0.0 <= recent_fraction <= 1.0:
                raise ValueError("recent_fraction must be in [0, 1]")
            resolved.update(
                {
                    "recent_budget_ratio": target * recent_fraction,
                    "heavy_budget_ratio": target * (1.0 - recent_fraction),
                }
            )
        else:
            resolved.update(
                self._resolve_kivi_precision(
                    params,
                    target=target,
                    segment_tokens=int(segment_tokens) if segment_tokens is not None else None,
                    head_dim=(
                        int(ctx.kv_metadata["head_dim"])
                        if "head_dim" in ctx.kv_metadata
                        else None
                    ),
                )
            )
            if full_kv_calibration or target >= 1.0 - 1e-9:
                resolved["bypass_quantization"] = True
        return resolved


def built_in_schedulers() -> list[BudgetScheduler]:
    return [
        ConstantBudgetScheduler(),
        LayerPyramidBudgetScheduler(),
        PyramidKVLayerBudgetScheduler(),
        RetireAfterScheduler(),
        DenoiseLinearDecayScheduler(),
        DecodeLinearDecayScheduler(),
        BitWidthDecayScheduler(),
        FreshRatioSchedule(),
        ChunkedEMAAttentionMassBudgetScheduler(),
    ]
