"""H2O baseline and UniCache segment-aware adaptation.

The score update and heavy-hitter/recent selection follow the public H2O
quality-evaluation implementation at FMInference/H2O commit
ac75c2a8a9e76832b2a4139b9363373b56336bfb. BAGEL keeps K/V materialized in
this prototype, so both operators apply the selected cache through a mask.
"""

from __future__ import annotations

import torch

from ..core.capabilities import CapabilityDescriptor
from ..core.constants import BAGEL_KV_TYPE_CURRENT_VAE
from ..core.decisions import SelectionDecision
from ..core.registry import OperatorExecutionContext, ProcessingOperator
from ..storage import LogicalHostArchive


class _H2OAttentionMaskBase(ProcessingOperator):
    family = "eviction"
    stages = {"attention"}
    executable = True
    uses_final_attention = True
    dynamic_budget_supported = False
    capabilities = CapabilityDescriptor(
        stages=frozenset(stages),
        layouts=frozenset({"attention_view", "dense"}),
        devices=frozenset({"cpu", "cuda", "mps"}),
        requires=frozenset({"query", "key", "value", "segment_mask", "protected_mask"}),
        produces_selection=True,
        stateful=True,
        mutation_scope="attention_mask",
        physical_storage_mutation=False,
    )

    def build_params(self, params: dict) -> dict:
        out = dict(params)
        working_set_mode = str(out.get("working_set_mode", "mask_only"))
        if working_set_mode not in {"mask_only", "irreversible", "host_backed"}:
            raise ValueError(
                f"{self.name} working_set_mode must be mask_only, irreversible, "
                f"or host_backed, got {working_set_mode!r}"
            )
        out["working_set_mode"] = working_set_mode
        if working_set_mode == "host_backed" and bool(out.get("monotonic_retirement", False)):
            raise ValueError(
                f"{self.name} host_backed mode conflicts with monotonic_retirement=true"
            )
        if working_set_mode == "irreversible":
            out["monotonic_retirement"] = True
        score_source = str(out.get("score_source", "effective_attention"))
        if score_source not in {
            "effective_attention",
            "original_attention",
            "pre_eviction_attention",
        }:
            raise ValueError(
                f"{self.name} score_source must be effective_attention, "
                f"original_attention, or pre_eviction_attention, got {score_source!r}"
            )
        out["score_source"] = score_source
        score_update_policy = str(out.get("score_update_policy", "cumulative"))
        if score_update_policy not in {"cumulative", "first_observation"}:
            raise ValueError(
                f"{self.name} score_update_policy must be cumulative or "
                f"first_observation, got {score_update_policy!r}"
            )
        out["score_update_policy"] = score_update_policy
        for name in ("heavy_budget_ratio", "recent_budget_ratio"):
            value = float(out.get(name, 0.0))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{self.name} {name} must be in [0, 1], got {value}")
        for name in ("heavy_size", "hh_size", "recent_size"):
            if name in out and int(out[name]) < 0:
                raise ValueError(f"{self.name} {name} must be non-negative")
        recall_selector = str(out.get("recall_selector", "none"))
        if recall_selector not in {"none", "block_centroid"}:
            raise ValueError(
                f"{self.name} recall_selector must be none or block_centroid, "
                f"got {recall_selector!r}"
            )
        if recall_selector != "none" and working_set_mode != "host_backed":
            raise ValueError(
                f"{self.name} recall_selector requires working_set_mode='host_backed'"
            )
        recall_fraction = float(out.get("recall_budget_fraction", 0.0))
        if recall_fraction < 0.0 or recall_fraction > 1.0:
            raise ValueError(
                f"{self.name} recall_budget_fraction must be in [0, 1], "
                f"got {recall_fraction}"
            )
        block_size = int(out.get("recall_block_size", 8))
        representatives = int(out.get("recall_representatives", 4))
        if block_size <= 0:
            raise ValueError(f"{self.name} recall_block_size must be positive")
        if representatives <= 0 or representatives > block_size:
            raise ValueError(
                f"{self.name} recall_representatives must be in [1, recall_block_size]"
            )
        out.update(
            {
                "recall_selector": recall_selector,
                "recall_budget_fraction": recall_fraction,
                "recall_block_size": block_size,
                "recall_representatives": representatives,
            }
        )
        return out

    @staticmethod
    def _split_total_budget(
        total_budget: int,
        *,
        recent_fraction: float,
        recall_fraction: float,
    ) -> tuple[int, int, int]:
        total_budget = max(0, int(total_budget))
        recall_budget = min(
            total_budget,
            max(0, int(round(total_budget * recall_fraction))),
        )
        persistent_budget = total_budget - recall_budget
        recent_budget = min(
            persistent_budget,
            max(0, int(round(persistent_budget * recent_fraction))),
        )
        return persistent_budget - recent_budget, recent_budget, recall_budget

    @staticmethod
    def _recall_enabled(ctx: OperatorExecutionContext) -> bool:
        return (
            str(ctx.params.get("working_set_mode", "mask_only")) == "host_backed"
            and str(ctx.params.get("recall_selector", "none")) != "none"
            and float(ctx.params.get("recall_budget_fraction", 0.0)) > 0.0
        )

    @staticmethod
    def _score_source_label(ctx: OperatorExecutionContext) -> str:
        source = str(ctx.params.get("score_source", "effective_attention"))
        if source == "original_attention":
            return "original_attention"
        if source == "pre_eviction_attention":
            return "pre_eviction_attention"
        return "final_shared_attention"

    @staticmethod
    def _align_mask(
        value: torch.Tensor | None,
        shape: tuple[int, int],
        *,
        device: torch.device,
        fill: bool,
    ) -> torch.Tensor:
        heads, key_len = shape
        aligned = torch.full(shape, fill, device=device, dtype=torch.bool)
        if value is None:
            return aligned
        copy_heads = min(heads, int(value.shape[0]))
        copy_tokens = min(key_len, int(value.shape[1]))
        aligned[:copy_heads, :copy_tokens] = value[
            :copy_heads, :copy_tokens
        ].to(device=device, dtype=torch.bool)
        return aligned

    @staticmethod
    def _align_scores(
        value: torch.Tensor,
        shape: tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        heads, key_len = shape
        aligned = torch.zeros(shape, device=device, dtype=torch.float32)
        copy_heads = min(heads, int(value.shape[0]))
        copy_tokens = min(key_len, int(value.shape[1]))
        aligned[:copy_heads, :copy_tokens] = value[
            :copy_heads, :copy_tokens
        ].to(device=device, dtype=torch.float32)
        return aligned

    @staticmethod
    def _state_key(ctx: OperatorExecutionContext) -> tuple:
        return (
            ctx.runtime.run_id,
            ctx.runtime.phase,
            ctx.runtime.branch,
            ctx.runtime.cfg_branch,
            int(ctx.runtime.layer_idx),
            int(ctx.runtime.batch_idx),
            ctx.step.segment_id,
            ctx.step.id,
        )

    @staticmethod
    def _selection_masks(ctx: OperatorExecutionContext) -> tuple[torch.Tensor, torch.Tensor]:
        key_len = int(ctx.k.shape[0])
        current_vae = torch.zeros(key_len, device=ctx.k.device, dtype=torch.bool)
        if ctx.key_type_ids is not None:
            current_vae = (
                ctx.key_type_ids.to(device=ctx.k.device) == BAGEL_KV_TYPE_CURRENT_VAE
            )
        selection_pool = ctx.segment_mask & ~current_vae
        forced_keep = ~ctx.segment_mask | current_vae
        return selection_pool, forced_keep

    def _initial_state(
        self,
        ctx: OperatorExecutionContext,
        *,
        selection_pool: torch.Tensor,
    ) -> dict:
        initial_len = int(selection_pool.sum().item())
        heavy_size = ctx.params.get("heavy_size", ctx.params.get("hh_size"))
        if heavy_size is None:
            heavy_budget = int(float(ctx.params.get("heavy_budget_ratio", 0.0)) * initial_len)
        else:
            heavy_budget = int(heavy_size)
        if "recent_size" in ctx.params:
            recent_budget = int(ctx.params["recent_size"])
        else:
            recent_budget = int(float(ctx.params.get("recent_budget_ratio", 0.0)) * initial_len)
        recent_budget = min(initial_len, max(0, recent_budget))
        heavy_budget = min(max(0, initial_len - recent_budget), max(0, heavy_budget))
        total_budget = heavy_budget + recent_budget
        recent_fraction = recent_budget / max(total_budget, 1)
        recall_fraction = float(ctx.params.get("recall_budget_fraction", 0.0))
        heavy_budget, recent_budget, recall_budget = self._split_total_budget(
            total_budget,
            recent_fraction=recent_fraction,
            recall_fraction=recall_fraction,
        )
        working_set_mode = str(ctx.params.get("working_set_mode", "mask_only"))
        monotonic_retirement = bool(ctx.params.get("monotonic_retirement", False))
        if working_set_mode == "irreversible":
            monotonic_retirement = True
        elif working_set_mode == "host_backed":
            monotonic_retirement = False
        return {
            "scores": None,
            "next_keep_mask": None,
            "heavy_budget": heavy_budget,
            "recent_budget": recent_budget,
            "recall_budget": recall_budget,
            "requested_total_budget": total_budget,
            "effective_total_budget": total_budget,
            "initial_length": initial_len,
            "score_source": self._score_source_label(ctx),
            "score_update_policy": str(
                ctx.params.get("score_update_policy", "cumulative")
            ),
            "score_observations": 0,
            "working_set_mode": working_set_mode,
            "monotonic_retirement": monotonic_retirement,
            "host_archive": None,
            "recall_metadata": None,
            "recall_metadata_builds": 0,
        }

    def _refresh_dynamic_budget(
        self,
        ctx: OperatorExecutionContext,
        state: dict,
        *,
        selection_pool: torch.Tensor,
    ) -> None:
        if not self.dynamic_budget_supported or not bool(ctx.params.get("dynamic_budget", False)):
            return
        budget_base = max(int(state["initial_length"]), int(selection_pool.sum().item()))
        target_ratio = float(
            ctx.params.get(
                "target_budget_ratio",
                float(ctx.params.get("heavy_budget_ratio", 0.0))
                + float(ctx.params.get("recent_budget_ratio", 0.0)),
            )
        )
        requested_total_budget = min(
            budget_base,
            max(1 if target_ratio > 0.0 else 0, int(round(target_ratio * budget_base))),
        )
        monotonic_retirement = bool(state.get("monotonic_retirement", False))
        total_budget = requested_total_budget
        if monotonic_retirement and state.get("next_keep_mask") is not None:
            previous_keep = self._align_mask(
                state["next_keep_mask"],
                (int(ctx.q.shape[1]), int(ctx.k.shape[0])),
                device=ctx.k.device,
                fill=False,
            )
            retained_per_head = (
                previous_keep & selection_pool.unsqueeze(0)
            ).sum(dim=-1)
            if int(retained_per_head.numel()) > 0:
                total_budget = min(total_budget, int(retained_per_head.min().item()))
        recent_fraction = (
            float(ctx.params.get("recent_budget_ratio", 0.0)) / target_ratio
            if target_ratio > 0.0
            else 0.0
        )
        heavy_budget, recent_budget, recall_budget = self._split_total_budget(
            total_budget,
            recent_fraction=recent_fraction,
            recall_fraction=float(ctx.params.get("recall_budget_fraction", 0.0)),
        )
        state.update(
            {
                "heavy_budget": heavy_budget,
                "recent_budget": recent_budget,
                "recall_budget": recall_budget,
                "dynamic_target_budget_ratio": target_ratio,
                "requested_total_budget": requested_total_budget,
                "effective_total_budget": total_budget,
                "budget_shortfall_tokens": requested_total_budget - total_budget,
                "monotonic_retirement": monotonic_retirement,
            }
        )

    def _apply_refreshed_budget_to_cached_scores(
        self,
        ctx: OperatorExecutionContext,
        state: dict,
        *,
        selection_pool: torch.Tensor,
        forced_keep: torch.Tensor,
    ) -> None:
        """Project scores collected on the previous call to the new budget."""

        if not bool(ctx.params.get("dynamic_budget", False)):
            return
        scores = state.get("scores")
        previous_keep = state.get("next_keep_mask")
        if scores is None or previous_keep is None:
            return
        num_heads = int(ctx.q.shape[1])
        key_len = int(ctx.k.shape[0])
        cumulative = self._align_scores(
            scores,
            (num_heads, key_len),
            device=ctx.k.device,
        )
        candidate_pool = selection_pool.unsqueeze(0).expand(num_heads, -1).clone()
        if bool(state.get("monotonic_retirement", False)):
            candidate_pool &= self._align_mask(
                previous_keep,
                (num_heads, key_len),
                device=ctx.k.device,
                fill=False,
            )
        recent_budget = int(state["recent_budget"])
        heavy_budget = int(state["heavy_budget"])
        next_keep = forced_keep.unsqueeze(0).expand(num_heads, -1).clone()
        for head_idx in range(num_heads):
            pool_idx = torch.nonzero(candidate_pool[head_idx], as_tuple=False).flatten()
            recent_k = min(int(pool_idx.numel()), recent_budget)
            if recent_k > 0:
                recent_idx = pool_idx[-recent_k:]
                history_idx = pool_idx[:-recent_k]
                next_keep[head_idx, recent_idx] = True
            else:
                history_idx = pool_idx
            heavy_k = min(int(history_idx.numel()), heavy_budget)
            if heavy_k > 0:
                heavy_local = torch.topk(
                    cumulative[head_idx, history_idx], k=heavy_k, largest=True
                ).indices
                next_keep[head_idx, history_idx[heavy_local]] = True
        state["next_keep_mask"] = next_keep.detach()

    @staticmethod
    def _selection_mask_matches_archive(
        archived: torch.Tensor,
        current: torch.Tensor,
    ) -> bool:
        archived = archived.to(device="cpu", dtype=torch.bool)
        current = current.detach().to(device="cpu", dtype=torch.bool)
        if int(current.numel()) < int(archived.numel()):
            return False
        if not torch.equal(current[: archived.numel()], archived):
            return False
        return not bool(current[archived.numel() :].any())

    def _ensure_block_recall_metadata(
        self,
        ctx: OperatorExecutionContext,
        state: dict,
        *,
        selection_pool: torch.Tensor,
    ) -> dict | None:
        if not self._recall_enabled(ctx):
            return None
        cached = state.get("recall_metadata")
        if cached is not None:
            if not self._selection_mask_matches_archive(
                cached["selection_mask"], selection_pool
            ):
                raise ValueError(
                    "Block recall metadata cannot be reused after conditioning "
                    "token identity changes"
                )
            return cached

        block_size = int(ctx.params.get("recall_block_size", 8))
        num_representatives = int(ctx.params.get("recall_representatives", 4))
        token_indices = torch.nonzero(selection_pool, as_tuple=False).flatten()
        if int(token_indices.numel()) == 0:
            return None

        blocks = []
        representatives = []
        metadata_bytes = 0
        for start in range(0, int(token_indices.numel()), block_size):
            block = token_indices[
                start : min(start + block_size, int(token_indices.numel()))
            ]
            block_keys = ctx.k.detach()[block].float()
            subblocks = torch.tensor_split(
                block_keys,
                min(num_representatives, int(block_keys.shape[0])),
                dim=0,
            )
            block_representatives = torch.stack(
                [subblock.mean(dim=0) for subblock in subblocks],
                dim=0,
            ).to(dtype=ctx.k.dtype)
            blocks.append(block.detach().to(device="cpu", dtype=torch.long))
            representatives.append(block_representatives)
            metadata_bytes += int(
                block_representatives.numel() * block_representatives.element_size()
            )
            metadata_bytes += int(block.numel() * 4)

        cached = {
            "selection_mask": selection_pool.detach().to(device="cpu", dtype=torch.bool),
            "blocks": blocks,
            "representatives": representatives,
            "metadata_bytes": metadata_bytes,
            "block_size": block_size,
            "num_representatives": num_representatives,
        }
        state["recall_metadata"] = cached
        state["recall_metadata_builds"] = int(
            state.get("recall_metadata_builds", 0)
        ) + 1
        return cached

    def _apply_block_recall(
        self,
        ctx: OperatorExecutionContext,
        state: dict,
        *,
        current_keep: torch.Tensor,
        selection_pool: torch.Tensor,
    ) -> torch.Tensor:
        metadata = self._ensure_block_recall_metadata(
            ctx,
            state,
            selection_pool=selection_pool,
        )
        if metadata is None or state.get("next_keep_mask") is None:
            state["current_recall_blocks"] = 0
            state["current_recall_head_tokens"] = 0
            return current_keep

        selector = str(ctx.params.get("recall_selector", "none"))
        if selector != "block_centroid":
            raise ValueError(f"Unsupported recall selector: {selector}")
        recall_budget = int(state.get("recall_budget", 0))
        total_budget = int(
            state.get(
                "effective_total_budget",
                int(state.get("heavy_budget", 0))
                + int(state.get("recent_budget", 0))
                + recall_budget,
            )
        )
        if recall_budget <= 0 or total_budget <= 0:
            return current_keep

        query = ctx.q.detach().float().mean(dim=0)
        block_scores = []
        for block_representatives in metadata["representatives"]:
            representatives = block_representatives.to(
                device=query.device,
                dtype=query.dtype,
            )
            if int(representatives.shape[1]) != int(query.shape[0]):
                if int(query.shape[0]) % int(representatives.shape[1]) != 0:
                    raise ValueError(
                        "Block recall requires query heads divisible by KV heads"
                    )
                representatives = representatives.repeat_interleave(
                    int(query.shape[0]) // int(representatives.shape[1]),
                    dim=1,
                )
            representatives = representatives.permute(1, 0, 2)
            block_scores.append(
                (query[:, None, :] * representatives)
                .sum(dim=-1)
                .amax(dim=1)
                .mean()
            )

        blocks = metadata["blocks"]
        if not block_scores:
            return current_keep
        ranked_blocks = torch.argsort(torch.stack(block_scores), descending=True)
        next_keep = current_keep.clone()
        persistent = (next_keep & selection_pool.unsqueeze(0)).sum(dim=-1)
        remaining = torch.clamp(total_budget - persistent, min=0, max=recall_budget)
        selected_blocks = 0
        selected_head_tokens = 0
        for block_index in ranked_blocks.tolist():
            block = blocks[int(block_index)].to(device=next_keep.device)
            new_tokens = ~next_keep[:, block]
            new_counts = new_tokens.sum(dim=-1)
            eligible = (new_counts > 0) & (new_counts <= remaining)
            if not bool(eligible.any()):
                continue
            eligible_heads = torch.nonzero(eligible, as_tuple=False).flatten()
            for head_idx in eligible_heads.tolist():
                add = new_tokens[head_idx]
                next_keep[head_idx, block[add]] = True
                added = int(add.sum().item())
                remaining[head_idx] -= added
                selected_head_tokens += added
            selected_blocks += 1
            if not bool((remaining > 0).any()):
                break

        state["current_recall_blocks"] = selected_blocks
        state["current_recall_head_tokens"] = selected_head_tokens
        state["current_recall_mask"] = (next_keep & ~current_keep).detach()
        return next_keep

    def execute(self, ctx: OperatorExecutionContext) -> OperatorExecutionContext:
        key_len = int(ctx.k.shape[0])
        num_heads = int(ctx.q.shape[1])
        if key_len == 0:
            return ctx
        selection_pool, forced_keep = self._selection_masks(ctx)
        state_key = self._state_key(ctx)
        state = ctx.state.get(state_key)
        if state is None:
            state = self._initial_state(ctx, selection_pool=selection_pool)
            ctx.state[state_key] = state
        self._refresh_dynamic_budget(ctx, state, selection_pool=selection_pool)
        self._apply_refreshed_budget_to_cached_scores(
            ctx,
            state,
            selection_pool=selection_pool,
            forced_keep=forced_keep,
        )

        current_keep = self._align_mask(
            state.get("next_keep_mask"),
            (num_heads, key_len),
            device=ctx.k.device,
            fill=True,
        )
        current_keep[:, forced_keep] = True
        current_keep = self._apply_block_recall(
            ctx,
            state,
            current_keep=current_keep,
            selection_pool=selection_pool,
        )
        ctx.attention_keep_mask_override = current_keep
        state["current_keep_mask"] = current_keep.detach()
        return ctx

    def update_after_attention(
        self,
        ctx: OperatorExecutionContext,
        attention_probs: torch.Tensor,
    ) -> OperatorExecutionContext:
        key_len = int(ctx.k.shape[0])
        num_heads = int(ctx.q.shape[1])
        selection_pool, forced_keep = self._selection_masks(ctx)
        state_key = self._state_key(ctx)
        state = ctx.state[state_key]

        current_keep = self._align_mask(
            state.get("current_keep_mask"),
            (num_heads, key_len),
            device=ctx.k.device,
            fill=True,
        )
        monotonic_retirement = bool(state.get("monotonic_retirement", False))
        candidate_pool = selection_pool.unsqueeze(0).expand(num_heads, -1)
        if monotonic_retirement:
            candidate_pool = candidate_pool & current_keep

        current_scores = attention_probs.float().sum(dim=1).detach()
        if monotonic_retirement:
            current_scores = current_scores * candidate_pool.to(current_scores.dtype)
        old_scores = state.get("scores")
        score_update_policy = str(
            state.get("score_update_policy", "cumulative")
        )
        reuse_first_observation = (
            score_update_policy == "first_observation" and old_scores is not None
        )
        if reuse_first_observation:
            cumulative = self._align_scores(
                old_scores,
                (num_heads, key_len),
                device=ctx.k.device,
            )
        else:
            cumulative = current_scores.clone()
        if old_scores is not None and not reuse_first_observation:
            copy_heads = min(num_heads, int(old_scores.shape[0]))
            copy_tokens = min(key_len, int(old_scores.shape[1]))
            cumulative[:copy_heads, :copy_tokens] += old_scores[
                :copy_heads, :copy_tokens
            ].to(device=cumulative.device, dtype=cumulative.dtype)

        recent_budget = int(state["recent_budget"])
        heavy_budget = int(state["heavy_budget"])
        next_keep = forced_keep.unsqueeze(0).expand(num_heads, -1).clone()
        for head_idx in range(num_heads):
            pool_idx = torch.nonzero(candidate_pool[head_idx], as_tuple=False).flatten()
            recent_k = min(int(pool_idx.numel()), recent_budget)
            if recent_k > 0:
                recent_idx = pool_idx[-recent_k:]
                history_idx = pool_idx[:-recent_k]
                next_keep[head_idx, recent_idx] = True
            else:
                history_idx = pool_idx
            heavy_k = min(int(history_idx.numel()), heavy_budget)
            if heavy_k > 0:
                heavy_local = torch.topk(
                    cumulative[head_idx, history_idx], k=heavy_k, largest=True
                ).indices
                next_keep[head_idx, history_idx[heavy_local]] = True

        if score_update_policy == "first_observation":
            stored_scores = cumulative
        else:
            stored_scores = cumulative * (
                next_keep & selection_pool.unsqueeze(0)
            ).to(cumulative.dtype)
        state.update(
            {
                "scores": stored_scores.detach(),
                "next_keep_mask": next_keep.detach(),
                "last_attention_scores": current_scores.detach(),
                "score_source": self._score_source_label(ctx),
                "score_update_policy": score_update_policy,
                "score_observations": int(state.get("score_observations", 0))
                + (0 if reuse_first_observation else 1),
            }
        )

        # H2O is mask-only in this prototype. Report the storage that a packed
        # implementation would require so it can be budget-matched to KIVI
        # without implying a realized allocator saving. The primary estimate
        # preserves H2O's per-query-head selections while normalizing KV
        # payload against BAGEL's compact GQA storage. A second estimate takes
        # the union within every GQA group and is a realizable upper bound for
        # a conventional per-KV-head cache layout.
        segment_tokens = int(state["initial_length"])
        storage_num_kv_heads = int(
            ctx.storage.layout.num_kv_heads
            if ctx.storage is not None
            else (ctx.storage_num_kv_heads or num_heads)
        )
        head_dim = int(ctx.k.shape[-1])
        elem_bytes = int(ctx.k.element_size())
        working_set_keep = (
            current_keep if self._recall_enabled(ctx) else next_keep
        )
        selected_per_query_head = working_set_keep & selection_pool.unsqueeze(0)
        kept_head_tokens = int(selected_per_query_head.sum().item())
        mean_keep_ratio = float(
            kept_head_tokens / max(num_heads * segment_tokens, 1)
        )
        original_storage = float(
            segment_tokens * storage_num_kv_heads * head_dim * 2 * elem_bytes
        )
        ideal_payload = float(original_storage * mean_keep_ratio)

        # A packed H2O implementation needs cumulative scores and token
        # indices. Count one float32 score and one int32 index for every kept
        # query-head/token pair.
        score_bytes = float(kept_head_tokens * 4)
        index_bytes = float(kept_head_tokens * 4)
        metadata_bytes = score_bytes + index_bytes

        if num_heads % storage_num_kv_heads == 0:
            repeat = num_heads // storage_num_kv_heads
            grouped = selected_per_query_head.reshape(
                storage_num_kv_heads, repeat, key_len
            )
            union_head_tokens = int(grouped.any(dim=1).sum().item())
            gqa_union_payload = float(
                union_head_tokens * head_dim * 2 * elem_bytes
            )
        else:
            repeat = 1
            union_head_tokens = kept_head_tokens
            gqa_union_payload = ideal_payload

        hierarchy_transition = None
        if state.get("working_set_mode") in {"irreversible", "host_backed"}:
            tracker = state.get("working_set_tracker")
            if tracker is None:
                tracker = LogicalHostArchive(
                    token_mask=selection_pool,
                    num_query_heads=num_heads,
                    num_kv_heads=storage_num_kv_heads,
                    head_dim=head_dim,
                    element_size=elem_bytes,
                )
                state["working_set_tracker"] = tracker
                if state.get("working_set_mode") == "host_backed":
                    state["host_archive"] = tracker
            hierarchy_transition = tracker.materialize(
                selected_per_query_head,
                token_mask=selection_pool,
            )

        storage_snapshot = {
                "segment_tokens": segment_tokens,
                "attention_num_heads": num_heads,
                "storage_num_kv_heads": storage_num_kv_heads,
                "gqa_repeat": repeat,
                "selected_query_head_tokens": kept_head_tokens,
                "gqa_union_kv_head_tokens": union_head_tokens,
                "original_storage_bytes": original_storage,
                "ideal_packed_payload_bytes": ideal_payload,
                "ideal_score_metadata_bytes": score_bytes,
                "ideal_index_metadata_bytes": index_bytes,
                "ideal_metadata_bytes": metadata_bytes,
                "ideal_total_storage_bytes": ideal_payload + metadata_bytes,
                "gqa_union_payload_bytes": gqa_union_payload,
                "gqa_union_total_storage_bytes": gqa_union_payload + metadata_bytes,
        }
        if hierarchy_transition is not None:
            storage_snapshot.update(
                {
                    "full_reference_bytes": float(
                        state["working_set_tracker"].archive_bytes
                    ),
                    "gpu_working_set_bytes": float(hierarchy_transition.resident_bytes),
                    "gpu_working_set_ratio": float(hierarchy_transition.resident_ratio),
                    "restored_logical_tokens": float(
                        hierarchy_transition.logical_restored_tokens
                    ),
                    "evicted_logical_tokens": float(
                        hierarchy_transition.logical_evicted_tokens
                    ),
                    "retained_logical_tokens": float(
                        hierarchy_transition.logical_retained_tokens
                    ),
                    "theoretical_h2d_bytes": float(hierarchy_transition.h2d_bytes),
                }
            )
        ctx.set_storage_snapshot(storage_snapshot)
        ctx.decision = SelectionDecision(
            token_count=key_len,
            scores=cumulative.mean(dim=0),
            budget=int(
                state.get(
                    "effective_total_budget",
                    heavy_budget + recent_budget + int(state.get("recall_budget", 0)),
                )
            ),
            reason="h2o_heavy_hitter_plus_recent_next_mask",
            metadata={
                "heavy_budget": heavy_budget,
                "recent_budget": recent_budget,
                "recall_budget": int(state.get("recall_budget", 0)),
                "recall_selector": str(ctx.params.get("recall_selector", "none")),
                "recall_block_size": int(ctx.params.get("recall_block_size", 8)),
                "recall_representatives": int(ctx.params.get("recall_representatives", 4)),
                "recall_metadata_builds": int(state.get("recall_metadata_builds", 0)),
                "recall_metadata_bytes": int(
                    (state.get("recall_metadata") or {}).get("metadata_bytes", 0)
                ),
                "current_recall_blocks": int(state.get("current_recall_blocks", 0)),
                "current_recall_head_tokens": int(
                    state.get("current_recall_head_tokens", 0)
                ),
                "initial_length": int(state["initial_length"]),
                "per_head_kept_tokens": [
                    int(item) for item in next_keep.sum(dim=-1).tolist()
                ],
                "applies_on_next_call": True,
                "physical_eviction": False,
                "score_source": self._score_source_label(ctx),
                "score_update_policy": score_update_policy,
                "score_observations": int(state.get("score_observations", 0)),
                "monotonic_retirement": monotonic_retirement,
                "working_set_mode": str(state.get("working_set_mode", "mask_only")),
                "requested_total_budget": int(
                    state.get("requested_total_budget", heavy_budget + recent_budget)
                ),
                "effective_total_budget": int(
                    state.get("effective_total_budget", heavy_budget + recent_budget)
                ),
                "budget_shortfall_tokens": int(state.get("budget_shortfall_tokens", 0)),
                "adaptation": self.adaptation_name,
            },
        )
        current_keep = state["current_keep_mask"]
        ctx.record(
            {
                "h2o_initial_tokens": int(state["initial_length"]),
                "h2o_heavy_budget": heavy_budget,
                "h2o_recent_budget": recent_budget,
                "h2o_recall_budget": int(state.get("recall_budget", 0)),
                "h2o_masked_tokens": int((~current_keep).sum().item()),
                "h2o_next_kept_head_tokens": int(next_keep.sum().item()),
                "h2o_current_working_set_head_tokens": int(working_set_keep.sum().item()),
                "h2o_recall_blocks": int(state.get("current_recall_blocks", 0)),
                "h2o_recall_head_tokens": int(
                    state.get("current_recall_head_tokens", 0)
                ),
                "h2o_requested_budget": int(
                    state.get("requested_total_budget", heavy_budget + recent_budget)
                ),
                "h2o_effective_budget": int(
                    state.get("effective_total_budget", heavy_budget + recent_budget)
                ),
                "h2o_budget_shortfall_tokens": int(state.get("budget_shortfall_tokens", 0)),
                "hierarchy_restored_tokens": int(
                    hierarchy_transition.logical_restored_tokens
                    if hierarchy_transition is not None
                    else 0
                ),
                "hierarchy_evicted_tokens": int(
                    hierarchy_transition.logical_evicted_tokens
                    if hierarchy_transition is not None
                    else 0
                ),
                "hierarchy_retained_tokens": int(
                    hierarchy_transition.logical_retained_tokens
                    if hierarchy_transition is not None
                    else 0
                ),
                "hierarchy_h2d_bytes": int(
                    hierarchy_transition.h2d_bytes
                    if hierarchy_transition is not None
                    else 0
                ),
            }
        )
        return ctx


class H2OAttentionMaskOperator(_H2OAttentionMaskBase):
    """Original H2O heavy-hitter + recent policy over one global KV segment."""

    name = "h2o_attention_mask"
    adaptation_name = "original_global_h2o_mask_only"

    def supports(self, segment, ctx) -> bool:
        del ctx
        return segment.pattern_id == "whole_conditioning_kv"


class H2OSegmentAttentionMaskOperator(_H2OAttentionMaskBase):
    """UniCache adaptation that runs independent H2O state per semantic segment."""

    name = "h2o_segment_attention_mask"
    adaptation_name = "unicache_segment_aware_h2o"
    dynamic_budget_supported = True
