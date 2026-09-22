#!/usr/bin/env python3
"""BAGEL inference with Full KV, UniCache Torch, or the physical CUDA engine."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from model_loader import build_model, choose_device, choose_dtype, ensure_model
from inferencer import InterleaveInferencer
from modeling.bagel.qwen2_navit import clear_topk_kv_policy, set_topk_kv_policy
from modeling.qwen2 import Qwen2Tokenizer
from unicache.adapters import compile_plan_only_hook_policy, compile_unicache_hook_policy
from unicache.runtime import build_plan_from_config, load_config, write_plan_bundle
from unicache.efficiency import EfficiencyRecorder


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--task-type", dest="task_type", choices=("understanding", "text_to_image", "editing"), default="understanding")
    parser.add_argument("--backend", choices=("torch", "engine", "full"), default="torch")
    parser.add_argument("--plan-only", action="store_true", help="Compile the config without loading weights or running inference.")
    parser.add_argument("--model-path", type=Path, default=root / "checkpoints" / "BAGEL-7B-MoT")
    parser.add_argument("--image", type=Path, help="Required for understanding and editing.")
    parser.add_argument(
        "--repeat-image-count",
        type=int,
        default=1,
        help=(
            "Repeat the input image within one multimodal context. Supported for "
            "understanding and editing; editing uses every repeated image as source conditioning."
        ),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, required=True, help="Use a separate directory for each run.")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--min-image-size", type=int, default=512)
    parser.add_argument("--vit-image-size", type=int, default=980)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.5)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", choices=("global", "channel", "text_channel"), default="text_channel")
    parser.add_argument("--config", "--unicache-config", dest="unicache_config", type=Path, default=None)
    parser.add_argument("--efficiency-events", type=Path, default=None)
    parser.add_argument("--instrument-attention", action="store_true")
    parser.add_argument("--disable-runtime-diagnostics", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measure-runs", type=int, default=1)
    args = parser.parse_args()
    if args.backend == "full" and args.unicache_config is not None:
        parser.error("--backend full cannot be combined with --config")
    if args.backend != "full" and args.unicache_config is None:
        args.unicache_config = root / "configs" / args.backend / f"{args.task_type}.json"
    if args.task_type != "text_to_image" and args.image is None:
        parser.error("--image is required for understanding and editing")
    if not args.plan_only and args.image is not None and not args.image.is_file():
        parser.error(f"Input image does not exist: {args.image}")
    if args.measure_runs < 1 or args.warmup_runs < 0 or args.max_new_tokens < 1 or args.timesteps < 2:
        parser.error("Require measure-runs >= 1, warmup-runs >= 0, max-new-tokens >= 1 and timesteps >= 2")
    if args.image_size < 16 or args.image_size % 16 or args.min_image_size < 16:
        parser.error("image-size must be a positive multiple of 16; min-image-size must be >= 16")
    return args


def build_noop_config(args: argparse.Namespace) -> dict:
    image_count = args.repeat_image_count if args.task_type in {"understanding", "editing"} else 1
    source_images = [str(args.image)] * image_count if args.task_type in {"understanding", "editing"} else []
    return {
        "task": {
            "task_type": args.task_type,
            "prompt": args.prompt,
            "source_images": source_images,
            "output_modality": "text" if args.task_type == "understanding" else "image",
        },
        "plugins": ["bagel_typed", "full_kv"],
        "planners": [{"id": "bagel_typed_segments", "enabled": True}],
        "rules": [
            {
                "id": "identity_all",
                "match": {"tags": ["*"]},
                "operator": "identity",
                "priority": 0,
                "mode": "composable",
                "stage": "attention",
            }
        ],
        "runtime": {
            "plan_only": False,
            "fail_on_conflict": True,
        },
    }


def build_runtime_config(args: argparse.Namespace) -> dict:
    if args.unicache_config is None:
        return build_noop_config(args)
    config = load_config(args.unicache_config)
    physical = bool(config.get("runtime", {}).get("physical_backend", False))
    if physical != (args.backend == "engine"):
        raise ValueError("Config physical_backend must match --backend engine/torch")
    declared_task = config.get("task", {}).get("task_type")
    if declared_task and declared_task != args.task_type:
        raise ValueError(f"Config task {declared_task!r} does not match {args.task_type!r}")
    physical_operators = {"h2o_physical_gqa", "kivi_packed_quantization"}
    if args.backend == "torch" and any(r.get("operator") in physical_operators for r in config.get("rules", [])):
        raise ValueError("Physical operators require --backend engine")
    image_count = args.repeat_image_count if args.task_type in {"understanding", "editing"} else 1
    source_images = [str(args.image)] * image_count if args.task_type in {"understanding", "editing"} else []
    config["task"] = {
        "task_type": args.task_type,
        "prompt": args.prompt,
        "source_images": source_images,
        "output_modality": "text" if args.task_type == "understanding" else "image",
    }
    config.setdefault("runtime", {})
    config["runtime"].setdefault("fail_on_conflict", True)
    config["runtime"].setdefault("plan_only", False)
    return config


def main() -> None:
    args = parse_args()
    if args.repeat_image_count <= 0:
        raise ValueError("--repeat-image-count must be positive")
    if args.task_type == "text_to_image" and args.repeat_image_count != 1:
        raise ValueError("--repeat-image-count is supported only for understanding and editing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    config = build_runtime_config(args)
    bundle = build_plan_from_config(config)
    (args.output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.plan_only:
        write_plan_bundle(bundle, args.output_dir / "unicache_plan.json")
        print(f"Plan compiled: {args.output_dir / 'unicache_plan.json'} (no inference)")
        return
    if args.backend == "engine":
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The physical engine requires a CUDA GPU; use --plan-only for config checks.")
        if args.dtype == "float32":
            raise ValueError("The physical FlashAttention engine requires bfloat16 or float16.")
        from flash_attn import flash_attn_func  # Fail before loading weights.
        if any(r.get("operator") == "kivi_packed_quantization" for r in config.get("rules", [])):
            from unicache.storage.kivi_cuda import _load_official_kernel, _load_official_packer
            _load_official_kernel()
            _load_official_packer()
        os.environ.setdefault("UNICACHE_BATCH_PHYSICAL_ATTENTION", "1")
        if args.task_type != "understanding":
            os.environ.setdefault("UNICACHE_BATCH_CFG", "1")
            os.environ.setdefault("UNICACHE_CFG_BRANCH_CHUNK_SIZE", "2" if args.task_type == "text_to_image" else "3")
    if args.unicache_config is None or bool(config.get("runtime", {}).get("plan_only", False)):
        policy = compile_plan_only_hook_policy(bundle)
    else:
        policy = compile_unicache_hook_policy(bundle, enabled=True)
    if hasattr(policy, "collect_diagnostics"):
        policy.collect_diagnostics = not args.disable_runtime_diagnostics
    efficiency = EfficiencyRecorder(enabled=args.efficiency_events is not None)
    if args.efficiency_events is not None and hasattr(policy, "attach_efficiency_recorder"):
        policy.attach_efficiency_recorder(
            efficiency, instrument_attention=args.instrument_attention
        )
    set_topk_kv_policy(policy)

    plan_path = args.output_dir / "unicache_plan.json"
    summary_path = args.output_dir / "hook_policy_summary.json"
    budget_allocations_path = args.output_dir / "budget_allocations.jsonl"
    budget_by_type_path = args.output_dir / "budget_by_cache_type.json"
    budget_average_path = args.output_dir / "budget_average.json"
    metadata_path = args.output_dir / "run_metadata.json"
    answer_path = args.output_dir / "answer.txt"
    image_path = args.output_dir / "generated.png"
    transformed_input_path = args.output_dir / "input_after_vae_transform.png"
    write_plan_bundle(bundle, plan_path)

    print("UniCache recognized segments:")
    for segment in bundle.segment_plan.segments:
        print(f"- {segment.id}: tags={segment.tags}")
    print(f"UniCache plan: {plan_path}")

    try:
        device = choose_device(args.device)
        dtype = choose_dtype(args.dtype, device)
        model_path = ensure_model(args.model_path, skip_download=True)

        print(f"device: {device}")
        print(f"dtype: {dtype}")
        print(f"model_path: {model_path}")
        print(f"hook policy: {policy.__class__.__name__}(enabled={getattr(policy, 'enabled', False)})")

        efficiency.memory_snapshot("before_model_load")
        model, vae_model = build_model(model_path, device, dtype)
        efficiency.memory_snapshot("model_loaded")
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=ImageTransform(args.image_size, args.min_image_size, 16),
            vit_transform=ImageTransform(args.vit_image_size, 224, 14),
            new_token_ids=new_token_ids,
        )

        def run_once(current_state_kwargs):
            if args.task_type == "understanding":
                image = pil_img2rgb(Image.open(args.image))
                outputs = inferencer.interleave_inference(
                    [image.copy() for _ in range(args.repeat_image_count)] + [args.prompt],
                    understanding_output=True,
                    do_sample=False,
                    text_temperature=0.3,
                    max_think_token_n=args.max_new_tokens,
                )
                return {"image": None, "text": next((x for x in outputs if isinstance(x, str)), None)}
            elif args.task_type == "text_to_image":
                return inferencer(
                    text=args.prompt,
                    understanding_output=False,
                    think=False,
                    do_sample=False,
                    text_temperature=0.3,
                    max_think_token_n=args.max_new_tokens,
                    image_shapes=(args.image_size, args.image_size),
                    num_timesteps=args.timesteps,
                    cfg_text_scale=args.cfg_text_scale,
                    cfg_img_scale=args.cfg_img_scale,
                    cfg_interval=[args.cfg_interval_start, 1.0],
                    timestep_shift=args.timestep_shift,
                    cfg_renorm_min=args.cfg_renorm_min,
                    cfg_renorm_type=args.cfg_renorm_type,
                    **current_state_kwargs,
                )

            else:
                image = pil_img2rgb(Image.open(args.image))
                transformed_input = inferencer.vae_transform.resize_transform(image)
                transformed_input.save(transformed_input_path)
                editing_kwargs = dict(
                    understanding_output=False,
                    think=False,
                    do_sample=False,
                    text_temperature=0.3,
                    max_think_token_n=args.max_new_tokens,
                    num_timesteps=args.timesteps,
                    cfg_text_scale=args.cfg_text_scale,
                    cfg_img_scale=args.cfg_img_scale,
                    cfg_interval=[args.cfg_interval_start, 1.0],
                    timestep_shift=args.timestep_shift,
                    cfg_renorm_min=args.cfg_renorm_min,
                    cfg_renorm_type=args.cfg_renorm_type,
                    **current_state_kwargs,
                )
                if args.repeat_image_count == 1:
                    return inferencer(image=image, text=args.prompt, **editing_kwargs)

                outputs = inferencer.interleave_inference(
                    [image.copy() for _ in range(args.repeat_image_count)] + [args.prompt],
                    **editing_kwargs,
                )
                return {
                    "image": next((item for item in outputs if isinstance(item, Image.Image)), None),
                    "text": next((item for item in outputs if isinstance(item, str)), None),
                }

        def reset_run_seed():
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)

        for warmup_index in range(max(0, args.warmup_runs)):
            reset_run_seed()
            policy.begin_run(f"{args.task_type}-warmup-{warmup_index}")
            policy.set_context(total_steps=max(1, args.timesteps - 1))
            current_state_kwargs = policy.current_state_generation_kwargs()
            with torch.inference_mode():
                run_once(current_state_kwargs)
            if hasattr(policy, "finalize_efficiency_step"):
                policy.finalize_efficiency_step()

        if args.measure_runs <= 0:
            raise ValueError("--measure-runs must be positive")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        efficiency.memory_snapshot("before_inference")
        result = None
        for repeat_index in range(args.measure_runs):
            reset_run_seed()
            policy.begin_run(f"{args.task_type}-measure-{repeat_index}")
            policy.set_context(total_steps=max(1, args.timesteps - 1))
            current_state_kwargs = policy.current_state_generation_kwargs()
            with torch.inference_mode(), efficiency.measure(
                "end_to_end.inference",
                task_type=args.task_type,
                repeat_index=repeat_index,
            ):
                result = run_once(current_state_kwargs)
            if hasattr(policy, "finalize_efficiency_step"):
                policy.finalize_efficiency_step()
            efficiency.memory_snapshot(
                "after_inference_repeat", repeat_index=repeat_index
            )

        efficiency.memory_snapshot("after_inference")

        policy.capture_current_state_model_stats(model.language_model.model)

        answer = result.get("text") or ""
        if answer:
            answer_path.write_text(answer, encoding="utf-8")
        if result.get("image") is not None:
            result["image"].save(image_path)
        policy_summary = policy.summary()
        summary_path.write_text(json.dumps(policy_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        budget_accounting = policy_summary.get("budget_accounting", {})
        allocations = list(budget_accounting.get("allocations", []))
        budget_allocations_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in allocations),
            encoding="utf-8",
        )
        budget_by_type_path.write_text(
            json.dumps(budget_accounting.get("by_cache_type", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        budget_average_path.write_text(
            json.dumps(budget_accounting.get("average_budget", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata = {
            "backend": args.backend,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "runtime_environment": {k: v for k, v in os.environ.items() if k.startswith("UNICACHE_")},
            "task_type": args.task_type,
            "image": str(args.image) if args.image else None,
            "repeat_image_count": args.repeat_image_count,
            "prompt": args.prompt,
            "image_size": args.image_size,
            "min_image_size": args.min_image_size,
            "vit_image_size": args.vit_image_size,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
            "cfg_interval": [args.cfg_interval_start, 1.0],
            "timestep_shift": args.timestep_shift,
            "cfg_renorm_min": args.cfg_renorm_min,
            "cfg_renorm_type": args.cfg_renorm_type,
            "output_image": str(image_path) if result.get("image") is not None else None,
            "input_after_vae_transform": str(transformed_input_path) if transformed_input_path.exists() else None,
            "hook_policy_summary": str(summary_path),
            "budget_allocations": str(budget_allocations_path),
            "budget_by_cache_type": str(budget_by_type_path),
            "budget_average": str(budget_average_path),
            "unicache_plan": str(plan_path),
            "unicache_config": str(args.unicache_config) if args.unicache_config else None,
            "efficiency_events": str(args.efficiency_events) if args.efficiency_events else None,
            "instrument_attention": args.instrument_attention,
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        if answer:
            print("\nBAGEL answer:")
            print(answer)
            print(f"answer: {answer_path}")
        if result.get("image") is not None:
            print(f"generated image: {image_path}")
        if transformed_input_path.exists():
            print(f"input after VAE transform: {transformed_input_path}")
        print(f"hook summary: {summary_path}")
        print(f"budget allocations: {budget_allocations_path}")
        print(f"budget by cache type: {budget_by_type_path}")
        print(f"average budget: {budget_average_path}")
        print(f"run metadata: {metadata_path}")
    finally:
        if args.efficiency_events is not None:
            efficiency.write_jsonl(args.efficiency_events)
        clear_topk_kv_policy()


if __name__ == "__main__":
    main()
