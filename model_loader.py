"""BAGEL checkpoint loading and device helpers."""

from __future__ import annotations

import argparse

from pathlib import Path

import torch

from torch import nn

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from huggingface_hub import snapshot_download

from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb

from data.transforms import ImageTransform

from inferencer import InterleaveInferencer

from modeling.autoencoder import load_ae

from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from modeling.qwen2 import Qwen2Tokenizer

REPO_ID = "ByteDance-Seed/BAGEL-7B-MoT"

ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"]

class VaeInputDtypeAdapter(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    @property
    def input_dtype(self) -> torch.dtype:
        return next(self.module.parameters()).dtype

    @property
    def input_device(self) -> torch.device:
        return next(self.module.parameters()).device

    def _autocast_disabled(self, x: torch.Tensor):
        if x.device.type in {"cuda", "mps"}:
            return torch.autocast(device_type=x.device.type, enabled=False)
        return torch.autocast(device_type="cpu", enabled=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with self._autocast_disabled(x):
            return self.module.encode(x.to(device=self.input_device, dtype=self.input_dtype))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with self._autocast_disabled(z):
            return self.module.decode(z.to(device=self.input_device, dtype=self.input_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._autocast_disabled(x):
            return self.module(x.to(device=self.input_device, dtype=self.input_dtype))

def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu" or name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    return torch.bfloat16

def ensure_model(model_path: Path, skip_download: bool) -> Path:
    required = ["llm_config.json", "vit_config.json", "ae.safetensors", "ema.safetensors"]
    if all((model_path / name).exists() for name in required):
        return model_path
    if skip_download:
        missing = [name for name in required if not (model_path / name).exists()]
        raise FileNotFoundError(f"Missing model files under {model_path}: {', '.join(missing)}")
    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(model_path),
        cache_dir=str(model_path / "cache"),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=ALLOW_PATTERNS,
    )
    return model_path

def build_model(model_path: Path, device: torch.device, dtype: torch.dtype) -> tuple[Bagel, object]:
    llm_config = Qwen2Config.from_json_file(model_path / "llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    if not hasattr(llm_config, "pad_token_id") or llm_config.pad_token_id is None:
        llm_config.pad_token_id = llm_config.eos_token_id

    vit_config = SiglipVisionConfig.from_json_file(model_path / "vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))
    vae_model = VaeInputDtypeAdapter(vae_model.to(device=device, dtype=torch.float32).eval())

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(model_path / "ema.safetensors"),
        device_map={"": device.type},
        dtype=dtype,
        offload_buffers=True,
        offload_folder=str(model_path / "offload"),
        force_hooks=True,
    ).eval()
    return model, vae_model
