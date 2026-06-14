from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from .qwen import Qwen3Model, Qwen3ModelConfig


def load_qwen3_config(model_dir: str | Path) -> Qwen3ModelConfig:
    model_dir = Path(model_dir)
    with (model_dir / "config.json").open() as f:
        config = json.load(f)

    num_attention_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)
    return Qwen3ModelConfig(
        num_hidden_layers=config["num_hidden_layers"],
        hidden_size=hidden_size,
        vocab_size=config["vocab_size"],
        num_attention_heads=num_attention_heads,
        num_kv_heads=config.get("num_key_value_heads", num_attention_heads),
        intermediate_size=config["intermediate_size"],
        rms_norm_eps=config["rms_norm_eps"],
        head_dim=head_dim,
        max_position_embeddings=config.get("max_position_embeddings", 32768),
        rope_theta=config.get("rope_theta", 1000000),
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    )


def load_safetensors_state_dict(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = torch.float16,
) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    shard_paths = sorted(model_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        shard = load_file(str(shard_path), device=str(device))
        for name, tensor in shard.items():
            if dtype is not None and torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=dtype)
            state_dict[name] = tensor.contiguous()
    return state_dict


def load_qwen3_model_from_files(
    model_dir: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = torch.float16,
) -> Qwen3Model:
    config = load_qwen3_config(model_dir)
    state_dict = load_safetensors_state_dict(model_dir, device=device, dtype=dtype)
    return Qwen3Model(config, state_dict)
