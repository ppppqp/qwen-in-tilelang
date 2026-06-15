from __future__ import annotations

import pytest
import torch

from qwen_inference.qwen import (
    Qwen3MLP,
    Qwen3MultiHeadAttention,
    Qwen3TransformerBlock,
)


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() == 0, reason="TileLang kernels need CUDA"
)


def _module_dims() -> tuple[int, int, int, int, int]:
    hidden_size = 128
    num_attention_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_attention_heads
    intermediate_size = hidden_size
    return hidden_size, num_attention_heads, num_kv_heads, head_dim, intermediate_size


def _zeros(shape: tuple[int, ...], device: str) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float16, device=device)


def _ones(shape: tuple[int, ...], device: str) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.float16, device=device)


def test_qwen3_mlp_zero_weights_outputs_zero():
    device = "cuda"
    hidden_size, _, _, _, intermediate_size = _module_dims()
    mlp = Qwen3MLP(
        dim=hidden_size,
        hidden_dim=intermediate_size,
        w_gate=_zeros((hidden_size, intermediate_size), device),
        w_up=_zeros((hidden_size, intermediate_size), device),
        w_down=_zeros((intermediate_size, hidden_size), device),
    )
    x = torch.randn((16, 16, hidden_size), dtype=torch.float16, device=device)

    output = mlp(x)

    assert output.shape == x.shape
    assert torch.allclose(output, torch.zeros_like(output))
    assert not torch.isnan(output).any()


def test_qwen3_multi_head_attention_zero_weights_outputs_zero():
    device = "cuda"
    hidden_size, num_attention_heads, num_kv_heads, head_dim, _ = _module_dims()
    kv_hidden_size = num_kv_heads * head_dim
    attention = Qwen3MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        wq=_zeros((hidden_size, hidden_size), device),
        wk=_zeros((hidden_size, kv_hidden_size), device),
        wv=_zeros((hidden_size, kv_hidden_size), device),
        wo=_zeros((hidden_size, hidden_size), device),
        q_norm=_ones((head_dim,), device),
        k_norm=_ones((head_dim,), device),
    )
    x = torch.randn((16, 16, hidden_size), dtype=torch.float16, device=device)

    output = attention(x, is_causal=True)

    assert output.shape == x.shape
    assert torch.allclose(output, torch.zeros_like(output))
    assert not torch.isnan(output).any()


def test_qwen3_transformer_block_zero_weights_preserves_residual():
    device = "cuda"
    hidden_size, num_attention_heads, num_kv_heads, head_dim, intermediate_size = (
        _module_dims()
    )
    kv_hidden_size = num_kv_heads * head_dim
    block = Qwen3TransformerBlock(
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        rms_norm_eps=1e-5,
        wq=_zeros((hidden_size, hidden_size), device),
        wk=_zeros((hidden_size, kv_hidden_size), device),
        wv=_zeros((hidden_size, kv_hidden_size), device),
        wo=_zeros((hidden_size, hidden_size), device),
        q_norm=_ones((head_dim,), device),
        k_norm=_ones((head_dim,), device),
        w_gate=_zeros((hidden_size, intermediate_size), device),
        w_up=_zeros((hidden_size, intermediate_size), device),
        w_down=_zeros((intermediate_size, hidden_size), device),
        w_input_layernorm=_ones((hidden_size,), device),
        w_post_attention_layernorm=_ones((hidden_size,), device),
    )
    x = torch.randn((16, 16, hidden_size), dtype=torch.float16, device=device)

    output = block(x, is_causal=True)

    assert output.shape == x.shape
    assert torch.allclose(output, x)
    assert not torch.isnan(output).any()
