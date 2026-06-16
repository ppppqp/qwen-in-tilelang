from __future__ import annotations

import pytest
import torch

from qwen_inference.generate import _pad_tokens_to_multiple, simple_generate
from qwen_inference.qwen import Qwen3Model, Qwen3ModelConfig


class FakeTokenizer:
    eos_token_id = 5

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text == "hello"
        assert not add_special_tokens
        return [1, 2] * 16

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens
        pieces = {3: "A", 4: "B", self.eos_token_id: ""}
        return "".join(pieces[token_id] for token_id in token_ids)


def test_pad_tokens_to_multiple_right_pads_with_token_zero():
    tokens = torch.tensor([1, 2, 3], dtype=torch.long)

    padded = _pad_tokens_to_multiple(tokens, multiple=16)

    assert padded.tolist() == [1, 2, 3] + [0] * 13
    assert padded.dtype == tokens.dtype


def test_pad_tokens_to_multiple_returns_original_when_already_aligned():
    tokens = torch.arange(16, dtype=torch.long)

    padded = _pad_tokens_to_multiple(tokens, multiple=16)

    assert padded is tokens


def _tiny_qwen3_model(device: str) -> Qwen3Model:
    vocab_size = 128
    hidden_size = 128
    dtype = torch.float16

    embedding = torch.eye(vocab_size, hidden_size, dtype=dtype, device=device)

    # HF stores lm_head as (vocab_size, hidden_size). Qwen3Model transposes it
    # to the local TileLang linear layout (hidden_size, vocab_size).
    lm_head = torch.full((vocab_size, hidden_size), -10.0, dtype=dtype, device=device)
    lm_head[3, 2] = 10.0
    lm_head[4, 3] = 10.0
    lm_head[5, 4] = 10.0

    num_attention_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_attention_heads
    kv_hidden_size = num_kv_heads * head_dim

    config = Qwen3ModelConfig(
        num_hidden_layers=1,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=hidden_size,
        rms_norm_eps=1e-5,
        head_dim=head_dim,
        tie_word_embeddings=False,
    )
    state_dict = {
        "model.embed_tokens.weight": embedding,
        "model.norm.weight": torch.ones(hidden_size, dtype=dtype, device=device),
        "lm_head.weight": lm_head,
    }
    layer_prefix = "model.layers.0"
    state_dict.update(
        {
            f"{layer_prefix}.self_attn.q_proj.weight": torch.zeros(
                (hidden_size, hidden_size), dtype=dtype, device=device
            ),
            f"{layer_prefix}.self_attn.k_proj.weight": torch.zeros(
                (kv_hidden_size, hidden_size), dtype=dtype, device=device
            ),
            f"{layer_prefix}.self_attn.v_proj.weight": torch.zeros(
                (kv_hidden_size, hidden_size), dtype=dtype, device=device
            ),
            f"{layer_prefix}.self_attn.o_proj.weight": torch.zeros(
                (hidden_size, hidden_size), dtype=dtype, device=device
            ),
            f"{layer_prefix}.self_attn.q_norm.weight": torch.ones(
                head_dim, dtype=dtype, device=device
            ),
            f"{layer_prefix}.self_attn.k_norm.weight": torch.ones(
                head_dim, dtype=dtype, device=device
            ),
            f"{layer_prefix}.mlp.gate_proj.weight": torch.zeros(
                (hidden_size, hidden_size), dtype=dtype, device=device
            ),
            f"{layer_prefix}.mlp.up_proj.weight": torch.zeros(
                (hidden_size, hidden_size), dtype=dtype, device=device
            ),
            f"{layer_prefix}.mlp.down_proj.weight": torch.zeros(
                (hidden_size, hidden_size), dtype=dtype, device=device
            ),
            f"{layer_prefix}.input_layernorm.weight": torch.ones(
                hidden_size, dtype=dtype, device=device
            ),
            f"{layer_prefix}.post_attention_layernorm.weight": torch.ones(
                hidden_size, dtype=dtype, device=device
            ),
        }
    )
    return Qwen3Model(config, state_dict)


def test_simple_generate_with_real_qwen3_model_and_mock_weights(capsys):
    device = "cuda"
    model = _tiny_qwen3_model(device)
    tokenizer = FakeTokenizer()

    text = simple_generate(
        model,
        tokenizer,
        "hello",
        sampler=None,
        device=device,
        max_new_tokens=8,
    )

    assert text == "AB"
    assert "AB" in capsys.readouterr().out
