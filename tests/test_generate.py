from __future__ import annotations

import pytest
import torch

from qwen_inference.generate import simple_generate
from qwen_inference.qwen import Qwen3Model, Qwen3ModelConfig


class FakeTokenizer:
    eos_token_id = 5

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text == "hello"
        assert not add_special_tokens
        return [1, 2]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens
        pieces = {3: "A", 4: "B", self.eos_token_id: ""}
        return "".join(pieces[token_id] for token_id in token_ids)


def _tiny_qwen3_model(device: str) -> Qwen3Model:
    vocab_size = 6
    hidden_size = 6
    dtype = torch.float16

    embedding = torch.eye(vocab_size, hidden_size, dtype=dtype, device=device)

    # HF stores lm_head as (vocab_size, hidden_size). Qwen3Model transposes it
    # to the local TileLang linear layout (hidden_size, vocab_size).
    lm_head = torch.full((vocab_size, hidden_size), -10.0, dtype=dtype, device=device)
    lm_head[3, 2] = 10.0
    lm_head[4, 3] = 10.0
    lm_head[5, 4] = 10.0

    config = Qwen3ModelConfig(
        num_hidden_layers=0,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_attention_heads=1,
        num_kv_heads=1,
        intermediate_size=hidden_size,
        rms_norm_eps=1e-5,
        head_dim=hidden_size,
        tie_word_embeddings=False,
    )
    state_dict = {
        "model.embed_tokens.weight": embedding,
        "model.norm.weight": torch.ones(hidden_size, dtype=dtype, device=device),
        "lm_head.weight": lm_head,
    }
    return Qwen3Model(config, state_dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TileLang kernels need CUDA")
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
