from __future__ import annotations

import torch

from qwen_inference import embedding as embedding_module
from qwen_inference.embedding import Embedding


def test_embedding_as_linear_transposes_embedding_weight(monkeypatch):
    embedding = Embedding(
        vocab_size=5,
        embedding_dim=3,
        weight=torch.arange(15, dtype=torch.float32).reshape(5, 3),
    )
    x = torch.ones((2, 4, 3), dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def fake_linear(
        x_flat: torch.Tensor,
        weight: torch.Tensor,
        b: torch.Tensor | None = None,
        *,
        BLOCK_M: int = 16,
        BLOCK_N: int = 64,
        BLOCK_K: int = 64,
    ) -> torch.Tensor:
        captured["x_flat"] = x_flat
        captured["weight"] = weight
        return x_flat @ weight

    monkeypatch.setattr(embedding_module, "linear", fake_linear)

    output = embedding.as_linear(x)

    assert captured["x_flat"].shape == (8, 3)
    assert captured["weight"].shape == (3, 5)
    assert torch.equal(captured["weight"], embedding.weight.T)
    assert output.shape == (2, 4, 5)

