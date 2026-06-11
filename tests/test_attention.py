from __future__ import annotations

from tests.common_test_utils import (
    kernel_tester,
)
from qwen_inference.attention import attention
import torch


def test_attention():
    def ref_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        assert len(Q.shape) == 2
        assert len(K.shape) == 2
        assert len(V.shape) == 2
        assert Q.shape[0] == K.shape[0] == V.shape[0]  # B
        assert Q.shape[1] == K.shape[1] == V.shape[1]  # S
        assert Q.dtype == K.dtype == V.dtype == torch.float32
        return torch.softmax(Q * K, dim=1).mul_(V)

    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    match = kernel_tester(
        attention,
        ref_attention,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )
    assert match, "Attention test failed!"
    print("Attention test passed!")
