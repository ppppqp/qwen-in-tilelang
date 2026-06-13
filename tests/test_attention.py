from __future__ import annotations

from tests.common_test_utils import (
    kernel_tester,
)
from qwen_inference.attention import attention, causal_mask, grouped_attention
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


def test_grouped_attention():
    def ref_grouped_attention(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
    ):
        assert len(Q.shape) == 2
        assert len(K.shape) == 2
        assert len(V.shape) == 2
        assert len(mask.shape) == 2
        assert K.shape == V.shape
        assert mask.shape == K.shape
        assert Q.shape[1] == K.shape[1] == V.shape[1]  # S
        assert Q.shape[0] % K.shape[0] == 0  # QB = head_num * B
        assert Q.dtype == K.dtype == V.dtype == mask.dtype == torch.float32

        B, S = K.shape
        head_num = Q.shape[0] // B
        Q_grouped = Q.reshape(head_num, B, S)
        return (
            torch.softmax(Q_grouped * K.unsqueeze(0) + mask.unsqueeze(0), dim=2)
            * V.unsqueeze(0)
        ).reshape(Q.shape)

    QB = 512
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    Q = torch.randn((QB, S), dtype=torch.float32, device="cuda")
    K = torch.randn((B, S), dtype=torch.float32, device="cuda")
    V = torch.randn((B, S), dtype=torch.float32, device="cuda")
    mask = causal_mask(B, S, torch.float32, device=torch.device("cuda"))
    match = kernel_tester(
        grouped_attention,
        ref_grouped_attention,
        {"QB": QB, "B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
        inputs_in_torch_tensors=[Q, K, V, mask],
    )
    assert match, "Grouped attention test failed!"
    print("Grouped attention test passed!")
