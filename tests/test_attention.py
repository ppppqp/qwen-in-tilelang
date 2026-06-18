from __future__ import annotations

from tests.common_test_utils import (
    kernel_tester,
)
from qwen_inference.kernels.attention import attention_kernel
from qwen_inference.kernels.grouped_attention import grouped_attention_kernel
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="TileLang CUDA test")
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
        attention_kernel,
        ref_attention,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )
    assert match, "Attention test failed!"
    print("Attention test passed!")


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="TileLang CUDA test")
def test_grouped_attention():
    is_causal = True

    def ref_grouped_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        assert len(Q.shape) == 4
        assert len(K.shape) == 4
        assert len(V.shape) == 4
        assert K.shape == V.shape
        assert Q.dtype == K.dtype == V.dtype == torch.float16

        N, L, QH, D = Q.shape
        _, S, H, _ = K.shape
        assert QH % H == 0
        group_size = QH // H
        output = F.scaled_dot_product_attention(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            is_causal=is_causal,
            enable_gqa=True,
        )
        return output.transpose(1, 2)

    N = 32
    L = 128
    S = 128
    QH = 16
    H = 4
    D = 128
    BLOCK_L = 16
    BLOCK_S = 16
    Q = torch.randn((N, L, QH, D), dtype=torch.float16, device="cuda")
    K = torch.randn((N, S, H, D), dtype=torch.float16, device="cuda")
    V = torch.randn((N, S, H, D), dtype=torch.float16, device="cuda")
    match = kernel_tester(
        grouped_attention_kernel,
        ref_grouped_attention,
        {
            "N": N,
            "QH": QH,
            "H": H,
            "S": S,
            "D": D,
            "is_causal": is_causal,
            "BLOCK_L": BLOCK_L,
            "BLOCK_S": BLOCK_S,
        },
        inputs_in_torch_tensors=[Q, K, V],
        atol=1e-2,
        rtol=1e-2,
    )
    assert match, "Grouped attention test failed!"
    print("Grouped attention test passed!")
