from __future__ import annotations

import torch

from qwen_inference.kernels.rope import rope_kernel
from tests.common_test_utils import kernel_tester


def _ref_rope(X: torch.Tensor, offset: int = 0, base: float = 10000.0):
    assert len(X.shape) == 4
    assert X.dtype == torch.float32

    _, S, _, D = X.shape
    assert D % 2 == 0

    half_d = D // 2
    positions = torch.arange(offset, offset + S, dtype=torch.float32, device=X.device)
    dims = torch.arange(half_d, dtype=torch.float32, device=X.device)
    freqs = torch.pow(base, -dims / half_d)
    angles = torch.outer(positions, freqs)
    cos_basis = torch.cos(angles).reshape(1, S, 1, half_d)
    sin_basis = torch.sin(angles).reshape(1, S, 1, half_d)

    x_real = X[..., :half_d]
    x_imag = X[..., half_d:]
    real = x_real * cos_basis - x_imag * sin_basis
    imag = x_imag * cos_basis + x_real * sin_basis
    return torch.cat([real, imag], dim=-1)


def test_rope_no_offset():
    N = 2
    S = 16
    H = 4
    D = 8
    BLOCK_N = 1
    BLOCK_S = 16
    BLOCK_H = 1
    BLOCK_D = 4
    offset = 0

    def ref_rope(X: torch.Tensor):
        return _ref_rope(X, offset=offset)

    match = kernel_tester(
        rope_kernel,
        ref_rope,
        {
            "N": N,
            "S": S,
            "H": H,
            "D": D,
            "offset": offset,
            "base": 10000.0,
            "BLOCK_N": BLOCK_N,
            "BLOCK_S": BLOCK_S,
            "BLOCK_H": BLOCK_H,
            "BLOCK_D": BLOCK_D,
        },
    )
    assert match, "RoPE no-offset test failed!"
    print("RoPE no-offset test passed!")


def test_rope_with_offset():
    N = 2
    S = 16
    H = 4
    D = 8
    BLOCK_N = 1
    BLOCK_S = 16
    BLOCK_H = 1
    BLOCK_D = 4
    offset = 32

    def ref_rope(X: torch.Tensor):
        return _ref_rope(X, offset=offset)

    match = kernel_tester(
        rope_kernel,
        ref_rope,
        {
            "N": N,
            "S": S,
            "H": H,
            "D": D,
            "offset": offset,
            "base": 10000.0,
            "BLOCK_N": BLOCK_N,
            "BLOCK_S": BLOCK_S,
            "BLOCK_H": BLOCK_H,
            "BLOCK_D": BLOCK_D,
        },
    )
    assert match, "RoPE offset test failed!"
    print("RoPE offset test passed!")
