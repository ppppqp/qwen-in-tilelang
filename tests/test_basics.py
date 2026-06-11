from __future__ import annotations

import torch

from qwen_inference.basics import linear, rms_norm, softmax
from tests.common_test_utils import kernel_tester


def test_softmax():
    def ref_softmax(A: torch.Tensor):
        assert len(A.shape) == 2
        assert A.dtype == torch.float32
        return torch.softmax(A, dim=1)

    N = 256
    M = 1024
    BLOCK_N = 16
    BLOCK_M = 128
    match = kernel_tester(
        softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    assert match, "Softmax test failed!"
    print("Softmax test passed!")


def test_linear():
    def ref_linear(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        assert len(X.shape) == 2
        assert len(W.shape) == 2
        assert len(b.shape) == 2
        assert X.shape[1] == W.shape[0]
        assert b.shape == (X.shape[0], W.shape[1])
        assert X.dtype == W.dtype == b.dtype == torch.float16
        return X @ W + b

    M = 4096
    N = 4096
    K = 4096
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    match = kernel_tester(
        linear,
        ref_linear,
        {
            "M": M,
            "N": N,
            "K": K,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
        },
        atol=1e-1,
        rtol=1e-1,
    )
    assert match, "Linear test failed!"
    print("Linear test passed!")


def test_rms_norm():
    def ref_rms_norm(X: torch.Tensor, weight: torch.Tensor):
        assert len(X.shape) == 2
        assert len(weight.shape) == 1
        assert X.shape[1] == weight.shape[0]
        assert X.dtype == weight.dtype == torch.float16
        X_square = X * X
        mean = X_square.mean(dim=1, keepdim=True)
        return (X / torch.sqrt(mean + eps) * weight).to(torch.float16)

    M = 128
    N = 128
    BLOCK_M = 16
    BLOCK_N = 128
    eps = 1e-5
    match = kernel_tester(
        rms_norm,
        ref_rms_norm,
        {
            "M": M,
            "N": N,
            "eps": eps,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
        },
        atol=1e-2,
        rtol=1e-2,
    )
    assert match, "RMS norm test failed!"
    print("RMS norm test passed!")
