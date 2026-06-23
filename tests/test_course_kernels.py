import pytest
import torch

import kernels.grouped_attention as learner_grouped_attention
import kernels.linear as learner_linear
import kernels.rms_norm as learner_rms_norm
import kernels.rope as learner_rope
import kernels.silu as learner_silu
import ref_kernels.grouped_attention as ref_grouped_attention
import ref_kernels.linear as ref_linear
import ref_kernels.rms_norm as ref_rms_norm
import ref_kernels.rope as ref_rope
import ref_kernels.silu as ref_silu


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() == 0,
    reason="course kernel checks require CUDA",
)


def test_course_silu_matches_reference():
    x = torch.randn((16, 64), dtype=torch.float16, device="cuda")

    actual = learner_silu.silu(x, BLOCK_M=16, BLOCK_N=64)
    expected = ref_silu.silu(x, BLOCK_M=16, BLOCK_N=64)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_course_rms_norm_matches_reference():
    x = torch.randn((16, 64), dtype=torch.float16, device="cuda")
    weight = torch.randn((64,), dtype=torch.float16, device="cuda")

    actual = learner_rms_norm.rms_norm(x, weight, 1e-5, BLOCK_M=16, BLOCK_N=64)
    expected = ref_rms_norm.rms_norm(x, weight, 1e-5, BLOCK_M=16, BLOCK_N=64)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_course_linear_matches_reference():
    x = torch.randn((16, 64), dtype=torch.float16, device="cuda")
    w = torch.randn((64, 32), dtype=torch.float16, device="cuda")

    actual = learner_linear.linear(x, w, BLOCK_M=16, BLOCK_N=32, BLOCK_K=64)
    expected = ref_linear.linear(x, w, BLOCK_M=16, BLOCK_N=32, BLOCK_K=64)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_course_rope_matches_reference():
    x = torch.randn((1, 16, 4, 32), dtype=torch.float16, device="cuda")

    actual = learner_rope.rope(x, BLOCK_N=1, BLOCK_S=16, BLOCK_H=1, BLOCK_D=16)
    expected = ref_rope.rope(x, BLOCK_N=1, BLOCK_S=16, BLOCK_H=1, BLOCK_D=16)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_course_grouped_attention_matches_reference():
    q = torch.randn((1, 16, 4, 32), dtype=torch.float16, device="cuda")
    k = torch.randn((1, 16, 2, 32), dtype=torch.float16, device="cuda")
    v = torch.randn((1, 16, 2, 32), dtype=torch.float16, device="cuda")

    actual = learner_grouped_attention.grouped_attention(
        q, k, v, BLOCK_L=16, BLOCK_S=16
    )
    expected = ref_grouped_attention.grouped_attention(
        q, k, v, BLOCK_L=16, BLOCK_S=16
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
