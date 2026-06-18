import torch
import tilelang
import tilelang.language as T

from qwen_inference.utils import run_kernel


@tilelang.jit
def softmax_kernel(A, BLOCK_N: int, BLOCK_M: int, THREADS: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=THREADS) as bx:
        A_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        cur_max = T.alloc_fragment((BLOCK_N,), dtype)
        cur_exp = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        cur_sum = T.alloc_fragment((BLOCK_N,), dtype)

        lse = T.alloc_fragment((BLOCK_N), dtype)
        T.fill(lse, -T.infinity(dtype))

        for m_idx in T.Serial(T.ceildiv(M, BLOCK_M)):
            T.copy(A[bx * BLOCK_N, m_idx * BLOCK_M], A_local)
            T.reduce_max(A_local, cur_max, dim=1, clear=True)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                cur_exp[i, j] = T.exp2(A_local[i, j] * log2_e - cur_max[i] * log2_e)

            T.reduce_sum(cur_exp, cur_sum, dim=1, clear=True)
            for nn in T.Parallel(BLOCK_N):
                lse[nn] = (
                    T.log2(T.exp2(lse[nn] - cur_max[nn] * log2_e) + cur_sum[nn])
                    + cur_max[nn] * log2_e
                )

        for m_idx in T.Serial(T.ceildiv(M, BLOCK_M)):
            T.copy(A[bx * BLOCK_N, m_idx * BLOCK_M], A_local)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                B_local[i, j] = T.exp2(A_local[i, j] * log2_e - lse[i])

            T.copy(B_local, B[bx * BLOCK_N, m_idx * BLOCK_M])

    return B


def softmax(
    A: torch.Tensor,
    *,
    BLOCK_N: int = 16,
    BLOCK_M: int | None = None,
    THREADS: int = 256,
) -> torch.Tensor:
    assert A.ndim == 2
    BLOCK_M = A.shape[1] if BLOCK_M is None else BLOCK_M
    return run_kernel(
        softmax_kernel,
        inputs=[A],
        tl_hyper_params={
            "N": A.shape[0],
            "M": A.shape[1],
            "BLOCK_N": BLOCK_N,
            "BLOCK_M": BLOCK_M,
            "THREADS": THREADS,
        },
    )
