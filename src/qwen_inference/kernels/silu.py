import torch
import tilelang
import tilelang.language as T

from qwen_inference.utils import run_kernel


@tilelang.jit
def silu_kernel(X, BLOCK_M, BLOCK_N, THREADS: int):
    M, N = T.const("M, N")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((M, N), dtype)
    O = T.empty((M, N), dtype)
    log2_e = 1.44269504
    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=THREADS) as pid_n:
        X_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
        O_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)

        T.copy(X[pid_n * BLOCK_M, 0], X_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            x = X_local[i, j].astype(accum_dtype)
            O_local[i, j] = (x / (1 + T.exp2(-x * log2_e))).astype(dtype)

        T.copy(O_local, O[pid_n * BLOCK_M, 0])

    return O


def silu(
    X: torch.Tensor,
    *,
    BLOCK_M: int = 16,
    BLOCK_N: int | None = None,
    THREADS: int = 128,
) -> torch.Tensor:
    assert X.ndim == 2
    BLOCK_N = X.shape[1] if BLOCK_N is None else BLOCK_N
    return run_kernel(
        silu_kernel,
        inputs=[X],
        tl_hyper_params={
            "M": X.shape[0],
            "N": X.shape[1],
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "THREADS": THREADS,
        },
    )
