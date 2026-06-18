import torch
import tilelang
import tilelang.language as T

from qwen_inference.utils import run_kernel


@tilelang.jit
def linear_kernel(
    X, W, b, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, THREADS: int
):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((M, K), dtype)
    W: T.Tensor((K, N), dtype)
    b: T.Tensor((M, N), dtype)
    O = T.empty((M, N), dtype)
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (
        pid_n,
        pid_m,
    ):
        acc = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        X_local = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W_local = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)

        T.copy(b[pid_m * BLOCK_M, pid_n * BLOCK_N], acc)

        for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
            T.copy(X[pid_m * BLOCK_M, k * BLOCK_K], X_local)
            T.copy(W[k * BLOCK_K, pid_n * BLOCK_N], W_local)

            T.gemm(X_local, W_local, acc, clear_accum=False)

        T.copy(acc, O[pid_m * BLOCK_M, pid_n * BLOCK_N])
    return O


@tilelang.jit
def quantized_linear_kernel(
    X, W, b, group_size, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int
):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((M, K), dtype)
    W: T.Tensor((K, N), dtype)
    pass


def linear(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
    *,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
    THREADS: int = 128,
) -> torch.Tensor:
    assert X.ndim == 2
    assert W.ndim == 2
    assert X.shape[1] == W.shape[0]
    if b is None:
        b = torch.zeros((X.shape[0], W.shape[1]), dtype=X.dtype, device=X.device)
    assert b.shape == (X.shape[0], W.shape[1])
    return run_kernel(
        linear_kernel,
        inputs=[X, W, b],
        tl_hyper_params={
            "M": X.shape[0],
            "N": W.shape[1],
            "K": X.shape[1],
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "THREADS": THREADS,
        },
    )
