import torch
import tilelang
import tilelang.language as T

from qwen_inference.utils import run_kernel


@tilelang.jit
def rms_norm_kernel(
    X, weight, eps: float, BLOCK_M: int, BLOCK_N: int, THREADS: int
):
    M, N = T.const("M, N")

    dtype = T.float16
    accum_dtype = T.float32

    X: T.Tensor((M, N), dtype)
    weight: T.Tensor((N,), dtype)
    O = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=THREADS) as pid_m:
        X_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
        W_local = T.alloc_fragment((BLOCK_N,), dtype)
        X_square = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        sum_square = T.alloc_fragment((BLOCK_M,), accum_dtype)
        inv_rms = T.alloc_fragment((BLOCK_M,), accum_dtype)
        O_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)

        T.copy(X[pid_m * BLOCK_M, 0], X_local)
        T.copy(weight[0], W_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            x = X_local[i, j].astype(accum_dtype)
            X_square[i, j] = x * x

        T.reduce_sum(X_square, sum_square, dim=1, clear=True)

        for i in T.Parallel(BLOCK_M):
            inv_rms[i] = T.rsqrt(sum_square[i] / N + eps)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            O_local[i, j] = (
                X_local[i, j].astype(accum_dtype)
                * inv_rms[i]
                * W_local[j].astype(accum_dtype)
            ).astype(dtype)

        T.copy(O_local, O[pid_m * BLOCK_M, 0])

    return O


def rms_norm(
    X: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    BLOCK_M: int = 16,
    BLOCK_N: int | None = None,
    THREADS: int = 128,
) -> torch.Tensor:
    assert X.ndim == 2
    assert weight.ndim == 1
    assert X.shape[1] == weight.shape[0]
    BLOCK_N = X.shape[1] if BLOCK_N is None else BLOCK_N
    return run_kernel(
        rms_norm_kernel,
        inputs=[X, weight],
        tl_hyper_params={
            "M": X.shape[0],
            "N": X.shape[1],
            "eps": eps,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "THREADS": THREADS,
        },
    )


class RMSNorm:
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        out = rms_norm(x_flat, self.weight, self.eps, BLOCK_M=16, BLOCK_N=self.dim)
        return out.reshape(orig_shape)
