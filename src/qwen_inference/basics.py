import torch

from qwen_inference.utils import run_kernel
import tilelang
import tilelang.language as T

# TODO: handle more than 2D tensors


@tilelang.jit
def softmax_kernel(A, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as bx:
        A_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        cur_max = T.alloc_fragment((BLOCK_N,), dtype)
        cur_exp = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        cur_sum = T.alloc_fragment((BLOCK_N,), dtype)

        # log sum exponential
        lse = T.alloc_fragment((BLOCK_N), dtype)
        T.fill(lse, -T.infinity(dtype))

        for m_idx in T.Serial(T.ceildiv(M, BLOCK_M)):
            T.copy(A[bx * BLOCK_N, m_idx * BLOCK_M], A_local)
            T.reduce_max(A_local, cur_max, dim=1, clear=True)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                # minus current max only for numerical stability
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


@tilelang.jit
def linear_kernel(X, W, b, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((M, K), dtype)
    W: T.Tensor((K, N), dtype)
    b: T.Tensor((M, N), dtype)
    O = T.empty((M, N), dtype)
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (
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


# TODO: implement this
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


@tilelang.jit
def silu_kernel(X, BLOCK_M, BLOCK_N):
    M, N = T.const("M, N")
    dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((M, N), dtype)
    O = T.empty((M, N), dtype)
    log2_e = 1.44269504
    with T.Kernel(
        T.ceildiv(M, BLOCK_M),
        threads=128,
    ) as pid_n:
        X_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
        O_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)

        T.copy(X[pid_n * BLOCK_M, 0], X_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            x = X_local[i, j].astype(accum_dtype)
            O_local[i, j] = (x / (1 + T.exp2(-x * log2_e))).astype(dtype)

        T.copy(O_local, O[pid_n * BLOCK_M, 0])

    return O


@tilelang.jit
def rms_norm_kernel(X, weight, eps: float, BLOCK_M: int, BLOCK_N: int):
    M, N = T.const("M, N")

    dtype = T.float16
    accum_dtype = T.float32

    X: T.Tensor((M, N), dtype)
    weight: T.Tensor((N,), dtype)
    O = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=128) as pid_m:
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


def softmax(
    A: torch.Tensor,
    *,
    BLOCK_N: int = 16,
    BLOCK_M: int | None = None,
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
        },
    )


def linear(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
    *,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
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
        },
    )


def silu(
    X: torch.Tensor,
    *,
    BLOCK_M: int = 16,
    BLOCK_N: int | None = None,
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
        },
    )


def rms_norm(
    X: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    BLOCK_M: int = 16,
    BLOCK_N: int | None = None,
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
