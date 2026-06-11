import tilelang
import tilelang.language as T
from tilelang.jit import JITKernel, JITImpl
from tilelang.engine.param import KernelParam

# TODO: handle more than 2D tensors


@tilelang.jit
def softmax(A, BLOCK_N: int, BLOCK_M: int):
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
def linear(X, W, b, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
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


def silu(X):
    N, M = T.const("M, N")
    pass


@tilelang.jit
def rms_norm(X, weight, eps: float, BLOCK_M: int, BLOCK_N: int):
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
