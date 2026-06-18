import torch
import tilelang
import tilelang.language as T

from qwen_inference.utils import run_kernel


@tilelang.jit
def grouped_attention_kernel(
    Q, K, V, is_causal: bool, BLOCK_L: int, BLOCK_S: int, THREADS: int
):
    N, QH, H, S, D = T.const("N, QH, H, S, D")
    dtype = T.float16
    accum_dtype = T.float32
    L = S
    Q: T.Tensor((N, L, QH, D), dtype)
    K: T.Tensor((N, S, H, D), dtype)
    V: T.Tensor((N, S, H, D), dtype)
    O = T.empty((N, L, QH, D), dtype)
    scale = 1 / T.sqrt(T.cast(D, accum_dtype))
    log2_e = 1.44269504

    group_size = QH // H
    with T.Kernel(N, T.ceildiv(L, BLOCK_L), QH, threads=THREADS) as (
        pid_n,
        pid_l,
        pid_h,
    ):
        Q_shared = T.alloc_shared((BLOCK_L, D), dtype)
        K_shared = T.alloc_shared((BLOCK_S, D), dtype)
        V_shared = T.alloc_shared((BLOCK_S, D), dtype)
        O_shared = T.alloc_shared((BLOCK_L, D), dtype)

        current_max = T.alloc_fragment([BLOCK_L], accum_dtype)
        previous_max = T.alloc_fragment([BLOCK_L], accum_dtype)
        running_sum = T.alloc_fragment([BLOCK_L], accum_dtype)
        alpha = T.alloc_fragment([BLOCK_L], accum_dtype)
        acc_kv = T.alloc_fragment([BLOCK_L, BLOCK_S], accum_dtype)
        acc_kv_cast = T.alloc_fragment([BLOCK_L, BLOCK_S], dtype)
        acc_out = T.alloc_fragment([BLOCK_L, D], accum_dtype)

        T.fill(current_max, -T.infinity(accum_dtype))
        T.fill(running_sum, 0.0)
        T.fill(acc_out, 0.0)

        T.copy(Q[pid_n, pid_l * BLOCK_L : (pid_l + 1) * BLOCK_L, pid_h, :], Q_shared)

        loop_range = (
            T.min(T.ceildiv(S, BLOCK_S), T.ceildiv((pid_l + 1) * BLOCK_L, BLOCK_S))
            if is_causal
            else T.ceildiv(S, BLOCK_S)
        )

        for k in T.Pipelined(loop_range, num_stages=2):
            T.copy(
                K[pid_n, k * BLOCK_S : (k + 1) * BLOCK_S, pid_h // group_size, :],
                K_shared,
            )

            if is_causal:
                for i, j in T.Parallel(BLOCK_L, BLOCK_S):
                    acc_kv[i, j] = T.if_then_else(
                        pid_l * BLOCK_L + i >= k * BLOCK_S + j,
                        0,
                        -T.infinity(accum_dtype),
                    )
            else:
                for i, j in T.Parallel(BLOCK_L, BLOCK_S):
                    acc_kv[i, j] = T.if_then_else(
                        k * BLOCK_S + j >= S, -T.infinity(accum_dtype), 0
                    )

            T.gemm(
                Q_shared,
                K_shared,
                acc_kv,
                transpose_B=True,
                clear_accum=False,
                policy=T.GemmWarpPolicy.FullRow,
            )

            T.copy(current_max, previous_max)
            T.fill(current_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_kv, current_max, dim=1, clear=False)
            for i in T.Parallel(BLOCK_L):
                current_max[i] = T.max(current_max[i], previous_max[i])

            for i, j in T.Parallel(BLOCK_L, BLOCK_S):
                acc_kv[i, j] = T.exp2((acc_kv[i, j] - current_max[i]) * log2_e * scale)

            for i in T.Parallel(BLOCK_L):
                alpha[i] = T.exp2((previous_max[i] - current_max[i]) * log2_e * scale)
                running_sum[i] = running_sum[i] * alpha[i]

            T.reduce_sum(acc_kv, running_sum, dim=1, clear=False)

            for i, j in T.Parallel(BLOCK_L, D):
                acc_out[i, j] *= alpha[i]
            T.copy(acc_kv, acc_kv_cast)

            T.copy(
                V[pid_n, k * BLOCK_S : (k + 1) * BLOCK_S, pid_h // group_size, :],
                V_shared,
            )
            T.gemm(
                acc_kv_cast,
                V_shared,
                acc_out,
                clear_accum=False,
                policy=T.GemmWarpPolicy.FullRow,
            )

        for i, j in T.Parallel(BLOCK_L, D):
            acc_out[i, j] /= running_sum[i]

        T.copy(acc_out, O_shared)
        T.copy(O_shared, O[pid_n, pid_l * BLOCK_L : (pid_l + 1) * BLOCK_L, pid_h, :])

    return O


def grouped_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    is_causal: bool = True,
    BLOCK_L: int = 32,
    BLOCK_S: int = 32,
    THREADS: int = 32,
) -> torch.Tensor:
    assert Q.ndim == K.ndim == V.ndim == 4
    assert K.shape == V.shape
    N, L, QH, D = Q.shape
    _, S, H, _ = K.shape
    assert L == S
    assert QH % H == 0
    return run_kernel(
        grouped_attention_kernel,
        inputs=[Q, K, V],
        tl_hyper_params={
            "N": N,
            "QH": QH,
            "H": H,
            "S": S,
            "D": D,
            "is_causal": is_causal,
            "BLOCK_L": BLOCK_L,
            "BLOCK_S": BLOCK_S,
            "THREADS": THREADS,
        },
    )
