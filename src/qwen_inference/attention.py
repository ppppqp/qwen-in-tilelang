import torch

import tilelang
import tilelang.language as T
from qwen_inference.utils import run_kernel


# TODO: mask
# TODO: multihead
# TODO: group QA
@tilelang.jit
def attention_kernel(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    with T.Kernel(T.ceildiv(B, BLOCK_B), threads=256) as pid_b:
        Q_shared = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        K_shared = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        V_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        O_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)

        cur_QK = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
        cur_exp_QK = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
        cur_max_QK = T.alloc_fragment([BLOCK_B], dtype)
        cur_sum_exp_QK = T.alloc_fragment([BLOCK_B], dtype)

        lse = T.alloc_fragment([BLOCK_B], dtype)

        T.fill(lse, -T.infinity(dtype))

        # The first loop use an online algorithm to compute LSE.
        for s_blk_id in T.Serial(T.ceildiv(S, BLOCK_S)):
            T.copy(Q[pid_b * BLOCK_B, s_blk_id * BLOCK_S], Q_shared)
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_shared)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                cur_QK[i, j] = Q_shared[i, j] * K_shared[i, j]

            T.reduce_max(cur_QK, cur_max_QK, dim=1, clear=True)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                cur_exp_QK[i, j] = T.exp2(
                    cur_QK[i, j] * log2_e - cur_max_QK[i] * log2_e
                )

            T.reduce_sum(cur_exp_QK, cur_sum_exp_QK, dim=1, clear=True)

            for i in T.Parallel(BLOCK_B):
                lse[i] = cur_max_QK[i] * log2_e + T.log2(
                    T.exp2(lse[i] - cur_max_QK[i] * log2_e) + cur_sum_exp_QK[i]
                )

        # The second loop use LSE to get the final output.
        # TODO: improve the efficiency here. Maybe pipeline it?
        for s_blk_id in T.Serial(T.ceildiv(S, BLOCK_S)):
            T.copy(Q[pid_b * BLOCK_B, s_blk_id * BLOCK_S], Q_shared)
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_shared)
            T.copy(V[pid_b * BLOCK_B, s_blk_id * BLOCK_S], V_local)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                O_local[i, j] = (
                    T.exp2(Q_shared[i, j] * K_shared[i, j] * log2_e - lse[i])
                    * V_local[i, j]
                )

            T.copy(O_local, O[pid_b * BLOCK_B, s_blk_id * BLOCK_S])

    return O


# Grouped QA, similar to multi-head attention but with shared Q and separate K, V.
# TODO: implement custom scale and mask
# TODO: case where query seq length != key/value seq length
# TODO: optimize for sparse case?
# Normally the input tensors are (N, H, S, D)
# N is batch size, H is number of heads, S is sequence length, D is head dimension.
# However, in this kernel we use B to represent H * D
@tilelang.jit
def grouped_attention_kernel(Q, K, V, is_causal: bool, BLOCK_L: int, BLOCK_S: int):
    log2_e = 1.44269504
    N, QH, H, S, D = T.const("N, QH, H, S, D")
    dtype = T.float16
    accum_dtype = T.float32
    # Query: N, L, QH, D
    # Key, Value: N, S, H, D
    # Note that D is used as tiling directly (aka BLOCK_D)
    L = S
    Q: T.Tensor((N, L, QH, D), dtype)
    K: T.Tensor((N, S, H, D), dtype)
    V: T.Tensor((N, S, H, D), dtype)
    # mask: T.Tensor((S, S), dtype)
    O = T.empty((N, L, QH, D), dtype)
    scale = 1 / T.sqrt(T.cast(D, accum_dtype))
    log2_e = 1.44269504

    group_size = QH // H
    with T.Kernel(N, T.ceildiv(L, BLOCK_L), QH, threads=32) as (
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
                        pid_l * BLOCK_L + i >= k * BLOCK_S + j,  # row index > col index
                        0,
                        -T.infinity(accum_dtype),
                    )
            else:
                # have to have this block, even if tilelang guards against out of bound memory access
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

            # e.g. QH=16, H=4, then we have 4 groups and 16 threads, so pid_h // group_size means
            # 0, 1, 2, 3 thread compute the first group, 4, 5, 6, 7 thread compute the second group, etc.
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


def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    BLOCK_B: int = 16,
    BLOCK_S: int = 128,
) -> torch.Tensor:
    assert Q.ndim == K.ndim == V.ndim == 2
    assert Q.shape == K.shape == V.shape
    return run_kernel(
        attention_kernel,
        inputs=[Q, K, V],
        tl_hyper_params={
            "B": Q.shape[0],
            "S": Q.shape[1],
            "BLOCK_B": BLOCK_B,
            "BLOCK_S": BLOCK_S,
        },
    )


def grouped_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    is_causal: bool = True,
    BLOCK_L: int = 16,
    BLOCK_S: int = 16,
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
        },
    )


def paged_attention(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    page_size: int,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """
    Paged attention requires a backend-specific kernel. The previous
    implementation called an MLX C++/Metal extension, which is not available
    from Torch tensors.
    """
    raise NotImplementedError("Torch paged_attention needs a Torch/TileLang kernel.")


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    raise NotImplementedError("Torch paged_attention needs a Torch/TileLang kernel.")


# class SimpleMultiHeadAttention:
#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         wq: torch.Tensor,
#         wk: torch.Tensor,
#         wv: torch.Tensor,
#         wo: torch.Tensor,
#     ):
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         assert hidden_size % num_heads == 0
#         self.head_dim = hidden_size // num_heads
#         self.scale = self.head_dim**-0.5
#         assert wq.shape == (num_heads * self.head_dim, hidden_size)
#         assert wk.shape == (num_heads * self.head_dim, hidden_size)
#         assert wv.shape == (num_heads * self.head_dim, hidden_size)
#         assert wo.shape == (hidden_size, num_heads * self.head_dim)
#         self.wq = wq
#         self.wk = wk
#         self.wv = wv
#         self.wo = wo

#     def __call__(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         mask: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         N, L, _ = query.shape
#         assert query.shape == key.shape == value.shape
#         projection_q = (
#             linear(query, self.wq)
#             .reshape(N, L, self.num_heads, self.head_dim)
#             .permute(0, 2, 1, 3)
#         )
#         projection_k = (
#             linear(key, self.wk)
#             .reshape(N, L, self.num_heads, self.head_dim)
#             .permute(0, 2, 1, 3)
#         )
#         projection_v = (
#             linear(value, self.wv)
#             .reshape(N, L, self.num_heads, self.head_dim)
#             .permute(0, 2, 1, 3)
#         )
#         x = scaled_dot_product_attention_simple(
#             projection_q,
#             projection_k,
#             projection_v,
#             scale=self.scale,
#             mask=mask,
#         )
#         x = x.permute(0, 2, 1, 3).reshape(N, L, self.hidden_size)
#         return linear(x, self.wo)


def causal_mask(
    L: int,
    S: int,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    mask = torch.tril(
        torch.ones((L, S), dtype=torch.bool, device=device), diagonal=S - L
    )
    return torch.where(
        mask,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), -torch.inf, dtype=dtype, device=device),
    )
