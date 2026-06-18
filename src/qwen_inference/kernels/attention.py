import torch
import tilelang
import tilelang.language as T

from qwen_inference.utils import run_kernel


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
    raise NotImplementedError("Torch paged_attention needs a Torch/TileLang kernel.")


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    raise NotImplementedError("Torch paged_attention needs a Torch/TileLang kernel.")

