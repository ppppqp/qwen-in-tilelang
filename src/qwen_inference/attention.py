import torch

from qwen_inference.basics import linear

import tilelang
import tilelang.language as T


# TODO: mask
# TODO: multihead
# TODO: group QA
@tilelang.jit
def attention(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    with T.Kernel(B // BLOCK_B, threads=256) as pid_b:
        Q_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        K_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        V_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        O_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)

        cur_QK = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
        cur_exp_QK = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
        cur_max_QK = T.alloc_fragment([BLOCK_B], dtype)
        cur_sum_exp_QK = T.alloc_fragment([BLOCK_B], dtype)

        lse = T.alloc_fragment([BLOCK_B], dtype)

        T.fill(lse, -T.infinity(dtype))

        # The first loop use an online algorithm to compute LSE.
        for s_blk_id in T.Serial(S // BLOCK_S):
            T.copy(Q[pid_b * BLOCK_B, s_blk_id * BLOCK_S], Q_local)
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                cur_QK[i, j] = Q_local[i, j] * K_local[i, j]

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
        for s_blk_id in T.Serial(S // BLOCK_S):
            T.copy(Q[pid_b * BLOCK_B, s_blk_id * BLOCK_S], Q_local)
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)
            T.copy(V[pid_b * BLOCK_B, s_blk_id * BLOCK_S], V_local)

            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                O_local[i, j] = (
                    T.exp2(Q_local[i, j] * K_local[i, j] * log2_e - lse[i])
                    * V_local[i, j]
                )

            T.copy(O_local, O[pid_b * BLOCK_B, s_blk_id * BLOCK_S])

    return O


# Grouped QA, similar to multi-head attention but with shared Q and separate K, V.
# TODO: implement custom scale and mask
# TODO: case where query seq length != key/value seq length
# TODO: optimize for sparse case?
@tilelang.jit
def grouped_attention(Q, K, V, mask, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    QB, B, S = T.const("QB, B, S")
    dtype = T.float32
    # the only diff is that shape of Q is QB, S instead of B, S
    Q: T.Tensor((QB, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    mask: T.Tensor((B, S), dtype)
    O = T.empty((QB, S), dtype)

    head_num = QB // B
    with T.Kernel(B // BLOCK_B, threads=256) as pid_b:
        Q_local = T.alloc_fragment((head_num, BLOCK_B, BLOCK_S), dtype)
        K_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        V_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        O_local = T.alloc_fragment((head_num, BLOCK_B, BLOCK_S), dtype)

        cur_QK = T.alloc_fragment([head_num, BLOCK_B, BLOCK_S], dtype)
        cur_exp_QK = T.alloc_fragment([head_num, BLOCK_B, BLOCK_S], dtype)
        cur_max_QK = T.alloc_fragment([head_num, BLOCK_B], dtype)
        cur_sum_exp_QK = T.alloc_fragment([head_num, BLOCK_B], dtype)

        lse = T.alloc_fragment([head_num, BLOCK_B], dtype)

        T.fill(lse, -T.infinity(dtype))

        # The first loop use an online algorithm to compute LSE.
        for s_blk_id in T.Serial(S // BLOCK_S):
            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                Q_local[h, i, j] = Q[
                    h * B + pid_b * BLOCK_B + i, s_blk_id * BLOCK_S + j
                ]
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)

            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                # mask out the scores here
                # mask value is expected to be 0 for valid positions and -inf for masked positions, so we can directly add it to the score.
                cur_QK[h, i, j] = (
                    Q_local[h, i, j] * K_local[i, j]
                    + mask[pid_b * BLOCK_B + i, s_blk_id * BLOCK_S + j]
                )

            T.reduce_max(cur_QK, cur_max_QK, dim=2, clear=True)

            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                cur_exp_QK[h, i, j] = T.if_then_else(
                    cur_max_QK[h, i]
                    == -T.infinity(dtype),  # to avoid -inf - (-inf) = nan case
                    0,
                    T.exp2(cur_QK[h, i, j] * log2_e - cur_max_QK[h, i] * log2_e),
                )

            T.reduce_sum(cur_exp_QK, cur_sum_exp_QK, dim=2, clear=True)

            for h, i in T.Parallel(head_num, BLOCK_B):
                lse[h, i] = T.if_then_else(
                    cur_sum_exp_QK[h, i]
                    == 0,  # if all positions are masked, then no op on this row
                    lse[h, i],
                    cur_max_QK[h, i] * log2_e
                    + T.log2(
                        T.exp2(lse[h, i] - cur_max_QK[h, i] * log2_e)
                        + cur_sum_exp_QK[h, i]
                    ),
                )

        # The second loop use LSE to get the final output.
        # TODO: improve the efficiency here. Maybe pipeline it?
        for s_blk_id in T.Serial(S // BLOCK_S):
            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                Q_local[h, i, j] = Q[
                    h * B + pid_b * BLOCK_B + i, s_blk_id * BLOCK_S + j
                ]
            T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)
            T.copy(V[pid_b * BLOCK_B, s_blk_id * BLOCK_S], V_local)

            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                O_local[h, i, j] = (
                    T.exp2(
                        (
                            Q_local[h, i, j] * K_local[i, j]
                            + mask[pid_b * BLOCK_B + i, s_blk_id * BLOCK_S + j]
                        )
                        * log2_e
                        - lse[h, i]
                    )
                    * V_local[i, j]
                )
            for h, i, j in T.Parallel(head_num, BLOCK_B, BLOCK_S):
                O[h * B + pid_b * BLOCK_B + i, s_blk_id * BLOCK_S + j] = O_local[
                    h, i, j
                ]

    return O


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """
    Potential input of the mask:
    - torch.Tensor that can broadcast to B * H_q * L * S, which needs to be reshaped to match multi-head dimensions
    - None which will be ignored
    """
    factor = query.shape[-1] ** -0.5 if scale is None else scale
    if not torch.is_tensor(factor):
        factor = torch.tensor(factor, dtype=query.dtype, device=query.device)
    else:
        factor = factor.to(dtype=query.dtype, device=query.device)
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    B = query.shape[:-3]
    assert H_q % H == 0
    n_repeats = H_q // H

    query = query.reshape(*B, -1, H, n_repeats, L, D)
    key = key.reshape(*B, -1, H, 1, S, D)
    value = value.reshape(*B, -1, H, 1, S, D)

    scores = torch.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            mask = causal_mask(L, S, scores.dtype, scores.device)
            scores = scores + mask
        elif isinstance(mask, str):
            raise ValueError(f"Unsupported attention mask: {mask}")
        else:
            mask = torch.broadcast_to(mask, (*B, H_q, L, S))
            mask = mask.reshape(*B, 1, H, n_repeats, L, S)
            scores = scores + mask
    result = torch.matmul(torch.softmax(scores, dim=-1), value)
    return result.reshape(expected_shape)


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
    return scaled_dot_product_attention_grouped(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        scale=scale,
        mask=mask,
    )


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        assert wq.shape == (num_heads * self.head_dim, hidden_size)
        assert wk.shape == (num_heads * self.head_dim, hidden_size)
        assert wv.shape == (num_heads * self.head_dim, hidden_size)
        assert wo.shape == (hidden_size, num_heads * self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        x = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )
        x = x.permute(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(x, self.wo)


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
