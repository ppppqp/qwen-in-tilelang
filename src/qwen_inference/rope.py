import tilelang
import tilelang.language as T
import torch
from qwen_inference.utils import run_kernel


@tilelang.jit
def rope_kernel(X, offset, base, BLOCK_N, BLOCK_S, BLOCK_H, BLOCK_D):
    N, S, H, D = T.const("N, S, H, D")
    dtype = T.float32
    X: T.Tensor((N, S, H, D), dtype)
    O = T.empty((N, S, H, D), dtype)
    # can I compute constants here?
    half_D = D // 2

    num_h_blocks = T.ceildiv(H, BLOCK_H)
    num_d_blocks = T.ceildiv(half_D, BLOCK_D)
    with T.Kernel(
        T.ceildiv(N, BLOCK_N),
        T.ceildiv(S, BLOCK_S),
        num_h_blocks * num_d_blocks,
        # tilelang support 3 dimension at most, since hardware usually have at most 3 level of parallelism. So we need to combine H and D dimensions together.
        threads=256,
    ) as (
        pid_n,
        pid_s,
        pid_hd,
    ):
        pid_h = pid_hd // num_d_blocks
        pid_d = pid_hd % num_d_blocks
        X_local_real = T.alloc_fragment((BLOCK_N, BLOCK_S, BLOCK_H, BLOCK_D), dtype)
        X_local_imag = T.alloc_fragment((BLOCK_N, BLOCK_S, BLOCK_H, BLOCK_D), dtype)
        O_local_real = T.alloc_fragment((BLOCK_N, BLOCK_S, BLOCK_H, BLOCK_D), dtype)
        O_local_imag = T.alloc_fragment((BLOCK_N, BLOCK_S, BLOCK_H, BLOCK_D), dtype)
        cos_basis = T.alloc_fragment((BLOCK_S, BLOCK_D), dtype)
        sin_basis = T.alloc_fragment((BLOCK_S, BLOCK_D), dtype)
        for s, d in T.Parallel(BLOCK_S, BLOCK_D):
            seq_idx = offset + pid_s * BLOCK_S + s
            dim_idx = pid_d * BLOCK_D + d
            freq = T.pow(base, -dim_idx.astype("float32") / half_D)
            cos_basis[s, d] = T.cos(seq_idx * freq)
            sin_basis[s, d] = T.sin(seq_idx * freq)
        # for each block, we process
        # x[..., BLOCK_D * pid_d : BLOCK_D * (pid_d + 1)]
        # and x[..., BLOCK_D * pid_d + half_D : BLOCK_D * (pid_d + 1) + half_D]
        n_blk_id = pid_n
        s_blk_id = pid_s
        T.copy(
            X[
                n_blk_id * BLOCK_N : (n_blk_id + 1) * BLOCK_N,
                s_blk_id * BLOCK_S : (s_blk_id + 1) * BLOCK_S,
                pid_h * BLOCK_H : (pid_h + 1) * BLOCK_H,
                pid_d * BLOCK_D : (pid_d + 1) * BLOCK_D,
            ],
            X_local_real,
        )
        T.copy(
            X[
                n_blk_id * BLOCK_N : (n_blk_id + 1) * BLOCK_N,
                s_blk_id * BLOCK_S : (s_blk_id + 1) * BLOCK_S,
                pid_h * BLOCK_H : (pid_h + 1) * BLOCK_H,
                pid_d * BLOCK_D + half_D : (pid_d + 1) * BLOCK_D + half_D,
            ],
            X_local_imag,
        )

        for n, s, h, d in T.Parallel(BLOCK_N, BLOCK_S, BLOCK_H, BLOCK_D):
            real = (
                X_local_real[n, s, h, d] * cos_basis[s, d]
                - X_local_imag[n, s, h, d] * sin_basis[s, d]
            )
            imag = (
                X_local_imag[n, s, h, d] * cos_basis[s, d]
                + X_local_real[n, s, h, d] * sin_basis[s, d]
            )
            O_local_real[n, s, h, d] = real
            O_local_imag[n, s, h, d] = imag

        T.copy(
            O_local_real,
            O[
                n_blk_id * BLOCK_N : (n_blk_id + 1) * BLOCK_N,
                s_blk_id * BLOCK_S : (s_blk_id + 1) * BLOCK_S,
                pid_h * BLOCK_H : (pid_h + 1) * BLOCK_H,
                pid_d * BLOCK_D : (pid_d + 1) * BLOCK_D,
            ],
        )
        T.copy(
            O_local_imag,
            O[
                n_blk_id * BLOCK_N : (n_blk_id + 1) * BLOCK_N,
                s_blk_id * BLOCK_S : (s_blk_id + 1) * BLOCK_S,
                pid_h * BLOCK_H : (pid_h + 1) * BLOCK_H,
                pid_d * BLOCK_D + half_D : (pid_d + 1) * BLOCK_D + half_D,
            ],
        )
    return O


def rope(
    x: torch.Tensor,
    *,
    offset: int = 0,
    base: int = 10000,
    BLOCK_N: int = 1,
    BLOCK_S: int = 16,
    BLOCK_H: int = 1,
    BLOCK_D: int | None = None,
) -> torch.Tensor:
    assert x.ndim == 4
    assert x.shape[3] % 2 == 0
    x_dtype = x.dtype
    BLOCK_D = min(16, x.shape[3] // 2) if BLOCK_D is None else BLOCK_D
    out = run_kernel(
        rope_kernel,
        inputs=[x.to(torch.float32)],
        tl_hyper_params={
            "N": x.shape[0],
            "S": x.shape[1],
            "H": x.shape[2],
            "D": x.shape[3],
            "base": base,
            "offset": offset,
            "BLOCK_N": BLOCK_N,
            "BLOCK_S": BLOCK_S,
            "BLOCK_H": BLOCK_H,
            "BLOCK_D": BLOCK_D,
        },
    )
    return out.to(x_dtype)


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(self, x: torch.Tensor, _offset: int = 0) -> torch.Tensor:
        return rope(
            x,
            offset=_offset,
            base=self.base,
            BLOCK_N=1,
            BLOCK_S=16,
            BLOCK_H=1,
            BLOCK_D=min(16, self.dims // 2),
        )
