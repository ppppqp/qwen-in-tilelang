import tilelang
import tilelang.language as T
from tilelang.jit import JITKernel, JITImpl


@tilelang.jit
def rope(X, offset, BLOCK_N, BLOCK_S, BLOCK_H, BLOCK_D):
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
            freq = T.pow(10000.0, -dim_idx.astype("float32") / half_D)
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


# class RoPE:
#     def __init__(
#         self,
#         dims: int,
#         seq_len: int,
#         base: int = 10000,
#         traditional: bool = False,
#     ):
#         assert dims % 2 == 0, "dims must be even"
#         self.dims = dims
#         self.seq_len = seq_len
#         half_dims = dims // 2
#         inner = mx.arange(0, half_dims, dtype=mx.float32) / half_dims
#         freqs = mx.power(base, -inner)
#         t = mx.arange(seq_len)
#         freqs = mx.outer(t, freqs)
#         self.cos_freqs = mx.cos(freqs)
#         self.sin_freqs = mx.sin(freqs)
#         self.base = base
#         self.half_dims = half_dims
#         self.traditional = traditional

#     def __call__(
#         self, x: mx.array, offset: list[slice] | slice | None = None
#     ) -> mx.array:
#         N, S, H, D = x.shape
#         if offset is not None:
#             if isinstance(offset, slice):
#                 assert offset.stop - offset.start == S, f"offset must be of length {S}"
#             elif isinstance(offset, list):
#                 assert len(offset) == N, (
#                     f"offsets must have the same length as batch size {N}"
#                 )
#                 for o in offset:
#                     assert o.stop - o.start == S, f"offset must be of length {S}"
#                 offset = mx.array([list(range(i.start, i.stop)) for i in offset])
#         cos_basis = (
#             self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :]
#         )
#         sin_basis = (
#             self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :]
#         )
#         # reshape x: (b, s, n_heads, head_dim // 2, 2)
#         if self.traditional:
#             x = x.reshape(N, S, H, self.half_dims, 2)
#             x1 = x[..., 0]
#             x2 = x[..., 1]
#         else:
#             x1 = x[..., 0 : self.half_dims]
#             x2 = x[..., self.half_dims : self.dims]
#         # reshape basis: (1, s, 1, dims // 2, 2)
#         cos_basis = cos_basis.reshape(-1, S, 1, self.half_dims)
#         sin_basis = sin_basis.reshape(-1, S, 1, self.half_dims)
#         # manually doing complex number multiplication..
#         real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
#         imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)
#         if self.traditional:
#             y = mx.stack([real, imag], axis=-1)
#             y = y.reshape(N, S, H, D)
#         else:
#             y = mx.concat([real, imag], axis=-1)
#             y = y.reshape(N, S, H, D)
#         return y.astype(x.dtype)
