from __future__ import annotations

from contextvars import ContextVar, Token
from enum import StrEnum
from typing import Any, NoReturn

import torch


class KernelBackend(StrEnum):
    DEFAULT = "default"
    KERNELS = "kernels"
    REF_KERNELS = "ref_kernels"

_KERNEL_BACKEND: ContextVar[KernelBackend] = ContextVar(
    "qwen_inference_kernel_backend", default=KernelBackend.DEFAULT
)


def set_kernel_backend(name: str | KernelBackend) -> Token[KernelBackend]:
    try:
        backend = KernelBackend(name)
    except ValueError as exc:
        choices = ", ".join(backend.value for backend in KernelBackend)
        raise ValueError(
            f"Unknown kernel backend {name!r}. Expected one of: {choices}"
        ) from exc
    return _KERNEL_BACKEND.set(backend)


def reset_kernel_backend(token: Token[KernelBackend]) -> None:
    _KERNEL_BACKEND.reset(token)


def get_kernel_backend() -> KernelBackend:
    return _KERNEL_BACKEND.get()


def _unknown_backend() -> NoReturn:
    raise RuntimeError(f"Unknown kernel backend {get_kernel_backend()!r}")


def linear(*args: Any, **kwargs: Any) -> torch.Tensor:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            from qwen_inference.kernels.linear import linear as kernel
        case KernelBackend.KERNELS:
            from kernels.linear import linear as kernel
        case KernelBackend.REF_KERNELS:
            from ref_kernels.linear import linear as kernel
        case _:
            _unknown_backend()
    return kernel(*args, **kwargs)


def rms_norm(*args: Any, **kwargs: Any) -> torch.Tensor:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            from qwen_inference.kernels.rms_norm import rms_norm as kernel
        case KernelBackend.KERNELS:
            from kernels.rms_norm import rms_norm as kernel
        case KernelBackend.REF_KERNELS:
            from ref_kernels.rms_norm import rms_norm as kernel
        case _:
            _unknown_backend()
    return kernel(*args, **kwargs)


def silu(*args: Any, **kwargs: Any) -> torch.Tensor:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            from qwen_inference.kernels.silu import silu as kernel
        case KernelBackend.KERNELS:
            from kernels.silu import silu as kernel
        case KernelBackend.REF_KERNELS:
            from ref_kernels.silu import silu as kernel
        case _:
            _unknown_backend()
    return kernel(*args, **kwargs)


def rope(*args: Any, **kwargs: Any) -> torch.Tensor:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            from qwen_inference.kernels.rope import rope as kernel
        case KernelBackend.KERNELS:
            from kernels.rope import rope as kernel
        case KernelBackend.REF_KERNELS:
            from ref_kernels.rope import rope as kernel
        case _:
            _unknown_backend()
    return kernel(*args, **kwargs)


def grouped_attention(*args: Any, **kwargs: Any) -> torch.Tensor:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            from qwen_inference.kernels.grouped_attention import (
                grouped_attention as kernel,
            )
        case KernelBackend.KERNELS:
            from kernels.grouped_attention import grouped_attention as kernel
        case KernelBackend.REF_KERNELS:
            from ref_kernels.grouped_attention import grouped_attention as kernel
        case _:
            _unknown_backend()
    return kernel(*args, **kwargs)


class RMSNorm:
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        out = rms_norm(x_flat, self.weight, self.eps, BLOCK_M=16, BLOCK_N=self.dim)
        return out.reshape(orig_shape)


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
            THREADS=256,
        )
