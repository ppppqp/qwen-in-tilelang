from __future__ import annotations

from contextvars import ContextVar, Token
from enum import StrEnum
from typing import Any, NoReturn

import torch

from kernels.grouped_attention import grouped_attention as learner_grouped_attention
from kernels.linear import linear as learner_linear
from kernels.rms_norm import rms_norm as learner_rms_norm
from kernels.rope import rope as learner_rope
from kernels.silu import silu as learner_silu
from qwen_inference.kernels.grouped_attention import (
    grouped_attention as default_grouped_attention,
)
from qwen_inference.kernels.linear import linear as default_linear
from qwen_inference.kernels.rms_norm import rms_norm as default_rms_norm
from qwen_inference.kernels.rope import rope as default_rope
from qwen_inference.kernels.silu import silu as default_silu
from ref_kernels.grouped_attention import grouped_attention as ref_grouped_attention
from ref_kernels.linear import linear as ref_linear
from ref_kernels.rms_norm import rms_norm as ref_rms_norm
from ref_kernels.rope import rope as ref_rope
from ref_kernels.silu import silu as ref_silu


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


def _resolve_linear() -> Any:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            return default_linear
        case KernelBackend.KERNELS:
            return learner_linear
        case KernelBackend.REF_KERNELS:
            return ref_linear
        case _:
            _unknown_backend()


def linear(*args: Any, **kwargs: Any) -> torch.Tensor:
    return _resolve_linear()(*args, **kwargs)


def _resolve_rms_norm() -> Any:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            return default_rms_norm
        case KernelBackend.KERNELS:
            return learner_rms_norm
        case KernelBackend.REF_KERNELS:
            return ref_rms_norm
        case _:
            _unknown_backend()


def rms_norm(*args: Any, **kwargs: Any) -> torch.Tensor:
    return _resolve_rms_norm()(*args, **kwargs)


def _resolve_silu() -> Any:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            return default_silu
        case KernelBackend.KERNELS:
            return learner_silu
        case KernelBackend.REF_KERNELS:
            return ref_silu
        case _:
            _unknown_backend()


def silu(*args: Any, **kwargs: Any) -> torch.Tensor:
    return _resolve_silu()(*args, **kwargs)


def _resolve_rope() -> Any:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            return default_rope
        case KernelBackend.KERNELS:
            return learner_rope
        case KernelBackend.REF_KERNELS:
            return ref_rope
        case _:
            _unknown_backend()


def rope(*args: Any, **kwargs: Any) -> torch.Tensor:
    return _resolve_rope()(*args, **kwargs)


def _resolve_grouped_attention() -> Any:
    match get_kernel_backend():
        case KernelBackend.DEFAULT:
            return default_grouped_attention
        case KernelBackend.KERNELS:
            return learner_grouped_attention
        case KernelBackend.REF_KERNELS:
            return ref_grouped_attention
        case _:
            _unknown_backend()


def grouped_attention(*args: Any, **kwargs: Any) -> torch.Tensor:
    return _resolve_grouped_attention()(*args, **kwargs)


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
