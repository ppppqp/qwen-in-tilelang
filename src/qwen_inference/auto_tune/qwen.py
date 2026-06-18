from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from qwen_inference.auto_tune.tuner import (
    BenchmarkOptions,
    TuneReport,
    TuneSpec,
    autotune,
    config_grid,
    tilelang_candidate_factory,
)
from qwen_inference.kernels.grouped_attention import grouped_attention_kernel


QWEN3_06B_ATTENTION_SHAPE = {
    "QH": 16,
    "H": 8,
    "D": 128,
}


def qwen3_06b_grouped_attention_configs() -> list[dict[str, int]]:
    return config_grid(
        BLOCK_L=[16, 32],
        BLOCK_S=[16, 32, 64],
    )


def qwen3_06b_grouped_attention_spec(
    *,
    batch_size: int = 1,
    seq_len: int = 128,
    is_causal: bool = True,
    configs: Sequence[Mapping[str, Any]] | None = None,
    dtype: torch.dtype = torch.float16,
) -> TuneSpec:
    qh = QWEN3_06B_ATTENTION_SHAPE["QH"]
    h = QWEN3_06B_ATTENTION_SHAPE["H"]
    d = QWEN3_06B_ATTENTION_SHAPE["D"]
    configs = qwen3_06b_grouped_attention_configs() if configs is None else configs

    def input_factory(device: torch.device) -> list[torch.Tensor]:
        return [
            torch.randn((batch_size, seq_len, qh, d), dtype=dtype, device=device),
            torch.randn((batch_size, seq_len, h, d), dtype=dtype, device=device),
            torch.randn((batch_size, seq_len, h, d), dtype=dtype, device=device),
        ]

    def reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=is_causal,
            enable_gqa=True,
        )
        return output.transpose(1, 2)

    static_params = {
        "N": batch_size,
        "QH": qh,
        "H": h,
        "S": seq_len,
        "D": d,
        "is_causal": is_causal,
    }
    return TuneSpec(
        name=(
            "qwen3_0_6b_grouped_attention"
            f"_B{batch_size}_S{seq_len}_causal{int(is_causal)}"
        ),
        configs=configs,
        input_factory=input_factory,
        reference=reference,
        candidate_factory=tilelang_candidate_factory(
            grouped_attention_kernel,
            static_params,
        ),
    )


def tune_qwen3_06b_grouped_attention(
    *,
    batch_size: int = 1,
    seq_len: int = 128,
    is_causal: bool = True,
    configs: Sequence[Mapping[str, Any]] | None = None,
    options: BenchmarkOptions | None = None,
) -> TuneReport:
    spec = qwen3_06b_grouped_attention_spec(
        batch_size=batch_size,
        seq_len=seq_len,
        is_causal=is_causal,
        configs=configs,
    )
    return autotune(spec, options)
