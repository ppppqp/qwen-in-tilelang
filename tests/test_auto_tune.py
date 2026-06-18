from __future__ import annotations

import torch

from qwen_inference.auto_tune.tuner import (
    BenchmarkOptions,
    TuneSpec,
    autotune,
    config_grid,
)
from qwen_inference.auto_tune.qwen import qwen3_06b_grouped_attention_spec


def test_config_grid_builds_cartesian_product_in_order():
    configs = config_grid(BLOCK_M=[16, 32], BLOCK_N=[64, 128])

    assert configs == [
        {"BLOCK_M": 16, "BLOCK_N": 64},
        {"BLOCK_M": 16, "BLOCK_N": 128},
        {"BLOCK_M": 32, "BLOCK_N": 64},
        {"BLOCK_M": 32, "BLOCK_N": 128},
    ]


def test_autotune_filters_incorrect_candidates_and_selects_fastest():
    def input_factory(device: torch.device) -> list[torch.Tensor]:
        return [torch.tensor([1.0, 2.0], device=device)]

    def reference(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    def candidate_factory(config):
        def candidate(x: torch.Tensor) -> torch.Tensor:
            if config["name"] == "bad":
                return x + 2
            return x + 1

        return candidate

    spec = TuneSpec(
        name="fake_add",
        configs=[
            {"name": "bad"},
            {"name": "good"},
        ],
        input_factory=input_factory,
        reference=reference,
        candidate_factory=candidate_factory,
    )

    report = autotune(
        spec,
        BenchmarkOptions(
            device="cpu",
            warmup=0,
            repeat=1,
            clone_inputs=False,
        ),
    )

    assert len(report.results) == 2
    assert report.results[0].compiled
    assert not report.results[0].correct
    assert report.results[0].error == "candidate output did not match reference"
    assert report.best is report.results[1]
    assert report.best.config == {"name": "good"}
    assert "Autotune report: fake_add" in report.format_summary()


def test_autotune_records_candidate_factory_errors():
    def input_factory(device: torch.device) -> list[torch.Tensor]:
        return [torch.tensor([1.0], device=device)]

    def candidate_factory(config):
        raise RuntimeError("compile failed")

    spec = TuneSpec(
        name="compile_error",
        configs=[{"BLOCK_M": 16}],
        input_factory=input_factory,
        reference=lambda x: x,
        candidate_factory=candidate_factory,
    )

    report = autotune(
        spec,
        BenchmarkOptions(device="cpu", warmup=0, repeat=1),
    )

    assert report.best is None
    assert report.results[0].error == "compile failed"
    assert "RuntimeError: compile failed" in report.results[0].traceback


def test_qwen3_06b_grouped_attention_spec_uses_model_shape():
    spec = qwen3_06b_grouped_attention_spec(batch_size=2, seq_len=32)
    q, k, v = spec.input_factory(torch.device("cpu"))

    assert spec.name == "qwen3_0_6b_grouped_attention_B2_S32_causal1"
    assert q.shape == (2, 32, 16, 128)
    assert k.shape == (2, 32, 8, 128)
    assert v.shape == (2, 32, 8, 128)
    assert {"BLOCK_L": 16, "BLOCK_S": 16, "THREADS": 32} in spec.configs
