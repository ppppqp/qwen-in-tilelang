from __future__ import annotations

import pytest

from qwen_inference.profiler import InferenceProfiler, _format_bytes


class FakeClock:
    def __init__(self, values: list[float]):
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


def test_inference_profiler_records_steps_and_throughput():
    clock = FakeClock([0.0, 0.1, 0.4, 0.5, 0.9, 1.0])
    profiler = InferenceProfiler(device="cpu", clock=clock)

    profiler.start(prompt_tokens=8)
    profiler.step_begin()
    profiler.step_end(output_token=True)
    profiler.step_begin()
    profiler.step_end(output_token=False)
    profile = profiler.finish()

    assert profile.prompt_tokens == 8
    assert profile.inference_steps == 2
    assert profile.output_tokens == 1
    assert profile.total_seconds == 1.0
    assert profile.step_seconds == pytest.approx([0.3, 0.4])
    assert profile.steps_per_second == 2.0
    assert profile.output_tokens_per_second == 1.0
    assert profile.average_step_seconds == pytest.approx(0.35)
    assert profile.max_step_seconds == pytest.approx(0.4)


def test_inference_profiler_summary_includes_common_metrics():
    clock = FakeClock([0.0, 0.1, 0.2, 0.4])
    profiler = InferenceProfiler(device="cpu", clock=clock)

    profiler.start(prompt_tokens=4)
    profiler.step_begin()
    profiler.step_end(output_token=True)
    profiler.finish()

    summary = profiler.format_summary()
    assert "Inference profile:" in summary
    assert "prompt tokens: 4" in summary
    assert "inference steps: 1" in summary
    assert "output tokens/sec:" in summary
    assert "avg step latency:" in summary


def test_format_bytes_uses_binary_units():
    assert _format_bytes(512) == "512.00 B"
    assert _format_bytes(1024) == "1.00 KiB"
    assert _format_bytes(1024 * 1024) == "1.00 MiB"
