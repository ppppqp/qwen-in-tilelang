from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import time
import traceback
from typing import Any

import torch
from tilelang.jit import JITImpl


TensorTree = torch.Tensor | Sequence["TensorTree"]


@dataclass(frozen=True)
class BenchmarkOptions:
    warmup: int = 5
    repeat: int = 20
    device: str | torch.device = "cuda"
    rtol: float = 1e-2
    atol: float = 1e-2
    check_correctness: bool = True
    clone_inputs: bool = True
    stop_on_error: bool = False


@dataclass(frozen=True)
class TuneSpec:
    name: str
    configs: Sequence[Mapping[str, Any]]
    input_factory: Callable[[torch.device], Sequence[torch.Tensor]]
    reference: Callable[..., TensorTree] | None
    candidate_factory: Callable[[Mapping[str, Any]], Callable[..., TensorTree]]


@dataclass
class TuneResult:
    config: dict[str, Any]
    compiled: bool
    correct: bool
    latency_ms: float | None = None
    compile_seconds: float | None = None
    max_abs_diff: float | None = None
    error: str | None = None
    traceback: str | None = None


@dataclass
class TuneReport:
    name: str
    results: list[TuneResult] = field(default_factory=list)

    @property
    def successful_results(self) -> list[TuneResult]:
        return [
            result
            for result in self.results
            if result.compiled and result.correct and result.latency_ms is not None
        ]

    @property
    def best(self) -> TuneResult | None:
        successful = self.successful_results
        if not successful:
            return None
        return min(successful, key=lambda result: result.latency_ms or float("inf"))

    def sorted_results(self) -> list[TuneResult]:
        return sorted(
            self.results,
            key=lambda result: (
                not (result.compiled and result.correct),
                result.latency_ms if result.latency_ms is not None else float("inf"),
            ),
        )

    def format_summary(self) -> str:
        lines = [f"Autotune report: {self.name}"]
        best = self.best
        if best is None:
            lines.append("  best: none")
        else:
            lines.append(
                f"  best: {best.latency_ms:.4f} ms, config={best.config}"
            )
        for result in self.sorted_results():
            status = "ok" if result.compiled and result.correct else "failed"
            latency = (
                f"{result.latency_ms:.4f} ms"
                if result.latency_ms is not None
                else "n/a"
            )
            lines.append(f"  - {status:6} {latency:>12} config={result.config}")
            if result.error:
                lines.append(f"    error: {result.error}")
        return "\n".join(lines)


def tilelang_candidate_factory(
    kernel: JITImpl,
    static_params: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], Callable[..., TensorTree]]:
    def make_candidate(config: Mapping[str, Any]) -> Callable[..., TensorTree]:
        tl_hyper_params = {**static_params, **dict(config)}
        tl_kernel = kernel.compile(**tl_hyper_params)

        def run_candidate(*inputs: torch.Tensor) -> TensorTree:
            return tl_kernel(*inputs)

        return run_candidate

    return make_candidate


def autotune(spec: TuneSpec, options: BenchmarkOptions | None = None) -> TuneReport:
    options = BenchmarkOptions() if options is None else options
    device = torch.device(options.device)
    inputs = list(spec.input_factory(device))
    reference_output = None
    if options.check_correctness:
        if spec.reference is None:
            raise ValueError("reference is required when check_correctness=True")
        reference_output = spec.reference(*_clone_inputs(inputs))

    report = TuneReport(name=spec.name)
    for config in spec.configs:
        try:
            compile_start = time.perf_counter()
            candidate = spec.candidate_factory(config)
            compile_seconds = time.perf_counter() - compile_start

            max_abs_diff = None
            if options.check_correctness:
                output = candidate(*_clone_inputs(inputs))
                correct, max_abs_diff = _outputs_close(
                    reference_output,
                    output,
                    atol=options.atol,
                    rtol=options.rtol,
                )
                if not correct:
                    report.results.append(
                        TuneResult(
                            config=dict(config),
                            compiled=True,
                            correct=False,
                            compile_seconds=compile_seconds,
                            max_abs_diff=max_abs_diff,
                            error="candidate output did not match reference",
                        )
                    )
                    continue

            latency_ms = benchmark_candidate(
                candidate,
                inputs,
                warmup=options.warmup,
                repeat=options.repeat,
                clone_inputs=options.clone_inputs,
            )
            report.results.append(
                TuneResult(
                    config=dict(config),
                    compiled=True,
                    correct=True,
                    latency_ms=latency_ms,
                    compile_seconds=compile_seconds,
                    max_abs_diff=max_abs_diff,
                )
            )
        except Exception as exc:
            if options.stop_on_error:
                raise
            report.results.append(
                TuneResult(
                    config=dict(config),
                    compiled=False,
                    correct=False,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
    return report


def benchmark_candidate(
    candidate: Callable[..., TensorTree],
    inputs: Sequence[torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    clone_inputs: bool,
) -> float:
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    for _ in range(warmup):
        candidate(*_inputs_for_run(inputs, clone_inputs))
    _sync_if_cuda(inputs)

    if _has_cuda_inputs(inputs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            candidate(*_inputs_for_run(inputs, clone_inputs))
        end.record()
        torch.cuda.synchronize(inputs[0].device)
        return start.elapsed_time(end) / repeat

    start_time = time.perf_counter()
    for _ in range(repeat):
        candidate(*_inputs_for_run(inputs, clone_inputs))
    return (time.perf_counter() - start_time) * 1000 / repeat


def config_grid(**choices: Sequence[Any]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [{}]
    for name, values in choices.items():
        configs = [
            {**config, name: value}
            for config in configs
            for value in values
        ]
    return configs


def _inputs_for_run(
    inputs: Sequence[torch.Tensor],
    clone_inputs: bool,
) -> list[torch.Tensor]:
    return _clone_inputs(inputs) if clone_inputs else list(inputs)


def _clone_inputs(inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [input_tensor.clone() for input_tensor in inputs]


def _sync_if_cuda(inputs: Sequence[torch.Tensor]) -> None:
    if _has_cuda_inputs(inputs):
        torch.cuda.synchronize(inputs[0].device)


def _has_cuda_inputs(inputs: Sequence[torch.Tensor]) -> bool:
    return bool(inputs) and inputs[0].is_cuda


def _outputs_close(
    expected: TensorTree,
    actual: TensorTree,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    expected_tensors = _flatten_outputs(expected)
    actual_tensors = _flatten_outputs(actual)
    if len(expected_tensors) != len(actual_tensors):
        return False, float("inf")

    max_abs_diff = 0.0
    all_close = True
    for expected_tensor, actual_tensor in zip(expected_tensors, actual_tensors):
        if expected_tensor.shape != actual_tensor.shape:
            return False, float("inf")
        diff = torch.max(torch.abs(expected_tensor - actual_tensor)).item()
        max_abs_diff = max(max_abs_diff, float(diff))
        all_close = (
            all_close
            and torch.allclose(expected_tensor, actual_tensor, atol=atol, rtol=rtol)
        )
    return all_close, max_abs_diff


def _flatten_outputs(output: TensorTree) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    flattened: list[torch.Tensor] = []
    for item in output:
        flattened.extend(_flatten_outputs(item))
    return flattened

