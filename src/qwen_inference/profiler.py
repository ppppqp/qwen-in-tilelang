from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import torch


@dataclass
class InferenceProfile:
    prompt_tokens: int = 0
    inference_steps: int = 0
    output_tokens: int = 0
    total_seconds: float = 0.0
    step_seconds: list[float] = field(default_factory=list)
    peak_memory_allocated_bytes: int | None = None
    peak_memory_reserved_bytes: int | None = None
    memory_allocated_bytes: int | None = None
    memory_reserved_bytes: int | None = None

    @property
    def steps_per_second(self) -> float:
        if self.total_seconds == 0:
            return 0.0
        return self.inference_steps / self.total_seconds

    @property
    def output_tokens_per_second(self) -> float:
        if self.total_seconds == 0:
            return 0.0
        return self.output_tokens / self.total_seconds

    @property
    def average_step_seconds(self) -> float:
        if not self.step_seconds:
            return 0.0
        return sum(self.step_seconds) / len(self.step_seconds)

    @property
    def max_step_seconds(self) -> float:
        if not self.step_seconds:
            return 0.0
        return max(self.step_seconds)


class InferenceProfiler:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        synchronize_cuda: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.device = torch.device(device)
        self.synchronize_cuda = synchronize_cuda
        self.clock = clock
        self.profile = InferenceProfile()
        self._start_time: float | None = None
        self._step_start_time: float | None = None

    def start(self, prompt_tokens: int = 0) -> None:
        self.profile = InferenceProfile(prompt_tokens=prompt_tokens)
        if self._is_cuda_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self._synchronize()
        self._start_time = self.clock()

    def step_begin(self) -> None:
        self._synchronize()
        self._step_start_time = self.clock()

    def step_end(self, *, output_token: bool = True) -> None:
        if self._step_start_time is None:
            raise RuntimeError("step_end called before step_begin")
        self._synchronize()
        elapsed = self.clock() - self._step_start_time
        self.profile.step_seconds.append(elapsed)
        self.profile.inference_steps += 1
        if output_token:
            self.profile.output_tokens += 1
        self._step_start_time = None

    def finish(self) -> InferenceProfile:
        if self._start_time is None:
            raise RuntimeError("finish called before start")
        self._synchronize()
        self.profile.total_seconds = self.clock() - self._start_time
        if self._is_cuda_available():
            self.profile.peak_memory_allocated_bytes = torch.cuda.max_memory_allocated(
                self.device
            )
            self.profile.peak_memory_reserved_bytes = torch.cuda.max_memory_reserved(
                self.device
            )
            self.profile.memory_allocated_bytes = torch.cuda.memory_allocated(
                self.device
            )
            self.profile.memory_reserved_bytes = torch.cuda.memory_reserved(self.device)
        return self.profile

    def format_summary(self) -> str:
        return format_profile_summary(self.profile)

    def _is_cuda_available(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _synchronize(self) -> None:
        if self.synchronize_cuda and self._is_cuda_available():
            torch.cuda.synchronize(self.device)


def format_profile_summary(profile: InferenceProfile) -> str:
    lines = [
        "Inference profile:",
        f"  prompt tokens: {profile.prompt_tokens}",
        f"  inference steps: {profile.inference_steps}",
        f"  output tokens: {profile.output_tokens}",
        f"  total time: {profile.total_seconds:.4f}s",
        f"  steps/sec: {profile.steps_per_second:.2f}",
        f"  output tokens/sec: {profile.output_tokens_per_second:.2f}",
        f"  avg step latency: {profile.average_step_seconds * 1000:.2f} ms",
        f"  max step latency: {profile.max_step_seconds * 1000:.2f} ms",
    ]
    if profile.peak_memory_allocated_bytes is not None:
        lines.extend(
            [
                "  CUDA memory:",
                f"    peak allocated: {_format_bytes(profile.peak_memory_allocated_bytes)}",
                f"    peak reserved: {_format_bytes(profile.peak_memory_reserved_bytes or 0)}",
                f"    final allocated: {_format_bytes(profile.memory_allocated_bytes or 0)}",
                f"    final reserved: {_format_bytes(profile.memory_reserved_bytes or 0)}",
            ]
        )
    return "\n".join(lines)


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"

