from __future__ import annotations
import logging
import sys

from .qwen import Qwen3Model
from .profiler import InferenceProfiler
from typing import Any, Callable
import torch


def _release_kv_cache(kv_cache):
    for layer in kv_cache:
        layer.release()


def _pad_tokens_to_multiple(tokens: torch.Tensor, multiple: int) -> torch.Tensor:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    padding = (-tokens.shape[0]) % multiple
    if padding == 0:
        return tokens
    return torch.nn.functional.pad(tokens, (0, padding))


def _print_generation_progress(step: int, total: int) -> None:
    width = 30
    filled = width * step // total if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rGenerating [{bar}] {step}/{total}\033[K",
        end="",
        file=sys.stderr,
        flush=True,
    )


def simple_generate(
    model: Qwen3Model,
    tokenizer: Any,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
    device: str | torch.device = "cuda",
    max_new_tokens: int = 128,
    profiler: InferenceProfiler | None = None,
) -> str:
    profiler = profiler or InferenceProfiler(device=device)

    def _step(model, y):
        padded_y = _pad_tokens_to_multiple(y, multiple=16)
        logits = model(padded_y[None])
        logits = logits[:, y.shape[0] - 1, :]
        logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        if sampler is None:
            y = torch.argmax(logprobs, dim=-1)
        else:
            y = sampler(logprobs)
        return y

    # prefill with the prompt
    tokens = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        device=device,
        dtype=torch.long,
    )

    generated: list[int] = []
    text = ""
    profiler.start(prompt_tokens=tokens.shape[0])
    # generate/decode
    for i in range(max_new_tokens):
        logging.debug(f"Step {i}, current tokens: {len(generated)}")
        profiler.step_begin()
        token = _step(model, tokens)
        token_id = int(token.item())
        profiler.step_end(output_token=token_id != tokenizer.eos_token_id)
        tokens = torch.cat([tokens, token])
        _print_generation_progress(i + 1, max_new_tokens)
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        next_text = tokenizer.decode(generated, skip_special_tokens=True)
        if next_text.startswith(text):
            print(next_text[len(text) :], end="", flush=True)
        else:
            print(next_text, end="", flush=True)
        text = next_text
    print(file=sys.stderr, flush=True)
    print(flush=True)
    profiler.finish()
    return text
