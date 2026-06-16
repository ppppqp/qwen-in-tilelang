from __future__ import annotations
import logging

from .qwen import Qwen3Model
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


def simple_generate(
    model: Qwen3Model,
    tokenizer: Any,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
    device: str | torch.device = "cuda",
    max_new_tokens: int = 128,
) -> str:
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

    # print("tokens shape:", tokens.shape)
    generated: list[int] = []
    # generate/decode
    for i in range(max_new_tokens):
        logging.info(f"Step {i}, current tokens: {len(generated)}")
        token = _step(model, tokens)
        token_id = int(token.item())
        tokens = torch.cat([tokens, token])
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        print(
            tokenizer.decode(generated, skip_special_tokens=True), end="\r", flush=True
        )
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text, flush=True)
    return text
