from __future__ import annotations

from .qwen import Qwen3Model
from typing import Any, Callable
import torch


def _release_kv_cache(kv_cache):
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3Model,
    tokenizer: Any,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
    device: str | torch.device = "cuda",
    max_new_tokens: int = 128,
) -> str:
    def _step(model, y):
        logits = model(y[None])
        logits = logits[:, -1, :]
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
    # generate/decode
    for _ in range(max_new_tokens):
        token = _step(model, tokens)
        token_id = int(token.item())
        tokens = torch.cat([tokens, token])
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        print(tokenizer.decode(generated, skip_special_tokens=True), end="\r", flush=True)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text, flush=True)
    return text
