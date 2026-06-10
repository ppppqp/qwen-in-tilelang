# import mlx.core as mx

# from mlx_lm.tokenizer_utils import TokenizerWrapper
# from .kv_cache import *
from .qwen import Qwen3Model
from typing import Callable
import torch


def _release_kv_cache(kv_cache):
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3Model,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[torch.tensor], torch.tensor] | None,
) -> str:
    def _step(model, y):
        logits = model(y[None])
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(
            logits, keepdims=True
        )  # optional -- for numerical stability
        if sampler is None:
            y = mx.argmax(logprobs, axis=-1)
        else:
            y = sampler(logprobs)
        return y

    # prefill with the prompt
    tokens = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        device="cuda",
        dtype=torch.long,
    )
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    # generate/decode
    while True:
        token = _step(model, tokens)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)
