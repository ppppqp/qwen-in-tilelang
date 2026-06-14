from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


class QwenTokenizer:
    def __init__(self, tokenizer: Tokenizer, eos_token_id: int):
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "QwenTokenizer":
        model_dir = Path(model_dir)
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        eos_token = _read_eos_token(model_dir)
        eos_token_id = tokenizer.token_to_id(eos_token)
        if eos_token_id is None:
            raise ValueError(f"EOS token {eos_token!r} not found in tokenizer.json")
        return cls(tokenizer, eos_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]:
        prompt_parts: list[str] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            prompt_parts.append("<|im_start|>assistant\n")
            if not enable_thinking:
                prompt_parts.append("<think>\n\n</think>\n\n")
        prompt = "".join(prompt_parts)
        if tokenize:
            return self.encode(prompt, add_special_tokens=False)
        return prompt


def _read_eos_token(model_dir: Path) -> str:
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.exists():
        return "<|im_end|>"

    with config_path.open() as f:
        config = json.load(f)

    eos_token = config.get("eos_token", "<|im_end|>")
    if isinstance(eos_token, str):
        return eos_token
    if isinstance(eos_token, dict):
        content = eos_token.get("content")
        if isinstance(content, str):
            return content
    return "<|im_end|>"
