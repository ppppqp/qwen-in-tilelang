import argparse
from qwen_inference.sampler import make_sampler
from qwen_inference.generate import simple_generate
from qwen_inference.qwen import Qwen3Model, Qwen3ModelConfig
from src.qwen_inference import models

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-0.6b")
parser.add_argument(
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--sampler-temp", type=float, default=0)
parser.add_argument("--sampler-top-p", type=float, default=None)
parser.add_argument("--sampler-top-k", type=int, default=None)
parser.add_argument("--enable-thinking", action="store_true")
parser.add_argument("--enable-flash-attn", action="store_true")

args = parser.parse_args()

args.model = models.shortcut_name_to_full_name(args.model)
mlx_model, tokenizer = load(args.model)

model = Qwen3Model(mlx_model, Qwen3ModelConfig())
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": args.prompt},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=args.enable_thinking,
)
sampler = make_sampler(
    args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
)
simple_generate(model, tokenizer, prompt, sampler=sampler)
