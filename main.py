import argparse
import logging
import torch
import tilelang
from qwen_inference.model import load_qwen3_model_from_files
from qwen_inference.sampler import make_sampler
from qwen_inference.generate import simple_generate
from qwen_inference.profiler import InferenceProfiler
from qwen_inference.tokenizer import QwenTokenizer

"""
Example usage:
python main.py \
    --prompt "Give me a short introduction to large language model." \
    --max-new-tokens 128
"""


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="models/Qwen3-0.6B",
    help="Path to a local Qwen3 directory containing config.json and .safetensors files.",
)
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
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument(
    "--dtype",
    type=str,
    default="float16",
    choices=("float16", "bfloat16", "float32"),
)
parser.add_argument(
    "--logging-level",
    default="info",
    choices=("debug", "info", "warning", "error", "critical"),
    help="Logging level (debug, info, warning, error)",
)


args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.logging_level.upper()),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)
dtype_by_name = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

model = load_qwen3_model_from_files(
    args.model,
    device=args.device,
    dtype=dtype_by_name[args.dtype],
)

tokenizer = QwenTokenizer.from_pretrained(args.model)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": args.prompt},
]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=args.enable_thinking,
)
sampler = make_sampler(
    args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
)
profiler = InferenceProfiler(device=args.device)
response = simple_generate(
    model,
    tokenizer,
    prompt,
    sampler=sampler,
    device=args.device,
    max_new_tokens=args.max_new_tokens,
    profiler=profiler,
)
print(f"\nGenerated response: {response}")
print(profiler.format_summary())
