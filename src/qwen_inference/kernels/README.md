# TileLang Kernels

This package keeps each TileLang kernel next to its Python driver.

- `softmax.py`: row-wise online softmax
- `linear.py`: dense matrix multiplication wrapper for transformer projections
- `silu.py`: SiLU activation
- `rms_norm.py`: RMSNorm kernel and `RMSNorm` module wrapper
- `attention.py`: simple attention lesson kernel and shared attention helpers
- `grouped_attention.py`: Qwen-style grouped-query attention
- `rope.py`: rotary position embedding kernel and `RoPE` wrapper

New kernels should live here as standalone files. Keep the TileLang `*_kernel`
function and the torch-facing wrapper in the same file so each module can be
read as a complete lesson.

