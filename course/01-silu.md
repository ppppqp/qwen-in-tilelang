# SiLU

## Goal

Implement the SiLU activation:

```text
silu(x) = x / (1 + exp(-x))
```

Work in:

- `kernels/silu.py`

Compare against:

- `ref_kernels/silu.py`
- `src/qwen_inference/kernels/silu.py`

## TileLang Concepts

- `@tilelang.jit`
- `T.Tensor`
- `T.Kernel`
- `T.alloc_fragment`
- `T.copy`
- `T.Parallel`
- Elementwise scalar math

## Design Considerations

- Tile a 2D tensor into blocks.
- Keep input and output in `float16`.
- Use local fragments so the kernel structure is visible and easy to inspect.
- Start with correctness before tuning block sizes.

## Checkpoint

```bash
python -m pytest tests/test_course_kernels.py -k silu
```

## Challenge

Change `BLOCK_M` and `BLOCK_N`, then record whether correctness or latency
changes.
