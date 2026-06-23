# RoPE

## Goal

Implement rotary position embedding for query and key tensors.

Work in:

- `kernels/rope.py`

Compare against:

- `ref_kernels/rope.py`
- `src/qwen_inference/kernels/rope.py`

## TileLang Concepts

- 4D tensor indexing
- Position-dependent math
- `T.cos`, `T.sin`, `T.pow`
- Pairwise rotation over the head dimension

## Design Considerations

- Input layout is `(batch, sequence, heads, head_dim)`.
- `head_dim` must be even.
- Preserve the original tensor dtype at the wrapper boundary.
- Compute in `float32` when evaluating trigonometric functions.

## Checkpoint

```bash
python -m pytest tests/test_course_kernels.py -k rope
```

## Challenge

Run the same input with two different offsets and inspect how token position
changes the rotated output.
