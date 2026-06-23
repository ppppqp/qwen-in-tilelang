# Grouped Attention

## Goal

Implement Qwen-style grouped-query attention.

Work in:

- `kernels/grouped_attention.py`

Compare against:

- `ref_kernels/grouped_attention.py`
- `src/qwen_inference/kernels/grouped_attention.py`

## TileLang Concepts

- Online softmax
- Causal block traversal
- `T.gemm` for QK and PV phases
- Shared memory for Q, K, and V tiles
- Group mapping from query heads to key/value heads

## Design Considerations

- Input layouts:
  - `Q`: `(batch, query_len, query_heads, head_dim)`
  - `K`: `(batch, seq_len, kv_heads, head_dim)`
  - `V`: `(batch, seq_len, kv_heads, head_dim)`
- `query_heads` must be divisible by `kv_heads`.
- Causal mode should not attend to future positions.
- Online softmax avoids storing the full attention score matrix.

## Checkpoint

```bash
python -m pytest tests/test_course_kernels.py -k grouped_attention
```

## Challenge

Compare latency as sequence length grows. Explain which part of the kernel
scales with sequence length.
