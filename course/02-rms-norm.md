# RMSNorm

## Goal

Implement row-wise RMSNorm:

```text
y = x * rsqrt(mean(x * x) + eps) * weight
```

Work in:

- `kernels/rms_norm.py`

Compare against:

- `ref_kernels/rms_norm.py`
- `src/qwen_inference/kernels/rms_norm.py`

## TileLang Concepts

- Row-wise reductions
- `T.reduce_sum`
- Accumulating in `float32`
- Applying per-channel weights

## Design Considerations

- The reduction dimension is the hidden size.
- Accumulate square sums in higher precision.
- Decide whether each program handles one row or a block of rows.
- `BLOCK_N` should match or cover the hidden dimension for the first version.

## Checkpoint

```bash
python -m pytest tests/test_course_kernels.py -k rms_norm
```

## Challenge

Try a smaller `BLOCK_N`. Explain whether the current algorithm still works and
what would need to change for multi-block reductions.
