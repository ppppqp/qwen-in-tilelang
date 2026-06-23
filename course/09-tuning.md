# Tuning

## Goal

Learn how kernel parameters affect correctness and performance.

Start with:

- `kernels/linear.py`
- `kernels/rms_norm.py`
- `kernels/grouped_attention.py`

## Parameters To Explore

- `BLOCK_M`
- `BLOCK_N`
- `BLOCK_K`
- `BLOCK_L`
- `BLOCK_S`
- `THREADS`

## Rules

1. Correctness comes first.
2. Benchmark after warmup.
3. Change one parameter at a time.
4. Record the shape, backend, device, and timing.
5. Compare against `ref_kernels`.

## Exercise

Create a small table:

```text
kernel | shape | parameter change | correct | latency
```

Use it to tune `linear` for one Qwen projection shape.

## Challenge

Find one tuning change that improves latency without changing model output.
