# Linear

## Goal

Implement the dense projection used throughout the transformer:

```text
Y = X @ W + b
```

Work in:

- `kernels/linear.py`

Compare against:

- `ref_kernels/linear.py`
- `src/qwen_inference/kernels/linear.py`

## TileLang Concepts

- Shared memory tiles
- `T.gemm`
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
- Accumulator fragments
- Optional bias handling in the torch-facing wrapper

## Design Considerations

- Reuse `X` and `W` tiles instead of reloading from global memory repeatedly.
- Tune tile sizes around matrix shape and hardware occupancy.
- Keep the wrapper signature compatible with inference.
- Greedy decoding calls this kernel for QKV projections, MLP projections, and
  the final logits projection.

## Checkpoint

```bash
python -m pytest tests/test_course_kernels.py -k linear
```

## Challenge

Start with a simple tiled matmul. Then add bias support and verify the same
function can run inside `Qwen3MLP`.
