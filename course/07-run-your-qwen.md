# Run Your Qwen

## Goal

Run inference with the learner kernels.

Command:

```bash
python main.py --kernel-backend kernels --prompt "Explain TileLang in one sentence." --max-new-tokens 32
```

Reference comparison:

```bash
python main.py --kernel-backend ref_kernels --prompt "Explain TileLang in one sentence." --max-new-tokens 32
```

## Milestone Checklist

- `kernels/silu.py` passes its checkpoint.
- `kernels/rms_norm.py` passes its checkpoint.
- `kernels/linear.py` passes its checkpoint.
- `kernels/rope.py` passes its checkpoint.
- `kernels/grouped_attention.py` passes its checkpoint.
- `python main.py --kernel-backend kernels` generates text.

## Debugging Strategy

When inference breaks, narrow the problem:

1. Run the single-kernel course test.
2. Compare learner output against `ref_kernels`.
3. Run the Qwen module tests.
4. Run short inference with `--max-new-tokens 1`.
5. Increase token count once the first step is correct.
