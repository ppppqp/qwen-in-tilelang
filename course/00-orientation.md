# Orientation

## Goal

Understand how the repo is wired before writing kernels.

By the end, learners should know:

- Where production kernels live: `src/qwen_inference/kernels/`
- Where learner kernels live: `kernels/`
- Where reference kernels live: `ref_kernels/`
- How inference chooses a backend: `--kernel-backend`
- How each torch-facing wrapper calls a TileLang `*_kernel`

## Read First

- `src/qwen_inference/kernel_backend.py`
- `kernels/linear.py`
- `ref_kernels/linear.py`
- `src/qwen_inference/kernels/linear.py`
- `src/qwen_inference/utils.py`

## Exercise

Run the reference backend and learner backend. At the start of the course they
should behave the same because `kernels/` delegates to `ref_kernels/`.

```bash
python main.py --kernel-backend ref_kernels --max-new-tokens 8
python main.py --kernel-backend kernels --max-new-tokens 8
```

## Design Notes

The backend split lets learners replace one kernel at a time without changing
the model. The final reward is running the same inference command with their
own kernels.
