# TileLang Qwen Inference Course

This course teaches TileLang by rebuilding the kernels used by a small Qwen
inference engine. The end goal is practical: learners replace the files in
`kernels/`, then run inference with the kernels they wrote.

## Course Loop

Each lesson follows the same loop:

1. Read the PyTorch/reference behavior.
2. Implement or improve one TileLang kernel in `kernels/`.
3. Compare against `ref_kernels/`.
4. Run the focused tests.
5. Run inference with `--kernel-backend kernels`.

## Setup

Use the normal Python packaging flow for the course:

```bash
pip install -e .
```

See [setup.md](setup.md) for environment details.

## Backend Layout

- `qwen_inference.kernels`: production kernels used by the default backend.
- `ref_kernels`: known-good course references.
- `kernels`: learner workspace. These files initially delegate to `ref_kernels`
  so the course runs before learners replace implementations.

Run inference with the learner backend:

```bash
python main.py --kernel-backend kernels --prompt "Explain TileLang in one sentence."
```

Run inference with the reference backend:

```bash
python main.py --kernel-backend ref_kernels --prompt "Explain TileLang in one sentence."
```

## Lessons

1. [Orientation](00-orientation.md)
2. [SiLU](01-silu.md)
3. [RMSNorm](02-rms-norm.md)
4. [Linear](03-linear.md)
5. [RoPE](04-rope.md)
6. [Attention](05-attention.md)
7. [Grouped Attention](06-grouped-attention.md)
8. [Run Your Qwen](07-run-your-qwen.md)
9. [Profiling](08-profiling.md)
10. [Tuning](09-tuning.md)
11. [Fusion](10-fusion.md)

## Verification

Focused course tests:

```bash
python -m pytest tests/test_course_kernels.py tests/test_kernel_backend.py
```

Full focused generation check:

```bash
python -m pytest tests/test_generate.py tests/test_course_kernels.py
```

The CUDA kernel tests require a CUDA-capable PyTorch and GPU.
