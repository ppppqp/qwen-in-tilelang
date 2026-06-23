# Setup

## Install

From the repo root:

```bash
pip install -e .
```

This installs:

- `qwen_inference`
- `kernels`
- `ref_kernels`

TileLang, PyTorch, and CUDA should be installed in the active environment used
for the course.

## Verify Imports

```bash
python -c "import qwen_inference, kernels, ref_kernels; print('ok')"
```

## Verify Backend Selection

```bash
python main.py --help
```

The help output should include:

```text
--kernel-backend {default,kernels,ref_kernels}
```

## Run Tests

Backend tests:

```bash
python -m pytest tests/test_kernel_backend.py
```

Course kernel checks:

```bash
python -m pytest tests/test_course_kernels.py
```

The course kernel checks require a CUDA GPU.

## Run Inference

Reference backend:

```bash
python main.py --kernel-backend ref_kernels --max-new-tokens 16
```

Learner backend:

```bash
python main.py --kernel-backend kernels --max-new-tokens 16
```
