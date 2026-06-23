# Profiling

## Goal

Measure the inference engine and individual kernel behavior.

Relevant files:

- `src/qwen_inference/profiler.py`
- `src/qwen_inference/utils.py`
- `main.py`

## Concepts

- CUDA synchronization before timing
- Compile time vs execution time
- First-token latency vs later-token latency
- Tokens per second
- Peak CUDA memory
- Kernel cache effects

## Exercise

Run a short prompt twice:

```bash
python main.py --kernel-backend kernels --max-new-tokens 8
python main.py --kernel-backend kernels --max-new-tokens 8
```

Compare the profile output. Explain why the first run may include more compile
or cache overhead.

## Challenge

Change one block size in `kernels/linear.py`, rerun the test and inference, and
record both correctness and latency.
