# Attention

## Goal

Understand causal attention before moving to grouped attention.

Primary source files:

- `src/qwen_inference/kernels/attention.py`
- `tests/test_attention.py`

## Concepts

- Query/key dot products
- Causal masking
- Softmax stability
- Value accumulation
- Why materializing the full attention matrix is expensive

## Exercise

Write a small PyTorch reference for causal attention and compare it against the
existing simple attention kernel on a small shape.

## Design Considerations

- Attention combines matmul, masking, softmax, and another matmul.
- A simple implementation is valuable for learning, even if it is not the final
  inference kernel.
- The performance goal is to avoid unnecessary intermediate tensors.

## Checkpoint

```bash
python -m pytest tests/test_attention.py -k attention
```
