# Fusion

## Goal

Understand when combining kernels can reduce memory traffic.

Potential fusions:

- Linear + bias
- RMSNorm + linear
- Gate projection + SiLU + up projection multiply
- Attention score + mask + softmax

## Design Considerations

- Fusion can remove intermediate tensor writes.
- Fusion can increase register pressure.
- Fusion can make debugging harder.
- The fused kernel should still have a clear reference implementation.

## Exercise

Pick one simple fusion candidate and write a PyTorch reference first. Then
outline the TileLang implementation before writing code.

## Capstone

Fuse part of the MLP path:

```text
gate = linear(x, w_gate)
up = linear(x, w_up)
out = silu(gate) * up
```

Compare correctness and latency against the unfused path.
