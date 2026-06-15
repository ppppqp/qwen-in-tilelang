from tilelang.jit import JITImpl
import torch


def run_kernel(
    kernel: JITImpl,
    inputs: list,
    tl_hyper_params: dict = {},
) -> torch.tensor:
    tl_kernel = kernel.compile(**tl_hyper_params)
    output = tl_kernel(*inputs)
    return output
