import os

from tilelang.jit import JITImpl
import torch


def _compile_target(inputs: list) -> dict[str, str] | None:
    if os.environ.get("TILELANG_TARGET"):
        return None

    for input_value in inputs:
        if isinstance(input_value, torch.Tensor) and input_value.is_cuda:
            major, minor = torch.cuda.get_device_capability(input_value.device)
            return {"kind": "cuda", "arch": f"sm_{major}{minor}"}

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return {"kind": "cuda", "arch": f"sm_{major}{minor}"}

    return {"kind": "cuda", "arch": "sm_80"}


def run_kernel(
    kernel: JITImpl,
    inputs: list,
    tl_hyper_params: dict = {},
) -> torch.tensor:
    target = _compile_target(inputs)
    previous_target = kernel.target
    if target is not None:
        kernel.target = target
    try:
        tl_kernel = kernel.compile(**tl_hyper_params)
    finally:
        kernel.target = previous_target
    output = tl_kernel(*inputs)
    return output
