from __future__ import annotations
from typing import Callable
from tilelang.jit import JITImpl, JITKernel
from tilelang.engine.param import KernelParam
import torch


def rand_torch_tensor(
    shape: list[int], dtype: torch.dtype, device="cuda"
) -> torch.Tensor:
    """Get a random torch tensor."""

    if dtype == torch.float16:
        # Uniform distribution, N(0, 1), range (-1, 1) for float16.
        torch_tensor = torch.randn(shape, dtype=torch.float16, device=device)
    elif dtype == torch.float32:
        torch_tensor = torch.randn(shape, dtype=torch.float32, device=device)
    elif dtype == torch.uint8:
        torch_tensor = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}, {type(dtype)}")

    return torch_tensor


def _tvm_ffi_dtype_to_torch_dtype(ffi_dtype) -> torch.dtype:
    if ffi_dtype == "float16":
        return torch.float16
    elif ffi_dtype == "float32":
        return torch.float32
    elif ffi_dtype == "uint8":
        return torch.uint8
    elif ffi_dtype == "int32":
        return torch.int32
    elif ffi_dtype == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported dtype: {ffi_dtype}")


def _torch_tensor_materialize(params: list[KernelParam]):
    inputs_in_torch_tensors: list[torch.Tensor] = []

    for idx, tl_param in enumerate(params):
        if idx == len(params) - 1:
            # Skip the last param (output)
            continue

        shape = tl_param.shape
        dtype = tl_param.dtype
        torch_dtype = _tvm_ffi_dtype_to_torch_dtype(dtype)
        inputs_in_torch_tensors.append(
            rand_torch_tensor(shape, torch_dtype, device="cuda")
        )

    return inputs_in_torch_tensors


def test_kernel(
    kernel_tl: JITImpl,
    kernel_torch: Callable,
    tl_hyper_params: dict = {},
    print_log: bool = False,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    """Test a puzzle solution with given hyper parameters."""

    tl_kernel: JITKernel = kernel_tl.compile(**tl_hyper_params)
    # print(tl_kernel.get_kernel_source())
    # exit()
    inputs_in_torch_tensors = _torch_tensor_materialize(tl_kernel.params)

    # As the kernel may modify the input tensors, we make a copy of them.
    inputs_copy = [i.clone() for i in inputs_in_torch_tensors]

    output_torch = kernel_torch(*inputs_in_torch_tensors)
    output_tl = tl_kernel(*inputs_copy)

    match = torch.allclose(output_torch, output_tl, atol=atol, rtol=rtol)
    match_emoji = "✅" if match else "❌"
    print(match_emoji, "Results match:", match)

    if not match or print_log:
        # print("Hyper parameters: ", hyper_params)
        print("TileLang Hyper parameters: ", tl_hyper_params)
        print("Inputs: ", inputs_in_torch_tensors)
        print("Yours:", output_tl.dtype, output_tl.shape, "\n", output_tl)
        print("Spec:", output_torch.dtype, output_torch.shape, "\n", output_torch)
        diff = torch.isclose(output_torch, output_tl, atol=atol, rtol=rtol)
        print(
            "Diff (True: correct, False: incorrect):",
            "\n",
            diff,
        )
        # idx = torch.where(~diff)
        # print(idx)
        print("Max diff:", torch.max(torch.abs(output_torch - output_tl)))
        print("Mean diff:", torch.mean(torch.abs(output_torch - output_tl)))
