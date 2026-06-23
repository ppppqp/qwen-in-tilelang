import pytest

from qwen_inference import kernel_backend


def test_kernel_backend_can_select_course_backends():
    token = kernel_backend.set_kernel_backend(kernel_backend.KernelBackend.DEFAULT)
    try:
        kernel_backend.set_kernel_backend("kernels")
        assert kernel_backend.get_kernel_backend() == kernel_backend.KernelBackend.KERNELS

        kernel_backend.set_kernel_backend("ref_kernels")
        assert (
            kernel_backend.get_kernel_backend()
            == kernel_backend.KernelBackend.REF_KERNELS
        )
    finally:
        kernel_backend.reset_kernel_backend(token)


def test_kernel_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown kernel backend"):
        kernel_backend.set_kernel_backend("missing")
