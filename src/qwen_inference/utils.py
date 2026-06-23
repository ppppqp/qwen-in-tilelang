import logging

from tilelang.jit import JITImpl
import torch
import time


class KernelCache:
    def __init__(self):
        self.cache = {}

    def get(self, kernel: JITImpl, **hyper_params):

        key = (kernel.func.__name__, frozenset(hyper_params.items()))
        logging.debug(
            "Checking kernel cache for %s with hyper params %s",
            kernel.func.__name__,
            hyper_params,
        )
        return self.cache.get(key)

    def set(self, kernel: JITImpl, compiled_kernel, **hyper_params):
        key = (kernel.func.__name__, frozenset(hyper_params.items()))
        self.cache[key] = compiled_kernel


kernel_cache = KernelCache()


def run_kernel(
    kernel: JITImpl,
    inputs: list,
    tl_hyper_params: dict = {},
) -> torch.tensor:
    cached_kernel = kernel_cache.get(kernel, **tl_hyper_params)
    if cached_kernel is not None:
        logging.debug(
            "Using cached kernel for %s with hyper params %s",
            kernel.func.__name__,
            tl_hyper_params,
        )
        return cached_kernel(*inputs)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    tl_kernel = kernel.compile(**tl_hyper_params)
    kernel_cache.set(kernel, tl_kernel, **tl_hyper_params)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    output = tl_kernel(*inputs)

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    logging.debug("Kernel compilation time: %.3f ms", (t1 - t0) * 1000)
    logging.debug("Kernel execution time: %.3f ms", (t2 - t1) * 1000)
    return output
