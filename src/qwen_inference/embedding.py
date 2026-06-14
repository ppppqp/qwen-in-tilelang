from qwen_inference.basics import linear

# from qwen_inference.quantize import QuantizedWeights, quantized_linear
import torch
from qwen_inference.utils import run_kernel


class Embedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: torch.Tensor,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: what is this?
        # array index lookup
        return self.weight[x, :]

    def as_linear(self, x: torch.Tensor) -> torch.Tensor:
        return run_kernel(
            kernel=linear,
            inputs=[
                x,
                self.weight,
                torch.zeros(self.embedding_dim, dtype=torch.float16),
            ],
            tl_hyper_params={"BLOCK_M": 128, "BLOCK_N": 128},
        )


# class QuantizedEmbedding:
#     def __init__(
#         self,
#         vocab_size: int,
#         embedding_dim: int,
#         weight: QuantizedWeights,
#     ):
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.weight = weight

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         biases = self.weight.biases[x] if self.weight.biases is not None else None
#         return mx.dequantize(
#             self.weight.weight[x],
#             self.weight.scales[x],
#             biases,
#             self.weight.group_size,
#             self.weight.bits,
#         )

#     def as_linear(self, x: torch.Tensor) -> torch.Tensor:
#         return quantized_linear(x, self.weight)
