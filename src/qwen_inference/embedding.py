from qwen_inference.kernels.linear import linear
import torch


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
        # array index lookup
        return self.weight[x, :]

    def as_linear(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.embedding_dim)
        weight = self.weight.T.contiguous()
        output = linear(x_flat, weight, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)
        return output.reshape(*orig_shape[:-1], self.vocab_size)
