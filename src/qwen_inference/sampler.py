import copy

import torch


# TODO: maybe kernelize?
def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: torch.Tensor):
        if temp == 0:
            return torch.argmax(logprobs, dim=-1)
        logprobs = copy.copy(logprobs)  # TODO: do we really need a copy?
        if top_k is not None and top_k > 0:
            topk_values = torch.topk(logprobs, k=top_k, dim=-1).values
            threshold = topk_values[:, -1, None]
            logprobs = torch.where(logprobs >= threshold, logprobs, -torch.inf)
        if top_p is not None and top_p > 0:
            sorted_logprobs, sorted_idx = torch.sort(logprobs, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logprobs, dim=-1)
            keep = torch.cumsum(sorted_probs, dim=-1) <= top_p
            keep[..., 0] = True
            filtered_sorted = torch.where(keep, sorted_logprobs, -torch.inf)
            logprobs = torch.full_like(logprobs, -torch.inf)
            logprobs.scatter_(dim=-1, index=sorted_idx, src=filtered_sorted)
        logprobs = logprobs / temp
        probs = torch.softmax(logprobs, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sample
