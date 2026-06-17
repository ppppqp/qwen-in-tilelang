from __future__ import annotations

import torch

from qwen_inference.kv_cache.kv_cache import BatchingKvCache, TinyKvFullCache


def test_tiny_kv_full_cache_appends_and_rewinds_torch_tensors():
    cache = TinyKvFullCache()
    first_key = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    first_value = first_key + 100
    second_key = torch.arange(4, 6, dtype=torch.float32).reshape(1, 1, 1, 2)
    second_value = second_key + 100

    keys, values, seq_len, mask = cache.update_and_fetch(first_key, first_value)

    assert torch.equal(keys, first_key)
    assert torch.equal(values, first_value)
    assert seq_len == 2
    assert mask is None

    keys, values, seq_len, _ = cache.update_and_fetch(second_key, second_value)

    assert seq_len == 3
    assert torch.equal(keys, torch.cat([first_key, second_key], dim=2))
    assert torch.equal(values, torch.cat([first_value, second_value], dim=2))

    cache.rewind(1)

    keys, values = cache.key_values
    assert cache.offset == 2
    assert torch.equal(keys, first_key)
    assert torch.equal(values, first_value)


def test_batching_kv_cache_right_aligns_active_requests_and_masks():
    batch_cache = BatchingKvCache(max_active_requests=2, max_seq_len=8)
    batch_cache.add_request(TinyKvFullCache(), id=0)

    prefilled = TinyKvFullCache()
    prefill_key = torch.tensor([[[[10.0, 11.0], [12.0, 13.0]]]])
    prefill_value = prefill_key + 100
    prefilled.update_and_fetch(prefill_key, prefill_value)
    batch_cache.add_request(prefilled, id=1)

    keys = torch.tensor(
        [
            [[[1.0, 2.0]]],
            [[[14.0, 15.0]]],
        ]
    )
    values = keys + 100

    updated_keys, updated_values, seq_len, mask = batch_cache.update_and_fetch(
        keys, values, mask_length=1
    )

    assert seq_len is None
    assert updated_keys.shape == (2, 1, 3, 2)
    assert updated_values.shape == (2, 1, 3, 2)
    assert torch.equal(updated_keys[0, :, :2, :], torch.zeros((1, 2, 2)))
    assert torch.equal(updated_values[0, :, :2, :], torch.zeros((1, 2, 2)))
    assert torch.equal(updated_keys[0, :, 2:, :], keys[0])
    assert torch.equal(updated_values[0, :, 2:, :], values[0])
    assert torch.equal(
        updated_keys[1],
        torch.cat([prefill_key[0], keys[1]], dim=1),
    )
    assert torch.equal(
        updated_values[1],
        torch.cat([prefill_value[0], values[1]], dim=1),
    )
    assert mask.shape == (2, 1, 1, 3)
    assert torch.isneginf(mask[0, 0, 0, :2]).all()
    assert mask[0, 0, 0, 2].item() == 0
    assert torch.equal(mask[1], torch.zeros((1, 1, 3)))
