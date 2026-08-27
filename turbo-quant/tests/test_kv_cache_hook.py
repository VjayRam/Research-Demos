import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from kv_cache_hook import QuantizingCache
from turboquant import TurboQuantMSE, TurboQuantProd


def test_update_round_trips_through_quantizer_and_matches_shape():
    b, h, s, d = 1, 2, 3, 16
    key_q = TurboQuantMSE(d=d, bits=3, seed=0)
    val_q = TurboQuantMSE(d=d, bits=3, seed=1)
    cache = QuantizingCache(key_quantizer=key_q, value_quantizer=val_q)

    keys = torch.randn(b, h, s, d)
    values = torch.randn(b, h, s, d)
    cached_keys, cached_values = cache.update(keys, values, layer_idx=0)

    assert cached_keys.shape == keys.shape
    assert cached_values.shape == values.shape
    # Round-tripping through a lossy quantizer must change the values.
    assert not torch.allclose(cached_keys, keys)


def test_update_round_trips_with_dict_returning_quantizer():
    b, h, s, d = 1, 2, 3, 16
    key_q = TurboQuantProd(d=d, bits=2, seed=0)
    val_q = TurboQuantProd(d=d, bits=2, seed=1)
    cache = QuantizingCache(key_quantizer=key_q, value_quantizer=val_q)

    keys = torch.randn(b, h, s, d)
    values = torch.randn(b, h, s, d)
    cached_keys, cached_values = cache.update(keys, values, layer_idx=0)

    assert cached_keys.shape == keys.shape
    assert cached_values.shape == values.shape
    # Round-tripping through a lossy quantizer must change the values.
    assert not torch.allclose(cached_keys, keys)


def test_update_appends_across_calls():
    b, h, d = 1, 2, 8
    key_q = TurboQuantMSE(d=d, bits=2, seed=0)
    val_q = TurboQuantMSE(d=d, bits=2, seed=1)
    cache = QuantizingCache(key_quantizer=key_q, value_quantizer=val_q)

    first_keys = torch.randn(b, h, 3, d)
    first_values = torch.randn(b, h, 3, d)
    cache.update(first_keys, first_values, layer_idx=0)

    next_keys = torch.randn(b, h, 1, d)
    next_values = torch.randn(b, h, 1, d)
    cached_keys, cached_values = cache.update(next_keys, next_values, layer_idx=0)

    assert cached_keys.shape == (b, h, 4, d)
    assert cached_values.shape == (b, h, 4, d)
