"""Round-trips a HF model's KV cache through a turboquant quantizer.

This is a quality/correctness test harness, not a production compressed
cache: every new key/value vector is immediately quantized and dequantized
before being stored, so subsequent attention runs on the reconstructed
tensors and generation quality can be measured under a given
(algorithm, bits) setting.
"""

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache


class QuantizingCache(DynamicCache):
    """A DynamicCache that round-trips every key/value vector through a quantizer.

    key_quantizer / value_quantizer: any of turboquant's TurboQuantMSE,
    TurboQuantProd, or PolarQuant instances, sized to the model's head_dim.
    """

    def __init__(self, key_quantizer, value_quantizer):
        super().__init__()
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer

    @staticmethod
    def _round_trip(quantizer, states: torch.Tensor) -> torch.Tensor:
        b, h, s, d = states.shape
        flat = states.reshape(b * h * s, d).float()
        compressed = quantizer.quantize(flat)
        if isinstance(compressed, tuple):
            reconstructed = quantizer.dequantize(*compressed)
        else:
            reconstructed = quantizer.dequantize(compressed)
        return reconstructed.reshape(b, h, s, d).to(states.dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ):
        key_states = self._round_trip(self.key_quantizer, key_states)
        value_states = self._round_trip(self.value_quantizer, value_states)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
