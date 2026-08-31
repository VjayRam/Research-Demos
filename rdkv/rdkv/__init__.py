"""Paper-accurate implementation of RDKV rate-distortion KV cache bit
allocation (Rate-Distortion Bit Allocation for Joint Eviction and
Quantization of the KV Cache, arXiv:2605.08317)."""

from .mckp import bennett_distortion, mckp_bisect
from .pipeline import AllocationResult, RDKVAllocator
from .waterfilling import continuous_waterfill
from .weights import bennett_sigma, channel_weight_k, token_weight_v

__all__ = [
    "AllocationResult",
    "RDKVAllocator",
    "bennett_distortion",
    "bennett_sigma",
    "channel_weight_k",
    "continuous_waterfill",
    "mckp_bisect",
    "token_weight_v",
]
