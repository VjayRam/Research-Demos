"""Quantized Johnson-Lindenstrauss sign projection (Algorithm 2's residual stage)."""

import torch

_qjl_cache: dict[tuple[int, int], torch.Tensor] = {}


def generate_qjl_matrix(d: int, seed: int) -> torch.Tensor:
    """S with i.i.d. N(0,1) entries, d x d.

    TurboQuant applies QJL as one sign bit per residual coordinate (not a
    dimensionality-reducing projection), so S is always square.
    """
    key = (d, seed)
    if key in _qjl_cache:
        return _qjl_cache[key]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    s = torch.randn(d, d, generator=gen)
    _qjl_cache[key] = s
    return s


def sign_quantize(x: torch.Tensor) -> torch.Tensor:
    """sign(x), with the zero -> +1 tie-break (a floating-point tie-break,
    not an algorithmic choice -- exact zero has probability 0 under any
    continuous projection anyway)."""
    signs = torch.sign(x)
    signs[signs == 0] = 1.0
    return signs
