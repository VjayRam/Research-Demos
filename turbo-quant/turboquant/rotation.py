"""Haar-random orthogonal rotation via QR decomposition (paper's exact setup step)."""

import torch

_rotation_cache: dict[tuple[int, int, str], torch.Tensor] = {}


def generate_rotation_matrix(d: int, seed: int, device: str | None = None) -> torch.Tensor:
    """Haar-distributed random orthogonal d x d matrix, built by QR-decomposing
    a random Gaussian matrix and fixing the sign ambiguity in Q.

    Cached per (d, seed, device): this is the paper's "setup, once per (d,b)" step,
    data-independent and reused across every call at that (d, seed, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = (d, seed, device)
    if key in _rotation_cache:
        return _rotation_cache[key]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    g = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(g)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = (q * diag_sign.unsqueeze(0)).to(torch.float32).to(device)

    _rotation_cache[key] = q
    return q
