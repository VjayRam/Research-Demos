"""Shared-memory capacity check for the Triton kernel backend.

Each kernel here is a "load one or two D x D matrix tiles plus a BLOCK_M x D
operand tile into shared memory, do the whole computation resident" design
(see kernel/mse.py's and kernel/prod.py's module docstrings for why). That
means the shared-memory footprint is fully determined by (BLOCK_M, D,
n_resident_matrices, dtype itemsize) -- no profiling needed, just arithmetic,
using float32 (4 bytes) since that's what these kernels operate in.

`shared_memory_per_block_optin` (not the smaller default
`shared_memory_per_block`) is the correct comparison figure: these kernels
are compiled to opt into the higher per-block limit, and using the
conservative default would reject configs that actually run (e.g. this
package's own d=128 TurboQuantProd kernels, verified fitting under
`shared_memory_per_block_optin` but not under `shared_memory_per_block`).
"""

import torch

_DTYPE_ITEMSIZE = 4  # float32


def estimate_shared_memory_bytes(d: int, block_m: int, n_resident_matrices: int) -> int:
    """One BLOCK_M x D operand tile plus n_resident_matrices D x D matrix tiles."""
    operand_tile = block_m * d * _DTYPE_ITEMSIZE
    matrix_tiles = n_resident_matrices * d * d * _DTYPE_ITEMSIZE
    return operand_tile + matrix_tiles


def device_shared_memory_limit(device: str) -> int:
    """The per-block shared memory budget Triton kernels here can opt into."""
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    return torch.cuda.get_device_properties(index).shared_memory_per_block_optin


def fits_in_shared_memory(device: str, d: int, block_m: int, n_resident_matrices: int) -> bool:
    required = estimate_shared_memory_bytes(d, block_m, n_resident_matrices)
    return required <= device_shared_memory_limit(device)
