# TurboQuant Kernel-Optimized Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPU-only, Triton-based "kernel" backend to `TurboQuantMSE`, `TurboQuantProd`, and `PolarQuant`, selectable via a new `backend` constructor parameter, that fuses each algorithm's per-vector operations into single kernel launches and produces results numerically equivalent to the existing "native" backend, at the same or better latency.

**Architecture:** New `turboquant/kernel/` subpackage holds one Triton module per algorithm (`mse.py`, `prod.py`, `polar.py`) plus a shared `_require.py` guard. Each existing algorithm class gains `backend: Literal["native", "kernel"] = "native"`, validated eagerly at construction, dispatching internally to the new kernel module's launch-wrapper functions. `triton` is an optional dependency (`turboquant[kernel]` extra) imported lazily only when `backend="kernel"` is actually used.

**Tech Stack:** Python 3.13, PyTorch (CUDA 13 build), Triton (new optional dependency), pytest, `uv` workspace.

**Spec:** `docs/superpowers/specs/2026-08-28-turboquant-kernel-backend-design.md`

## Global Constraints

- `backend="kernel"` requires `device="cuda"` and a working `triton` import; both failures raise `RuntimeError` with an explicit message — never a silent fallback to native.
- `backend` not in `{"native", "kernel"}` raises `ValueError`.
- Kernel backend outputs must match native backend outputs: `quantize()` index tensors identical (integer equality), `dequantize()`/`inner_product()` outputs within `atol=1e-5`.
- Kernel backend latency must be the same as or better than native backend latency for the same `(algorithm, bits, head_dim)` config on GPU — verified via the extended perf benchmark, not a hardcoded test assertion (GPU-to-GPU speedup ratios aren't portable across hardware).
- No CPU Triton path. No bit-packing changes. No `@triton.autotune`. `PolarQuant`'s kernel path stays sequential across its `log2(d)` levels (matches native's control flow).
- `.quantize()`, `.dequantize()`, `.inner_product()` signatures and return types are unchanged regardless of backend — existing callers (`examples/kv_cache_hook.py`) need no changes.
- `triton` is an optional dependency; importing `turboquant` or using `backend="native"` (the default) must never require `triton` to be installed.
- Existing 55 tests in `turbo-quant/tests/` must continue passing unchanged.

---

## File Structure

```
turbo-quant/
├── pyproject.toml                       # Modify: add `kernel` optional-dependency extra
├── turboquant/
│   ├── kernel/                          # Create: new subpackage
│   │   ├── __init__.py                  # Create
│   │   ├── _require.py                  # Create
│   │   ├── mse.py                       # Create
│   │   ├── prod.py                      # Create
│   │   └── polar.py                     # Create
│   ├── cartesian.py                     # Modify: TurboQuantMSE, TurboQuantProd gain `backend` param
│   └── polar.py                         # Modify: PolarQuant gains `backend` param
├── tests/
│   ├── test_kernel_require.py           # Create: _require.py unit tests (no GPU needed)
│   └── test_kernel_backend.py           # Create: correctness parity tests (skipped without CUDA/Triton)
└── examples/
    └── run_perf_benchmark.py            # Modify: add `backend` sweep dimension
```

---

### Task 1: Kernel backend scaffolding — `_require.py`, package init, optional dependency

**Files:**
- Create: `turbo-quant/turboquant/kernel/__init__.py`
- Create: `turbo-quant/turboquant/kernel/_require.py`
- Modify: `turbo-quant/pyproject.toml`
- Test: `turbo-quant/tests/test_kernel_require.py`

**Interfaces:**
- Produces: `turboquant.kernel._require.require_kernel_backend(device: str) -> None` — raises `RuntimeError` if `device != "cuda"`, or if `import triton` fails. Returns `None` (no exception) otherwise. Later tasks call this from each algorithm class's `__init__` when `backend="kernel"`.

- [ ] **Step 1: Write the failing tests**

Create `turbo-quant/tests/test_kernel_require.py`:

```python
import builtins
import sys

import pytest

from turboquant.kernel._require import require_kernel_backend


def test_require_kernel_backend_raises_on_cpu():
    with pytest.raises(RuntimeError, match="cuda"):
        require_kernel_backend("cpu")


def test_require_kernel_backend_raises_when_triton_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("no triton here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "triton", raising=False)

    with pytest.raises(RuntimeError, match="triton"):
        require_kernel_backend("cuda")


def test_require_kernel_backend_passes_with_cuda_and_triton(monkeypatch):
    import types

    fake_triton = types.ModuleType("triton")
    monkeypatch.setitem(sys.modules, "triton", fake_triton)

    require_kernel_backend("cuda")  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest turbo-quant/tests/test_kernel_require.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'turboquant.kernel'`

- [ ] **Step 3: Implement `_require.py` and the package init**

Create `turbo-quant/turboquant/kernel/__init__.py`:

```python
"""GPU-only, Triton-based kernel backend for turboquant algorithms.

This subpackage is imported lazily by the core algorithm classes only when
``backend="kernel"`` is requested. It has no import-time side effects that
would require ``triton`` to be installed to use the default native backend.
"""
```

Create `turbo-quant/turboquant/kernel/_require.py`:

```python
"""Guard for the kernel backend's environment requirements."""


def require_kernel_backend(device: str) -> None:
    """Raise RuntimeError if the kernel backend cannot run on this device.

    The kernel backend is CUDA-only (Triton targets NVIDIA GPUs) and requires
    the optional `triton` dependency. Both failures are explicit errors, not
    silent fallbacks to the native backend.
    """
    if device != "cuda":
        raise RuntimeError(
            f"kernel backend requires device='cuda', got {device!r}. "
            "The kernel backend does not support CPU."
        )
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "kernel backend requires the 'triton' package, which is not "
            "installed. Install it via the 'kernel' extra: "
            "`uv pip install turboquant[kernel]` or `pip install turboquant[kernel]`."
        ) from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest turbo-quant/tests/test_kernel_require.py -v`
Expected: PASS, 3/3, pristine output

- [ ] **Step 5: Add the `kernel` extra to `pyproject.toml`**

In `turbo-quant/pyproject.toml`, add a new extra alongside the existing `examples` and `test` extras:

```toml
[project.optional-dependencies]
examples = [
    "transformers>=4.40",
    "accelerate>=0.30",
    "datasets>=2.19",
]
test = [
    "pytest>=7.0",
]
kernel = [
    "triton>=3.0; sys_platform == 'linux' or sys_platform == 'win32'",
]
```

(Keep the existing `examples` and `test` blocks exactly as they are — only add the `kernel` block.)

- [ ] **Step 6: Run the full existing test suite to confirm nothing broke**

Run: `uv run pytest turbo-quant/tests -v`
Expected: all previously-passing tests still pass, plus the 3 new ones (58/58 total)

- [ ] **Step 7: Commit**

```bash
git add turbo-quant/turboquant/kernel/__init__.py turbo-quant/turboquant/kernel/_require.py turbo-quant/pyproject.toml turbo-quant/tests/test_kernel_require.py
git commit -m "Add kernel backend scaffolding: require-guard, optional triton dependency"
```

---

### Task 2: MSE kernel backend (`kernel/mse.py`) + `TurboQuantMSE` wiring

**Files:**
- Create: `turbo-quant/turboquant/kernel/mse.py`
- Modify: `turbo-quant/turboquant/cartesian.py` (`TurboQuantMSE` only)
- Test: `turbo-quant/tests/test_kernel_backend.py` (new file, MSE section)

**Interfaces:**
- Consumes: `turboquant.kernel._require.require_kernel_backend(device: str) -> None` (Task 1)
- Produces:
  - `turboquant.kernel.mse.quantize(x: torch.Tensor, rotation: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` — same `(indices, norm)` contract as `Codebook`/`TurboQuantMSE.quantize`, indices are `torch.long`.
  - `turboquant.kernel.mse.dequantize(indices: torch.Tensor, norm: torch.Tensor, rotation: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor` — same contract as `TurboQuantMSE.dequantize`.
  - `TurboQuantMSE(d, bits, seed=0, device=None, backend="native")` — new `backend` param, later consumed by Task 3's `TurboQuantProd` (which constructs an internal `TurboQuantMSE` and must forward its own `backend`).

- [ ] **Step 1: Write the failing correctness-parity test**

Create `turbo-quant/tests/test_kernel_backend.py`:

```python
"""Correctness parity between the native and kernel backends.

Skipped entirely on machines without CUDA or without triton installed --
the kernel backend is CUDA-only by design (see the kernel backend spec).
"""

import pytest
import torch

from turboquant import TurboQuantMSE

CUDA_AND_TRITON_AVAILABLE = torch.cuda.is_available()
if CUDA_AND_TRITON_AVAILABLE:
    try:
        import triton  # noqa: F401
    except ImportError:
        CUDA_AND_TRITON_AVAILABLE = False

requires_kernel_backend = pytest.mark.skipif(
    not CUDA_AND_TRITON_AVAILABLE, reason="kernel backend requires CUDA and triton"
)


@requires_kernel_backend
def test_mse_kernel_quantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = TurboQuantMSE(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantMSE(d, bits, seed=1, device="cuda", backend="kernel")

    native_indices, native_norm = native.quantize(x)
    kernel_indices, kernel_norm = kernel.quantize(x)

    assert torch.equal(native_indices, kernel_indices)
    assert torch.allclose(native_norm, kernel_norm, atol=1e-5)


@requires_kernel_backend
def test_mse_kernel_dequantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = TurboQuantMSE(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantMSE(d, bits, seed=1, device="cuda", backend="kernel")

    indices, norm = native.quantize(x)
    native_x_hat = native.dequantize(indices, norm)
    kernel_x_hat = kernel.dequantize(indices, norm)

    assert torch.allclose(native_x_hat, kernel_x_hat, atol=1e-5)


def test_kernel_backend_rejects_cpu():
    with pytest.raises(RuntimeError, match="cuda"):
        TurboQuantMSE(64, 4, device="cpu", backend="kernel")


def test_invalid_backend_raises_value_error():
    with pytest.raises(ValueError, match="backend"):
        TurboQuantMSE(64, 4, device="cpu", backend="bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest turbo-quant/tests/test_kernel_backend.py -v`
Expected: the two `@requires_kernel_backend` tests are collected (skipped or failing depending on hardware — on this project's RTX 4070 machine they should attempt to run and fail with `TypeError: __init__() got an unexpected keyword argument 'backend'`); the two non-skipped tests fail the same way.

- [ ] **Step 3: Implement `kernel/mse.py`**

Create `turbo-quant/turboquant/kernel/mse.py`:

```python
"""Fused Triton kernels for TurboQuant_mse (Algorithm 1).

Each kernel processes one input vector per program, keeping the rotation
matrix and centroid array resident for the whole kernel so no intermediate
(normalized vector, rotated vector, per-coordinate distance) tensor is ever
written to global memory -- only the final (indices, norm) or x_hat tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mse_quantize_kernel(
    x_ptr, rotation_ptr, centroids_ptr, indices_ptr, norm_ptr,
    stride_x_row, stride_rot_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr, N_CENTROIDS: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x = tl.load(x_ptr + row * stride_x_row + offs_d, mask=mask_d, other=0.0)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    norm_safe = tl.maximum(norm, 1e-12)
    unit = x / norm_safe

    offs_c = tl.arange(0, N_CENTROIDS)
    centroids = tl.load(centroids_ptr + offs_c)

    for j in range(BLOCK_D):
        if j < D:
            rot_row = tl.load(rotation_ptr + j * stride_rot_row + offs_d, mask=mask_d, other=0.0)
            y_j = tl.sum(rot_row * unit, axis=0)
            diffs = tl.abs(y_j - centroids)
            best_idx = tl.argmin(diffs, axis=0)
            tl.store(indices_ptr + row * D + j, best_idx)

    tl.store(norm_ptr + row, norm)


@triton.jit
def _mse_dequantize_kernel(
    indices_ptr, norm_ptr, rotation_ptr, centroids_ptr, out_ptr,
    stride_rot_row, stride_out_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    idx = tl.load(indices_ptr + row * D + offs_d, mask=mask_d, other=0)
    y_hat = tl.load(centroids_ptr + idx, mask=mask_d, other=0.0)
    norm = tl.load(norm_ptr + row)

    for k in range(BLOCK_D):
        if k < D:
            rot_col = tl.load(rotation_ptr + offs_d * stride_rot_row + k, mask=mask_d, other=0.0)
            x_hat_k = tl.sum(rot_col * y_hat, axis=0)
            tl.store(out_ptr + row * stride_out_row + k, x_hat_k * norm)


def quantize(
    x: torch.Tensor, rotation: torch.Tensor, centroids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused normalize + rotate + per-coordinate argmin. Returns (indices, norm)."""
    orig_shape = x.shape
    d = orig_shape[-1]
    x_flat = x.reshape(-1, d).contiguous()
    n = x_flat.shape[0]
    n_centroids = centroids.shape[0]
    block_d = triton.next_power_of_2(d)

    indices = torch.empty((n, d), dtype=torch.int32, device=x.device)
    norm = torch.empty((n,), dtype=x.dtype, device=x.device)

    _mse_quantize_kernel[(n,)](
        x_flat, rotation, centroids, indices, norm,
        x_flat.stride(0), rotation.stride(0),
        D=d, BLOCK_D=block_d, N_CENTROIDS=n_centroids,
    )

    return indices.reshape(*orig_shape).long(), norm.reshape(*orig_shape[:-1])


def dequantize(
    indices: torch.Tensor, norm: torch.Tensor, rotation: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Fused centroid-lookup + unrotate + rescale."""
    orig_shape = indices.shape
    d = orig_shape[-1]
    indices_flat = indices.reshape(-1, d).contiguous().int()
    norm_flat = norm.reshape(-1).contiguous()
    n = indices_flat.shape[0]
    block_d = triton.next_power_of_2(d)

    out = torch.empty((n, d), dtype=centroids.dtype, device=indices.device)

    _mse_dequantize_kernel[(n,)](
        indices_flat, norm_flat, rotation, centroids, out,
        rotation.stride(0), out.stride(0),
        D=d, BLOCK_D=block_d,
    )

    return out.reshape(*orig_shape)
```

- [ ] **Step 4: Wire `backend` into `TurboQuantMSE`**

Modify `turbo-quant/turboquant/cartesian.py`. Add the import at the top (alongside the existing imports):

```python
from .kernel._require import require_kernel_backend
```

Replace the `TurboQuantMSE.__init__` method with:

```python
    def __init__(
        self, d: int, bits: int, seed: int = 0, device: str | None = None, backend: str = "native"
    ):
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if backend not in ("native", "kernel"):
            raise ValueError(f"backend must be 'native' or 'kernel', got {backend!r}")
        if backend == "kernel":
            require_kernel_backend(device)
        self.d = d
        self.bits = bits
        self.device = device
        self.backend = backend
        self.rotation = generate_rotation_matrix(d, seed, device=device)
        self.codebook = Codebook.for_density(beta_coordinate_density(d), bits)
```

Replace `TurboQuantMSE.quantize` and `TurboQuantMSE.dequantize` with:

```python
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: any nonzero vector(s), shape (..., d). Returns (indices, norm)."""
        if self.backend == "kernel":
            from .kernel import mse as kernel_mse

            return kernel_mse.quantize(x, self.rotation, self.codebook.centroids)
        norm = torch.norm(x, dim=-1, keepdim=True)
        unit = x / norm.clamp_min(1e-12)
        y = self.rotate(unit)
        indices = self.codebook.quantize(y)
        return indices, norm.squeeze(-1)

    def dequantize(self, indices: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        if self.backend == "kernel":
            from .kernel import mse as kernel_mse

            return kernel_mse.dequantize(indices, norm, self.rotation, self.codebook.centroids)
        y_hat = self.codebook.dequantize(indices)
        x_hat = self.unrotate(y_hat)
        return x_hat * norm.unsqueeze(-1)
```

Leave `rotate()` and `unrotate()` unchanged — they are still used by the native path.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest turbo-quant/tests/test_kernel_backend.py -v`
Expected: PASS (or SKIPPED if run on a machine without CUDA/triton) for all 4 tests. On this project's RTX 4070 dev machine, all 4 should run and pass — if a Triton compilation or numerical-mismatch error surfaces, debug it here before moving on; do not proceed to Task 3 with a failing MSE kernel.

- [ ] **Step 6: Run the full existing test suite**

Run: `uv run pytest turbo-quant/tests -v`
Expected: all tests pass (58 from Task 1 + the new ones in this task's file, minus double-counting `test_kernel_backend.py`'s 4 tests — full count will be reported by pytest; there must be zero failures and zero unexpected errors)

- [ ] **Step 7: Commit**

```bash
git add turbo-quant/turboquant/kernel/mse.py turbo-quant/turboquant/cartesian.py turbo-quant/tests/test_kernel_backend.py
git commit -m "Add Triton kernel backend for TurboQuantMSE"
```

---

### Task 3: Prod kernel backend (`kernel/prod.py`) + `TurboQuantProd` wiring

**Files:**
- Create: `turbo-quant/turboquant/kernel/prod.py`
- Modify: `turbo-quant/turboquant/cartesian.py` (`TurboQuantProd` only)
- Modify: `turbo-quant/tests/test_kernel_backend.py` (append Prod section)

**Interfaces:**
- Consumes: `turboquant.kernel.mse.quantize`/`.dequantize` (Task 2), `turboquant.kernel._require.require_kernel_backend` (Task 1)
- Produces:
  - `turboquant.kernel.prod.quantize(x, mse_stage, qjl_matrix) -> dict` — same `{"indices", "norm", "qjl_signs", "residual_norm"}` contract as `TurboQuantProd.quantize`. `mse_stage` is a `TurboQuantMSE` instance already constructed with `backend="kernel"`.
  - `turboquant.kernel.prod.dequantize(compressed, mse_stage, qjl_matrix, correction_scale) -> torch.Tensor`
  - `turboquant.kernel.prod.inner_product(y, compressed, mse_stage, qjl_matrix, correction_scale) -> torch.Tensor`

- [ ] **Step 1: Write the failing correctness-parity tests**

Append to `turbo-quant/tests/test_kernel_backend.py` (same `requires_kernel_backend` marker defined earlier in the file):

```python
from turboquant import TurboQuantProd


@requires_kernel_backend
def test_prod_kernel_quantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = TurboQuantProd(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantProd(d, bits, seed=1, device="cuda", backend="kernel")

    native_out = native.quantize(x)
    kernel_out = kernel.quantize(x)

    assert torch.equal(native_out["indices"], kernel_out["indices"])
    assert torch.allclose(native_out["norm"], kernel_out["norm"], atol=1e-5)
    assert torch.equal(native_out["qjl_signs"], kernel_out["qjl_signs"])
    assert torch.allclose(native_out["residual_norm"], kernel_out["residual_norm"], atol=1e-5)


@requires_kernel_backend
def test_prod_kernel_dequantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = TurboQuantProd(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantProd(d, bits, seed=1, device="cuda", backend="kernel")

    compressed = native.quantize(x)
    native_x_hat = native.dequantize(compressed)
    kernel_x_hat = kernel.dequantize(compressed)

    assert torch.allclose(native_x_hat, kernel_x_hat, atol=1e-5)


@requires_kernel_backend
def test_prod_kernel_inner_product_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")
    y = torch.randn(32, d, device="cuda")

    native = TurboQuantProd(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantProd(d, bits, seed=1, device="cuda", backend="kernel")

    compressed = native.quantize(x)
    native_ip = native.inner_product(y, compressed)
    kernel_ip = kernel.inner_product(y, compressed)

    assert torch.allclose(native_ip, kernel_ip, atol=1e-4)


def test_prod_kernel_backend_rejects_cpu():
    with pytest.raises(RuntimeError, match="cuda"):
        TurboQuantProd(64, 4, device="cpu", backend="kernel")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest turbo-quant/tests/test_kernel_backend.py -k prod -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'backend'`

- [ ] **Step 3: Implement `kernel/prod.py`**

Create `turbo-quant/turboquant/kernel/prod.py`:

```python
"""Fused Triton kernels for TurboQuant_prod (Algorithm 2).

Reuses `kernel.mse` for the (bits-1)-bit MSE stage. Adds two fused kernels:
one for the residual -> QJL-projection -> sign-quantize step (used by
`quantize`), and one for the QJL correction -> add-to-x_hat step (used by
`dequantize`). `inner_product` uses a dedicated fused kernel that avoids
materializing the full rotated-y and QJL-projected-y intermediates.
"""

import torch
import triton
import triton.language as tl

from . import mse as kernel_mse


@triton.jit
def _qjl_project_sign_kernel(
    x_ptr, x_hat_ptr, qjl_ptr, signs_ptr, residual_norm_ptr,
    stride_x_row, stride_xhat_row, stride_qjl_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x = tl.load(x_ptr + row * stride_x_row + offs_d, mask=mask_d, other=0.0)
    x_hat = tl.load(x_hat_ptr + row * stride_xhat_row + offs_d, mask=mask_d, other=0.0)
    residual = x - x_hat
    residual_norm = tl.sqrt(tl.sum(residual * residual, axis=0))
    tl.store(residual_norm_ptr + row, residual_norm)

    for j in range(BLOCK_D):
        if j < D:
            qjl_row = tl.load(qjl_ptr + j * stride_qjl_row + offs_d, mask=mask_d, other=0.0)
            proj_j = tl.sum(qjl_row * residual, axis=0)
            sign_j = tl.where(proj_j >= 0, 1.0, -1.0)
            tl.store(signs_ptr + row * D + j, sign_j)


@triton.jit
def _qjl_correct_kernel(
    x_hat_mse_ptr, signs_ptr, residual_norm_ptr, qjl_ptr, out_ptr,
    correction_scale,
    stride_xhat_row, stride_qjl_row, stride_out_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    signs = tl.load(signs_ptr + row * D + offs_d, mask=mask_d, other=0.0)
    residual_norm = tl.load(residual_norm_ptr + row)
    x_hat_mse = tl.load(x_hat_mse_ptr + row * stride_xhat_row + offs_d, mask=mask_d, other=0.0)

    for k in range(BLOCK_D):
        if k < D:
            qjl_col = tl.load(qjl_ptr + offs_d * stride_qjl_row + k, mask=mask_d, other=0.0)
            correction_k = tl.sum(qjl_col * signs, axis=0) * residual_norm * correction_scale
            x_hat_mse_k = tl.sum(tl.where(offs_d == k, x_hat_mse, 0.0), axis=0)
            tl.store(out_ptr + row * stride_out_row + k, x_hat_mse_k + correction_k)


@triton.jit
def _prod_inner_product_kernel(
    y_ptr, indices_ptr, centroids_ptr, norm_ptr,
    qjl_ptr, signs_ptr, residual_norm_ptr,
    rotation_ptr, out_ptr,
    correction_scale,
    stride_y_row, stride_rot_row, stride_qjl_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    y = tl.load(y_ptr + row * stride_y_row + offs_d, mask=mask_d, other=0.0)
    idx = tl.load(indices_ptr + row * D + offs_d, mask=mask_d, other=0)
    y_hat = tl.load(centroids_ptr + idx, mask=mask_d, other=0.0)
    signs = tl.load(signs_ptr + row * D + offs_d, mask=mask_d, other=0.0)
    norm = tl.load(norm_ptr + row)
    residual_norm = tl.load(residual_norm_ptr + row)

    term1_acc = 0.0
    term2_acc = 0.0
    for j in range(BLOCK_D):
        if j < D:
            rot_row = tl.load(rotation_ptr + j * stride_rot_row + offs_d, mask=mask_d, other=0.0)
            rotated_y_j = tl.sum(rot_row * y, axis=0)
            y_hat_j = tl.sum(tl.where(offs_d == j, y_hat, 0.0), axis=0)

            qjl_row = tl.load(qjl_ptr + j * stride_qjl_row + offs_d, mask=mask_d, other=0.0)
            y_proj_j = tl.sum(qjl_row * y, axis=0)
            signs_j = tl.sum(tl.where(offs_d == j, signs, 0.0), axis=0)

            term1_acc += y_hat_j * rotated_y_j
            term2_acc += y_proj_j * signs_j

    out = norm * term1_acc + residual_norm * correction_scale * term2_acc
    tl.store(out_ptr + row, out)


def quantize(x: torch.Tensor, mse_stage, qjl_matrix: torch.Tensor) -> dict:
    """mse_stage: a TurboQuantMSE instance already constructed with backend='kernel'."""
    orig_shape = x.shape
    d = orig_shape[-1]
    x_flat = x.reshape(-1, d).contiguous()
    n = x_flat.shape[0]
    block_d = triton.next_power_of_2(d)

    indices, norm = mse_stage.quantize(x)
    x_hat = mse_stage.dequantize(indices, norm)
    x_hat_flat = x_hat.reshape(-1, d).contiguous()

    signs = torch.empty((n, d), dtype=x.dtype, device=x.device)
    residual_norm = torch.empty((n,), dtype=x.dtype, device=x.device)

    _qjl_project_sign_kernel[(n,)](
        x_flat, x_hat_flat, qjl_matrix, signs, residual_norm,
        x_flat.stride(0), x_hat_flat.stride(0), qjl_matrix.stride(0),
        D=d, BLOCK_D=block_d,
    )

    return {
        "indices": indices,
        "norm": norm,
        "qjl_signs": signs.reshape(*orig_shape),
        "residual_norm": residual_norm.reshape(*orig_shape[:-1]),
    }


def dequantize(compressed: dict, mse_stage, qjl_matrix: torch.Tensor, correction_scale: float) -> torch.Tensor:
    x_hat_mse = mse_stage.dequantize(compressed["indices"], compressed["norm"])
    orig_shape = x_hat_mse.shape
    d = orig_shape[-1]
    x_hat_mse_flat = x_hat_mse.reshape(-1, d).contiguous()
    signs_flat = compressed["qjl_signs"].reshape(-1, d).contiguous()
    residual_norm_flat = compressed["residual_norm"].reshape(-1).contiguous()
    n = x_hat_mse_flat.shape[0]
    block_d = triton.next_power_of_2(d)

    out = torch.empty((n, d), dtype=x_hat_mse.dtype, device=x_hat_mse.device)

    _qjl_correct_kernel[(n,)](
        x_hat_mse_flat, signs_flat, residual_norm_flat, qjl_matrix, out,
        correction_scale,
        x_hat_mse_flat.stride(0), qjl_matrix.stride(0), out.stride(0),
        D=d, BLOCK_D=block_d,
    )

    return out.reshape(*orig_shape)


def inner_product(
    y: torch.Tensor, compressed: dict, mse_stage, qjl_matrix: torch.Tensor, correction_scale: float
) -> torch.Tensor:
    orig_shape = y.shape[:-1]
    d = y.shape[-1]
    y_flat = y.reshape(-1, d).contiguous()
    indices_flat = compressed["indices"].reshape(-1, d).contiguous().int()
    norm_flat = compressed["norm"].reshape(-1).contiguous()
    signs_flat = compressed["qjl_signs"].reshape(-1, d).contiguous()
    residual_norm_flat = compressed["residual_norm"].reshape(-1).contiguous()
    n = y_flat.shape[0]
    block_d = triton.next_power_of_2(d)

    out = torch.empty((n,), dtype=y.dtype, device=y.device)

    _prod_inner_product_kernel[(n,)](
        y_flat, indices_flat, mse_stage.codebook.centroids, norm_flat,
        qjl_matrix, signs_flat, residual_norm_flat,
        mse_stage.rotation, out,
        correction_scale,
        y_flat.stride(0), mse_stage.rotation.stride(0), qjl_matrix.stride(0),
        D=d, BLOCK_D=block_d,
    )

    return out.reshape(*orig_shape)
```

- [ ] **Step 4: Wire `backend` into `TurboQuantProd`**

Modify `turbo-quant/turboquant/cartesian.py`. Replace `TurboQuantProd.__init__`:

```python
    def __init__(
        self, d: int, bits: int, seed: int = 0, device: str | None = None, backend: str = "native"
    ):
        if bits < 2:
            raise ValueError(f"bits must be >= 2 for TurboQuantProd, got {bits}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if backend not in ("native", "kernel"):
            raise ValueError(f"backend must be 'native' or 'kernel', got {backend!r}")
        if backend == "kernel":
            require_kernel_backend(device)
        self.d = d
        self.bits = bits
        self.device = device
        self.backend = backend
        self.mse = TurboQuantMSE(d, bits - 1, seed=seed, device=device, backend=backend)
        self.qjl_matrix = generate_qjl_matrix(d, seed=seed + 1, device=device)
        self._correction_scale = math.sqrt(math.pi / 2) / self.d
```

Replace `TurboQuantProd.quantize`, `TurboQuantProd.dequantize`, and `TurboQuantProd.inner_product`:

```python
    def quantize(self, x: torch.Tensor) -> dict:
        if self.backend == "kernel":
            from .kernel import prod as kernel_prod

            return kernel_prod.quantize(x, self.mse, self.qjl_matrix)

        indices, norm = self.mse.quantize(x)
        x_hat = self.mse.dequantize(indices, norm)
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)

        projected = residual @ self.qjl_matrix.T
        qjl_signs = sign_quantize(projected)

        return {
            "indices": indices,
            "norm": norm,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }

    def _qjl_correction(self, compressed: dict) -> torch.Tensor:
        return (
            compressed["residual_norm"].unsqueeze(-1)
            * self._correction_scale
            * (compressed["qjl_signs"] @ self.qjl_matrix)
        )

    def dequantize(self, compressed: dict) -> torch.Tensor:
        if self.backend == "kernel":
            from .kernel import prod as kernel_prod

            return kernel_prod.dequantize(compressed, self.mse, self.qjl_matrix, self._correction_scale)

        x_hat_mse = self.mse.dequantize(compressed["indices"], compressed["norm"])
        return x_hat_mse + self._qjl_correction(compressed)

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Unbiased estimate of <x, y> using compressed x (Algorithm 2's payoff)."""
        if self.backend == "kernel":
            from .kernel import prod as kernel_prod

            return kernel_prod.inner_product(y, compressed, self.mse, self.qjl_matrix, self._correction_scale)

        x_hat_mse = self.mse.dequantize(compressed["indices"], compressed["norm"])
        term1 = (y * x_hat_mse).sum(dim=-1)

        y_projected = y @ self.qjl_matrix.T
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)
        term2 = compressed["residual_norm"] * self._correction_scale * qjl_ip

        return term1 + term2
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest turbo-quant/tests/test_kernel_backend.py -k prod -v`
Expected: PASS (or SKIPPED without CUDA/triton); on the project's RTX 4070 dev machine, all 4 should run and pass. Debug any Triton shape/stride or numerical mismatch here before proceeding.

- [ ] **Step 6: Run the full existing test suite**

Run: `uv run pytest turbo-quant/tests -v`
Expected: zero failures, zero unexpected errors

- [ ] **Step 7: Commit**

```bash
git add turbo-quant/turboquant/kernel/prod.py turbo-quant/turboquant/cartesian.py turbo-quant/tests/test_kernel_backend.py
git commit -m "Add Triton kernel backend for TurboQuantProd"
```

---

### Task 4: Polar kernel backend (`kernel/polar.py`) + `PolarQuant` wiring

**Files:**
- Create: `turbo-quant/turboquant/kernel/polar.py`
- Modify: `turbo-quant/turboquant/polar.py`
- Modify: `turbo-quant/tests/test_kernel_backend.py` (append Polar section)

**Interfaces:**
- Consumes: `turboquant.kernel._require.require_kernel_backend` (Task 1)
- Produces:
  - `turboquant.kernel.polar.quantize_level(v: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` — returns `(angle_indices, radius)` for one recursion level, `v` shape `(..., L)`, outputs shape `(..., L // 2)`.
  - `turboquant.kernel.polar.dequantize_level(angle_indices: torch.Tensor, radius: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor` — returns `v` of shape `(..., 2 * L_half)`, the inverse of `quantize_level`.

- [ ] **Step 1: Write the failing correctness-parity tests**

Append to `turbo-quant/tests/test_kernel_backend.py`:

```python
from turboquant import PolarQuant


@requires_kernel_backend
def test_polar_kernel_quantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = PolarQuant(d, bits, seed=1, device="cuda", backend="native")
    kernel = PolarQuant(d, bits, seed=1, device="cuda", backend="kernel")

    native_out = native.quantize(x)
    kernel_out = kernel.quantize(x)

    assert len(native_out["angle_indices"]) == len(kernel_out["angle_indices"])
    for native_level, kernel_level in zip(native_out["angle_indices"], kernel_out["angle_indices"]):
        assert torch.equal(native_level, kernel_level)
    assert torch.allclose(native_out["final_radius"], kernel_out["final_radius"], atol=1e-5)


@requires_kernel_backend
def test_polar_kernel_dequantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = PolarQuant(d, bits, seed=1, device="cuda", backend="native")
    kernel = PolarQuant(d, bits, seed=1, device="cuda", backend="kernel")

    compressed = native.quantize(x)
    native_x_hat = native.dequantize(compressed)
    kernel_x_hat = kernel.dequantize(compressed)

    assert torch.allclose(native_x_hat, kernel_x_hat, atol=1e-4)


def test_polar_kernel_backend_rejects_cpu():
    with pytest.raises(RuntimeError, match="cuda"):
        PolarQuant(64, 4, device="cpu", backend="kernel")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest turbo-quant/tests/test_kernel_backend.py -k polar -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'backend'`

- [ ] **Step 3: Implement `kernel/polar.py`**

Create `turbo-quant/turboquant/kernel/polar.py`:

```python
"""Fused Triton kernels for PolarQuant, one fused kernel per recursion level.

Each level pairs up coordinates, computes (radius, angle), and quantizes the
angle against that level's Lloyd-Max codebook in a single kernel -- matching
native's per-level computation but without materializing the intermediate
angle/radius tensors before quantization. Levels remain sequential (each
depends on the previous level's radii), matching native's control flow.
"""

import math

import torch
import triton
import triton.language as tl

_TWO_PI = 2.0 * math.pi


@triton.jit
def _polar_quantize_level_kernel(
    v_ptr, centroids_ptr, angle_idx_ptr, radius_ptr,
    stride_v_row, stride_out_row,
    HALF: tl.constexpr, BLOCK_HALF: tl.constexpr, N_CENTROIDS: tl.constexpr, TWO_PI: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF

    v0 = tl.load(v_ptr + row * stride_v_row + 2 * offs, mask=mask, other=0.0)
    v1 = tl.load(v_ptr + row * stride_v_row + 2 * offs + 1, mask=mask, other=0.0)

    radius = tl.sqrt(v0 * v0 + v1 * v1)
    angle = tl.math.atan2(v1, v0)
    angle = tl.where(angle < 0, angle + TWO_PI, angle)

    offs_c = tl.arange(0, N_CENTROIDS)
    centroids = tl.load(centroids_ptr + offs_c)

    diffs = tl.abs(angle[:, None] - centroids[None, :])
    idx = tl.argmin(diffs, axis=1)

    tl.store(angle_idx_ptr + row * stride_out_row + offs, idx, mask=mask)
    tl.store(radius_ptr + row * stride_out_row + offs, radius, mask=mask)


@triton.jit
def _polar_dequantize_level_kernel(
    angle_idx_ptr, radius_ptr, centroids_ptr, v_out_ptr,
    stride_in_row, stride_out_row,
    HALF: tl.constexpr, BLOCK_HALF: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF

    idx = tl.load(angle_idx_ptr + row * stride_in_row + offs, mask=mask, other=0)
    angle = tl.load(centroids_ptr + idx, mask=mask, other=0.0)
    radius = tl.load(radius_ptr + row * stride_in_row + offs, mask=mask, other=0.0)

    v0 = radius * tl.math.cos(angle)
    v1 = radius * tl.math.sin(angle)

    tl.store(v_out_ptr + row * stride_out_row + 2 * offs, v0, mask=mask)
    tl.store(v_out_ptr + row * stride_out_row + 2 * offs + 1, v1, mask=mask)


def quantize_level(v: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """v: (..., L). Returns (angle_indices, radius), each (..., L // 2)."""
    orig_shape = v.shape
    length = orig_shape[-1]
    half = length // 2
    v_flat = v.reshape(-1, length).contiguous()
    n = v_flat.shape[0]
    n_centroids = centroids.shape[0]
    block_half = triton.next_power_of_2(half)

    angle_idx = torch.empty((n, half), dtype=torch.int32, device=v.device)
    radius = torch.empty((n, half), dtype=v.dtype, device=v.device)

    _polar_quantize_level_kernel[(n,)](
        v_flat, centroids, angle_idx, radius,
        v_flat.stride(0), angle_idx.stride(0),
        HALF=half, BLOCK_HALF=block_half, N_CENTROIDS=n_centroids, TWO_PI=_TWO_PI,
    )

    out_shape = (*orig_shape[:-1], half)
    return angle_idx.reshape(out_shape).long(), radius.reshape(out_shape)


def dequantize_level(angle_indices: torch.Tensor, radius: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_level. angle_indices, radius: (..., L // 2). Returns (..., L)."""
    orig_shape = angle_indices.shape
    half = orig_shape[-1]
    idx_flat = angle_indices.reshape(-1, half).contiguous().int()
    radius_flat = radius.reshape(-1, half).contiguous()
    n = idx_flat.shape[0]
    block_half = triton.next_power_of_2(half)

    out = torch.empty((n, 2 * half), dtype=radius.dtype, device=radius.device)

    _polar_dequantize_level_kernel[(n,)](
        idx_flat, radius_flat, centroids, out,
        idx_flat.stride(0), out.stride(0),
        HALF=half, BLOCK_HALF=block_half,
    )

    return out.reshape(*orig_shape[:-1], 2 * half)
```

- [ ] **Step 4: Wire `backend` into `PolarQuant`**

First read `turbo-quant/turboquant/polar.py` in full to see the exact current `__init__`, `quantize`, and `dequantize` bodies (the recursion structure, how `self.codebooks` per level are stored, and the exact key names in the returned dict) — the redesign plan's Task 8 defined these but this plan does not reproduce that file's full current text. Then apply the same pattern used in Tasks 2 and 3:

- Add `backend: str = "native"` to `__init__`'s signature, with the same three validation lines (`backend not in (...)` → `ValueError`, `backend == "kernel"` → `require_kernel_backend(device)`) used in `TurboQuantMSE.__init__`, and `self.backend = backend`.
- In `quantize`, if `self.backend == "kernel"`, replace the per-level `self._decompose(...)` + `self.codebooks[level].quantize(...)` calls with `from .kernel import polar as kernel_polar` and a loop calling `kernel_polar.quantize_level(v, self.codebooks[level].centroids)` for each level, accumulating `angle_indices` the same way the native path does, keeping the final un-quantized radius (`final_radius`) handling identical to native (the last level's `radius` output, not quantized).
- In `dequantize`, if `self.backend == "kernel"`, replace the per-level reconstruction with `kernel_polar.dequantize_level(angle_indices[level], radius, self.codebooks[level].centroids)` called in `reversed(range(n_levels))`, exactly mirroring native's existing reconstruction order.
- Add the same `from .kernel._require import require_kernel_backend` import used in `cartesian.py`.

Keep every other method and the native code paths byte-for-byte unchanged — this task only adds the `backend` branch, mirroring the shape of Tasks 2 and 3's edits to `cartesian.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest turbo-quant/tests/test_kernel_backend.py -k polar -v`
Expected: PASS (or SKIPPED without CUDA/triton); on the project's RTX 4070 dev machine, all 3 should run and pass. Debug any level-indexing, stride, or `atan2`/`cos`/`sin` numerical mismatch here before proceeding — this is the most structurally different kernel of the three, most likely to need iteration.

- [ ] **Step 6: Run the full existing test suite**

Run: `uv run pytest turbo-quant/tests -v`
Expected: zero failures, zero unexpected errors

- [ ] **Step 7: Commit**

```bash
git add turbo-quant/turboquant/kernel/polar.py turbo-quant/turboquant/polar.py turbo-quant/tests/test_kernel_backend.py
git commit -m "Add Triton kernel backend for PolarQuant"
```

---

### Task 5: Extend `run_perf_benchmark.py` with a `backend` sweep and run it for real

**Files:**
- Modify: `turbo-quant/examples/run_perf_benchmark.py`

**Interfaces:**
- Consumes: `backend` param on `TurboQuantMSE`, `TurboQuantProd`, `PolarQuant` (Tasks 2-4)
- Produces: CSV rows with a new `backend` column, written via the existing `results_logger.write_csv`/`default_output_path` (no changes needed to `results_logger.py`)

- [ ] **Step 1: Read the current benchmark script**

Read `turbo-quant/examples/run_perf_benchmark.py` in full before editing — this plan does not reproduce its current body. Identify: the sweep loop structure (device × algorithm × bits × head_dim), the `torch.cuda.synchronize()` bracketing around each timed block, and the CSV row dict's exact field names (per the earlier redesign work's schema: `device, algorithm, bits, head_dim, batch_size, quantize_latency_ms_mean, quantize_latency_ms_min, quantize_throughput_vecs_per_sec, dequantize_latency_ms_mean, dequantize_latency_ms_min, dequantize_throughput_vecs_per_sec`).

- [ ] **Step 2: Add `backend` as a sweep dimension**

Edit the sweep so that for every `device == "cuda"` configuration, the benchmark runs twice — once with `backend="native"` and once with `backend="kernel"` — while `device == "cpu"` configurations run only `backend="native"` (kernel backend is CUDA-only; do not attempt to construct a `backend="kernel"` instance on CPU, that would raise `RuntimeError` per Task 1's guard). Add a `"backend": backend` key to every row dict written to the CSV, positioned right after `"device"` to match the existing column ordering convention in that file.

Concretely, wrap the existing per-`device` sweep body so the class-construction lines change from e.g.:

```python
quantizer = cls(head_dim, bits, seed=0, device=device)
```

to:

```python
backends = ["native", "kernel"] if device == "cuda" else ["native"]
for backend in backends:
    quantizer = cls(head_dim, bits, seed=0, device=device, backend=backend)
    ...  # existing timing logic unchanged
    row["backend"] = backend
    rows.append(row)
```

Preserve every existing timing detail (warm-up pass, `torch.cuda.synchronize()` before and after each timed block, `time.perf_counter()`, mean/min latency, throughput calculation) exactly as it is today — this task only adds the backend dimension around the existing, already-correct timing logic.

- [ ] **Step 3: Run the smoke test**

Run: `uv run python turbo-quant/examples/run_perf_benchmark.py --smoke-test` (or the script's equivalent quick-config flag — check the current CLI args in the file read in Step 1). Confirm the output CSV under `turbo-quant/examples/results/` has both `backend=native` and `backend=kernel` rows for every CUDA config, and only `backend=native` rows for CPU configs.

- [ ] **Step 4: Run the full real sweep on the project's GPU**

Run: `uv run python turbo-quant/examples/run_perf_benchmark.py` (full sweep, no smoke-test flag). This produces the actual "same or better" evidence required by the spec.

- [ ] **Step 5: Verify the "same or better" requirement against the real CSV**

Read the resulting CSV. For every `(algorithm, bits, head_dim)` row pair on `device="cuda"`, confirm `quantize_latency_ms_mean` and `dequantize_latency_ms_mean` for `backend="kernel"` are ≤ the corresponding `backend="native"` values (a small tolerance, e.g. within 5%, is acceptable noise — report exact numbers either way). If any configuration comes back meaningfully slower under `backend="kernel"`, do not hide it: report the specific config and either fix the kernel (e.g. `PolarQuant`'s per-level launch overhead is the likeliest culprit at small `d`, per the spec's stated risk) or bring it back to the user as an explicit finding before considering this task done.

- [ ] **Step 6: Commit**

```bash
git add turbo-quant/examples/run_perf_benchmark.py
git commit -m "Add kernel-vs-native backend sweep to run_perf_benchmark.py"
```

(Do not commit the generated `results/*.csv` files — `turbo-quant/examples/results/` is already gitignored per the earlier redesign work.)

---

## Final Verification

After all 5 tasks:

- [ ] `uv run pytest turbo-quant/tests -v` — full suite passes, zero failures, zero unexpected errors or warnings
- [ ] `uv run python turbo-quant/examples/run_perf_benchmark.py --smoke-test` — runs cleanly end to end
- [ ] The real (non-smoke-test) perf benchmark CSV from Task 5 Step 4 shows kernel backend matching or beating native backend latency for every CUDA configuration, or every exception is explicitly documented
- [ ] Every new public function (`turboquant.kernel.mse.quantize`/`dequantize`, `turboquant.kernel.prod.quantize`/`dequantize`/`inner_product`, `turboquant.kernel.polar.quantize_level`/`dequantize_level`, `require_kernel_backend`) has at least one direct or indirect test covering it
- [ ] `backend="native"` remains every class's default; no existing caller (`examples/kv_cache_hook.py`, `examples/run_benchmark.py`, `examples/run_experiments.py`) needed any changes
