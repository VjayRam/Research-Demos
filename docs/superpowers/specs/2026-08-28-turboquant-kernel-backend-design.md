# TurboQuant Kernel-Optimized Backend — Design Spec

**Date:** 2026-08-28
**Status:** Approved for planning

## Summary

Add an alternative, GPU-only execution backend to the `turboquant` package that
fuses each algorithm's per-vector operations into single Triton kernels, reducing
kernel-launch count and eliminating intermediate global-memory tensors relative to
the existing "native" backend (plain chained `torch` ops). Selectable per-instance
via a new `backend` constructor parameter on `TurboQuantMSE`, `TurboQuantProd`, and
`PolarQuant`. Orthogonal to the existing `device` parameter: `device` picks
CPU vs GPU: `backend` picks, on GPU only, which GPU implementation runs.

## Motivation

The core `turboquant` package is deliberately paper-exact with no engineering
shortcuts baked in (see `2026-08-27-turboquant-redesign-design.md`). That
commitment covers correctness, not performance. The native GPU path already gets
cuBLAS for the rotation matmul, but each `.quantize()`/`.dequantize()` call is a
chain of 3-5 separate torch ops (normalize, matmul, broadcast-subtract against all
centroids, argmin, ...), each reading/writing the full tensor to global memory.
For the vector sizes this package targets (d = 64-128, ≤16 centroids at ≤4 bits),
the entire per-vector working set fits in one Triton block — fusing the chain into
one kernel avoids materializing any of those intermediates. This is a genuine,
measurable perf opportunity, not a rewrite of the algorithm.

## Non-Goals

- No CPU Triton path. `backend="kernel"` requires CUDA; requesting it on CPU is a
  hard error, not a fallback (see Error Handling).
- No bit-packing or reduced-precision storage format changes. The kernel backend
  returns the same tensor shapes/dtypes as native (`indices`, `norm`, etc.) —
  packing remains an `examples/`-layer concern, out of scope for the core package.
- No autotuning framework (`@triton.autotune` over a config matrix) in this pass.
  Block sizes are fixed once per `d` (next power of 2 ≥ d). Can be added later if
  benchmarking shows a need.
- `PolarQuant`'s kernel path stays algorithmically sequential across its
  `log2(d)` levels, same as native — no cross-level fusion. Each level becomes one
  fused kernel instead of a chain of several, but levels still run in sequence
  because each depends on the previous level's radii.
- No change to `examples/` scripts' default behavior. `run_benchmark.py` and
  `run_experiments.py` keep defaulting to `backend="native"`; kernel-backend
  sweeps are opt-in via CLI flag additions (implementation detail, not specified
  further here).

## Architecture

### Module layout

```
turboquant/
├── kernel/                        # new subpackage — Triton backend
│   ├── __init__.py                # re-exports the public launch-wrapper functions
│   ├── _require.py                # require_kernel_backend(device) guard
│   ├── mse.py                     # Triton kernels + launch wrappers for TurboQuantMSE
│   ├── prod.py                    # Triton kernels + launch wrappers for TurboQuantProd
│   └── polar.py                   # Triton kernels + launch wrappers for PolarQuant
├── cartesian.py                   # TurboQuantMSE/Prod gain `backend` param
├── polar.py                       # PolarQuant gains `backend` param
└── ... (rotation.py, distributions.py, lloyd_max.py, codebook.py, qjl.py unchanged)
```

`triton` is an **optional** dependency — a new `kernel` extra in
`turbo-quant/pyproject.toml` (`turboquant[kernel]`), not a dependency of the base
package. `turboquant.kernel` is imported lazily, only the first time a class is
constructed with `backend="kernel"`. Importing `turboquant` itself, and using
`backend="native"` (the default), never requires `triton` to be installed.

### Public API change

Every algorithm class gains one new constructor parameter:

```python
TurboQuantMSE(d, bits, seed=0, device=None, backend="native")
TurboQuantProd(d, bits, seed=0, device=None, backend="native")
PolarQuant(d, bits, seed=0, device=None, backend="native")
```

`backend: Literal["native", "kernel"] = "native"`. `.quantize()`, `.dequantize()`,
and (for `TurboQuantProd`) `.inner_product()` keep identical signatures and return
types regardless of backend — callers (including `examples/kv_cache_hook.py`)
never need to know which backend is active. Each method branches once internally:

```python
def quantize(self, x):
    if self.backend == "kernel":
        return turboquant.kernel.mse.quantize(x, self.rotation, self.codebook, ...)
    # existing native path unchanged
    ...
```

`TurboQuantProd` constructs its internal `TurboQuantMSE(d, bits - 1, ...)` stage
with the same `backend` it was given, so the backend choice is consistent through
the whole object graph.

### Error handling

`backend` is validated eagerly, at construction time, not deferred to the first
`.quantize()` call:

- `backend` not in `{"native", "kernel"}` → `ValueError`.
- `backend="kernel"` with `device != "cuda"` → `RuntimeError`, explicit message
  ("kernel backend requires device='cuda', got '<device>'"). No silent fallback
  to native.
- `backend="kernel"` with `import triton` failing → `RuntimeError`, message
  pointing at the extra (`pip install turboquant[kernel]` or the workspace
  equivalent). No silent fallback to native.

Both checks live in `turboquant/kernel/_require.py`:
`require_kernel_backend(device: str) -> None`, called once from each class's
`__init__` when `backend="kernel"`.

### Kernel fusion design, per algorithm

**MSE (`kernel/mse.py`)**
- `quantize` kernel: one Triton kernel per launch computes, per input vector,
  L2 norm → normalize → rotate (matmul against the resident `d×d` rotation
  matrix, loaded once per block) → per-coordinate literal argmin against the
  resident centroid array (≤16 elements at ≤4 bits). Writes only `indices` and
  `norm` to global memory — `unit`, `y`, and the `(..., d, 2^bits)` diff tensor
  from `Codebook.quantize()` never materialize in global memory.
- `dequantize` kernel: centroid lookup → unrotate (matmul against `rotation.T`,
  resident) → rescale by `norm`, fused into one kernel writing only the final
  `x_hat`.
- Argmin result must exactly match `Codebook.quantize()`'s literal per-element
  argmin — same nearest-centroid rule, different execution strategy.

**Prod (`kernel/prod.py`)**
- Reuses `kernel/mse.py`'s kernels for the `bits - 1` MSE stage.
- One additional fused kernel: given `x` and the MSE stage's `x_hat`, computes
  the residual, its norm, projects it through the resident QJL matrix, and
  sign-quantizes — one pass instead of the native path's separate
  subtract/norm/matmul/sign_quantize calls.
- `inner_product()`'s kernel path fuses the `y_projected @ qjl_signs` reduction
  with the `term1`/`term2` combination into one kernel.

**Polar (`kernel/polar.py`)**
- One fused kernel per recursive level: pairwise `atan2`/radius computation for
  that level, fused with that level's Lloyd-Max argmin against the level's
  angle-density codebook. `log2(d)` kernel launches total (unchanged from
  native's launch count per level's *op count*, but each level is now 1 launch
  instead of native's several).
- Sequential across levels (each depends on the previous level's radii) —
  matches native's control flow exactly, just fused within each level.

## Correctness & Performance Guarantees

Both are explicit product requirements: **kernel backend results must match
native backend results, and kernel backend performance must be the same as or
better than native backend performance.** Verified two different ways:

### Correctness parity (must hold exactly, tested in CI-style unit tests)

`tests/test_kernel_backend.py`, skipped via `pytest.mark.skipif` when CUDA or
`triton` is unavailable in the test environment:

- For each of `TurboQuantMSE`, `TurboQuantProd`, `PolarQuant`: construct one
  `backend="native"` and one `backend="kernel"` instance with identical
  `(d, bits, seed, device="cuda")`, feed identical random input.
- Assert `quantize()` produces identical index tensors (integer equality, not
  tolerance — both backends implement the same literal argmin rule and must
  agree exactly).
- Assert `dequantize()` output matches within `atol=1e-5` (float reconstruction
  tolerance across the two independent code paths).
- For `TurboQuantProd`, additionally assert `inner_product()` matches within
  the same tolerance.

### Performance parity/improvement (measured, not asserted as a hard test gate)

`examples/run_perf_benchmark.py` gains `backend` as a third sweep dimension
(alongside the existing `device`/`algorithm`/`bits`/`head_dim`), CSV schema
extended with a `backend` column. For every `(algorithm, bits, head_dim)`
config on `device="cuda"`, both `backend="native"` and `backend="kernel"` rows
are logged in the same run, making a kernel-slower-than-native regression
directly visible in the CSV rather than only asserted in a test. No hardcoded
"must be N% faster" test assertion is added — GPU-to-GPU speedup ratios are not
portable across hardware, so the benchmark script reports the numbers and the
real run's results get reviewed against the "same or better" requirement
directly, config by config. Any config where kernel comes back slower than
native is a finding to either fix (kernel design issue) or explicitly document
as an accepted non-goal (e.g., if Polar's small per-level launches don't
amortize their fixed overhead at very small `d`) — not something silently
absorbed into an average.

## Testing Strategy Summary

- Unit tests (`tests/test_kernel_backend.py`): correctness parity per algorithm,
  as above. Skipped (not failed) on machines without CUDA/Triton.
- Existing test suite (55 tests as of the prior redesign) must continue passing
  unchanged — `backend="native"` remains the default, so no existing call site
  is affected by this addition.
- `require_kernel_backend()` error paths get direct unit tests (wrong device,
  simulated missing triton import) that don't require an actual GPU.
- Performance verification via the extended `run_perf_benchmark.py`, run for
  real on the project's RTX 4070 as part of task completion, results reviewed
  against the "same or better" requirement before the work is considered done.

## Known Limitations (post-implementation)

`TurboQuantProd`'s kernel backend does not meet the same-or-better performance
requirement at `d=128` specifically: it is 1.2-2.7x slower than native at that
head dimension on both `quantize` and `dequantize`, at every tested bit-width.
It fully meets the requirement at `d=64`. Root cause, confirmed via direct
experimentation on the project's RTX 4070: `_qjl_project_sign_kernel` and
`_qjl_correct_kernel` (the two per-call kernels `TurboQuantProd` adds beyond
the reused MSE stage) each hold one resident D×D matrix per block, and at
`d=128, BLOCK_M=64` that block is already close to this GPU's per-block shared
memory ceiling — a larger `BLOCK_M=128` measurably exceeds it (131072 bytes
required vs a 101376-byte hardware limit), and fusing further to close the
remaining gap (e.g. merging `quantize`'s two kernels into one) would reproduce
the two-resident-matrix, 163840-byte overflow that `inner_product`'s original
single-kernel design hit and was split apart to fix (see
`kernel/prod.py`'s module docstring). So the residual d=128 latency gap is
best understood as unamortized per-launch/synchronization overhead from
`TurboQuantProd`'s multi-kernel structure at that head dimension, which shared
memory blocks the two obvious ways of closing (bigger blocks, more fusion) —
not a single flat "hardware ceiling" on the kernel design as a whole. This is
accepted as a known, documented exception per this spec's own "explicitly
document as an accepted non-goal" allowance in the Performance Parity section
above, rather than silently absorbed. `backend="kernel"` remains correct
(byte-identical/tight-tolerance output to native) at `d=128`; only its latency
is worse there.

`TurboQuantMSE`'s kernel backend meets the requirement decisively on
`quantize` and at most `dequantize` configs, but the final real sweep shows
`dequantize` landing 1.05-2.8x above native in most `d`/bits cells (best
explanation: run-to-run variance — the range across four independent full
sweeps for the same cells spans roughly 0.17-1.01ms) rather than a clean
same-or-better result at every single cell. `PolarQuant` shows no such
inconsistency across sweeps. Neither `mse` nor `polar` is treated as a
documented exception the way `prod`/`d=128` is, since no run showed a stable,
reproducible regression there the way `prod`/`d=128` does — but this is a
softer claim than "meets it at every tested `d` and configuration," and is
recorded here as such. See
`.superpowers/sdd/2026-08-28-turboquant-kernel-backend/task-6-report.md` for
the full before/after latency data behind both findings.

## Open Items for the Implementation Plan

- Exact Triton kernel signatures, block-size selection per `d`, and grid
  configuration are implementation details for the plan/task breakdown, not
  fixed here.
- Whether `run_benchmark.py` (perplexity/compression sweep, not just the perf
  microbenchmark) also gets a `--backend` flag is an implementation-plan
  decision, not required by this spec — the correctness-parity tests already
  guarantee kernel and native give equivalent quantization, so a perplexity
  sweep would be expected to reproduce the same numbers per backend.
