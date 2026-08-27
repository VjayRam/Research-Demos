# TurboQuant Redesign — Paper-Accurate Importable Package

Date: 2026-08-27
Status: Approved for planning

## Problem

The existing `turbo-quant/` implementation (`turboquant.py`, `compressors.py`,
`lloyd_max.py`, `evaluate.py`) deviates from the TurboQuant paper
(arXiv:2504.19874) in ways that were reasonable engineering shortcuts but are
not what the paper specifies:

1. **Rotation**: uses a Fast Walsh-Hadamard Transform + random sign flips
   (`fwht` / `generate_hadamard_signs` in `turboquant.py`) instead of the
   paper's true Haar-random orthogonal matrix built via QR decomposition of a
   Gaussian matrix.
2. **Lloyd-Max density**: `lloyd_max.py`'s `solve_lloyd_max` defaults to
   `use_exact=False`, solving against a `N(0, 1/d)` Gaussian approximation
   rather than the paper's exact Beta density (Eq. 4:
   `f_X(x) = Γ(d/2) / (√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)`).
3. **Scope creep**: `compressors.py` (`TurboQuantV3`) bundles paper-unrelated
   KV-cache engineering (asymmetric key/value bit-widths, protected layers,
   residual windowing, bit-packed storage) directly into the quantization
   core, making it impossible to cleanly test "the paper's algorithm" in
   isolation.

The user wants a redesigned, paper-accurate implementation ("no assumptions,
no alternatives") packaged as an importable Python module, plus a way to
validate it against real, modern open-weight LLMs (Qwen, Gemma).

Reference for exact paper formulas: `turbo-quant/turboquant-primer.html`
(interactive primer built from the TurboQuant, QJL, and PolarQuant papers) —
sections `#algorithm-1`, `#algorithm-2`, and `#polarquant` are the ground
truth transcriptions used throughout this spec.

## Goals

- A `turboquant` Python package, pip-installable from the `turbo-quant/`
  directory, implementing:
  - **Algorithm 1** (`TurboQuant_mse`): rotate → per-coordinate Lloyd-Max
    quantize → dequantize → inverse rotate.
  - **Algorithm 2** (`TurboQuant_prod`): `(b-1)`-bit MSE stage + 1-bit QJL
    sign-quantization of the residual, for unbiased inner-product estimation.
  - **PolarQuant**: recursive Cartesian→polar decomposition with per-level
    Lloyd-Max codebooks on the sin-power angle densities.
- Every numerical choice traceable to a specific line/formula in the paper
  (via the primer). No Gaussian-approximation shortcuts, no FWHT substitution,
  no alternate "practical" variants inside the core package.
- A separate, non-core `examples/` harness that hooks the package into real
  HuggingFace models (small Qwen2.5 / Gemma-2 checkpoints) to measure
  perplexity vs. compression ratio when the KV cache is round-tripped through
  the quantizer.
- Old flat-file implementation (`turboquant.py`, `compressors.py`,
  `lloyd_max.py`, `test_algorithm.py`, `evaluate.py`) is fully replaced, not
  kept alongside the new package.

## Non-goals

- Production bit-packed storage / actual memory savings during real inference
  (the old `MSECompressor`'s byte-packing). The examples harness round-trips
  through the quantizer for correctness/quality measurement, not to build a
  deployable compressed-KV-cache runtime.
- Any KV-cache-specific heuristics from the old `TurboQuantV3` (asymmetric
  K/V bits, protected layers, residual windows) — out of scope for the core
  package; could be revisited later as a distinct example, not part of this
  spec.
- GPU-performance optimization (fused kernels, O(d log d) rotation, binary
  search over boundaries). Correctness and paper-fidelity take priority;
  `head_dim` sizes in KV caches (64–256) make the O(d²) QR rotation and
  O(2^b) argmin cheap enough.

## Architecture

```
turbo-quant/
  pyproject.toml
  turboquant/
    __init__.py
    rotation.py
    distributions.py
    lloyd_max.py
    codebook.py
    qjl.py
    cartesian.py
    polar.py
  examples/
    kv_cache_hook.py
    run_benchmark.py
  README.md
```

### `rotation.py`

`generate_rotation_matrix(d, seed) -> Tensor[d,d]`: QR-decompose a
`d×d` Gaussian matrix, fix the sign ambiguity (`Q * sign(diag(R))`), return
the Haar-distributed orthogonal `Π`. Cached per `(d, seed)` — data-independent
per the paper's "setup, once per (d,b)" step.

No FWHT, no sign-flip variant. This is the only rotation primitive in the
package.

### `distributions.py`

Two densities, each exposing `pdf(x)` and its support interval:

- `beta_coordinate_density(d)`: the exact Eq. 4 density used by Algorithm 1/2.
- `polar_angle_density(level)`: uniform on `[0, 2π)` for level 1;
  `∝ sin^(2^(level-1)-1)(2θ)` on `[0, π/2]` for level ≥ 2, used by PolarQuant.

Both implemented with `math.gamma`/`math.lgamma` for numerical stability at
higher `d` (the existing `math.gamma` direct-ratio approach in the old
`lloyd_max.py` is fine up to the `d` values used for KV-cache head dims, but
`lgamma`-based computation avoids overflow if larger `d` is ever tried).

### `lloyd_max.py`

`solve_lloyd_max(pdf, support, bits, max_iter=200, tol=1e-10) -> (centroids, boundaries)`:
the generic continuous Lloyd-Max iteration (boundaries = midpoints of
neighboring centroids; centroids = conditional mean of each bucket via
`scipy.integrate.quad`), parameterized by any `(pdf, support)` pair — no
knowledge of Beta vs. polar-angle baked in. Both `cartesian.py` and
`polar.py` call this with their respective density from `distributions.py`.

Single code path — no `use_exact` flag, no Gaussian branch.

### `codebook.py`

`Codebook` dataclass: `centroids`, `boundaries`; `quantize(x)` does literal
`argmin_k |x - c_k|` (the paper's Algorithm 1 line, not a bucketize
substitution — cheap since `2^b ≤ 16` in all tested configurations);
`dequantize(idx)` is a lookup. A module-level cache keyed by
`(density_signature, bits)` avoids re-solving Lloyd-Max for repeated
`(d, bits)` pairs within a process.

### `qjl.py`

`generate_qjl_matrix(d, seed) -> Tensor[d,d]` (`S_ij ~ N(0,1)`, always square
— TurboQuant's use of QJL is "1 bit per coordinate" on the residual, not a
dimensionality-reducing projection like the standalone QJL paper's `m < d`
case). `sign(S @ r)` with the zero→+1 tie-break from the existing code kept
(not paper-specified, but a necessary and inconsequential floating-point
tie-break, not an algorithmic alternative).

### `cartesian.py`

`TurboQuantMSE(d, bits, seed)`:
- `.rotate(x)` / `.unrotate(y)` using `rotation.py`'s `Π`.
- `.quantize(x)`: normalize by `‖x‖₂` (stored separately, one scalar per
  vector, per the paper's note), rotate, per-coordinate `Codebook.quantize`.
- `.dequantize(indices, norm)`: codebook lookup, `Πᵀ · ŷ` (orthogonal ⇒
  transpose = inverse), rescale by the stored norm.

`TurboQuantProd(d, bits, seed)`, implementing Algorithm 2 verbatim:
- Internally holds a `TurboQuantMSE(d, bits-1, seed)` for the base stage.
- `.quantize(x)`: `idx = mse.quantize(x)`; `r = x - mse.dequantize(idx)`;
  `qjl = sign(S @ r)`; returns `(idx, qjl, ‖r‖₂)`.
- `.dequantize(idx, qjl, rho)`: `x̂_mse + (√(π/2)/d) · rho · Sᵀ·qjl`.
- `.inner_product(y, compressed)`: the unbiased estimator combining the MSE
  reconstruction term and the QJL correction term, matching
  `E[⟨y,x̃⟩] = ⟨y,x⟩` exactly as derived in the primer.

Both classes take `d, bits, seed` only — no `qjl_dim` override, no algorithm
switch. One behavior per class.

### `polar.py`

`PolarQuant(d, bits, seed)`, `d` a power of 2:
- Precondition: rotate `x` by the same `rotation.py` Haar matrix (paper
  describes preconditioning as the same root idea as TurboQuant's rotation;
  reusing `rotation.py` avoids introducing a second, different randomization
  primitive).
- Recursive decomposition: pair up coordinates, compute `(r, θ)` per pair,
  recurse on the vector of radii for `log2(d)` levels, producing one final
  scalar radius and `d-1` angles total.
- Per-level codebooks: level-1 angles quantized against the uniform density;
  level `ℓ ≥ 2` angles quantized against `polar_angle_density(ℓ)` — both via
  the same `lloyd_max.py` solver.
- `.quantize(x)` / `.dequantize(...)` walk the recursion forward/backward.
- `bits` here means bits-per-angle (paper frames it as `O(log(1/ε))` bits per
  coordinate); constant across levels for a first implementation — no
  per-level bit allocation scheme, since the paper doesn't specify one.

### `__init__.py`

Re-exports `TurboQuantMSE`, `TurboQuantProd`, `PolarQuant` directly — no
factory wrapper. "Configurable by the user" means picking which class to
instantiate, not a runtime string-switch inside one god-class.

## Data flow (KV-cache test harness)

```
HF model forward (per layer)
  → attention module produces key/value tensors [B,H,S,D]
  → examples/kv_cache_hook.py: for each new token's K/V vector,
       quantizer.quantize(v) → compressed repr
       quantizer.dequantize(compressed) → v̂  (round-trip immediately)
  → v̂ replaces v in the KV cache used for subsequent attention
  → generation / perplexity computed on the reconstructed cache
```

`run_benchmark.py`:
- Loads `Qwen/Qwen2.5-0.5B` and `google/gemma-2-2b` (small enough for local
  CPU/GPU runs) via `transformers`.
- For each model, each of `TurboQuantMSE` / `TurboQuantProd` / `PolarQuant`,
  and `bits ∈ {1,2,3,4}`: runs a short wikitext-2 sample, computes perplexity
  with the KV cache round-tripped through the quantizer, and reports
  compression ratio (`fp16 bytes / packed-index-equivalent bytes`, computed
  analytically — no actual bit-packing needed since this harness measures
  quality, not deployable memory savings) alongside the perplexity delta vs.
  unquantized.
- Reuses the perplexity-measurement approach from the old `evaluate.py` where
  applicable, rewritten against the new package's API.

## Testing

- Unit tests (pytest, replacing `test_algorithm.py`) per module:
  - `rotation.py`: output is orthogonal (`Πᵀ·Π ≈ I`), Haar-ness spot-check
    (coordinate distribution of `Π·e_1` matches `beta_coordinate_density`
    via a KS test over many seeds).
  - `distributions.py`: `beta_coordinate_density(d)` integrates to 1 over
    `[-1,1]`; matches the primer's closed-form values at a few `(d, x)`
    points.
  - `lloyd_max.py`: for `d=128`, reproduces the primer's Theorem 1 table
    (`D_mse` within tolerance of `0.360/0.117/0.030/0.009` for `b=1..4`).
  - `cartesian.py`: reproduces the primer's two hand-worked examples
    (`#worked-simple`, `d=4,b=1`, input `x=(1,0,0,0)`;
    `#worked-moderate`, `d=8,b=2`) by feeding the primer's exact input
    vectors through the new package and asserting the resulting centroids,
    reconstruction, and error match the primer's reported numbers within
    floating-point tolerance; `TurboQuantProd.inner_product` is empirically
    unbiased (mean estimate over many random `(x,y)` pairs converges to true
    `⟨x,y⟩`).
  - `polar.py`: round-trip reconstruction error decreases monotonically with
    `bits`; angle distributions at each level match
    `polar_angle_density(level)` empirically.
- Integration: `examples/run_benchmark.py` is not part of the pytest suite
  (it downloads real model weights) but is a runnable script with a
  `--smoke-test` flag that runs one tiny model for a few tokens, for CI-free
  manual verification.

## Error handling

- `d` not a power of 2 in `PolarQuant`: raise `ValueError` at construction
  (recursion requires exact pairing at every level).
- `bits < 1`: raise `ValueError` — `TurboQuantProd` additionally requires
  `bits ≥ 2` since it needs `bits-1 ≥ 1` for its internal MSE stage.
- All randomness (`rotation.py`, `qjl.py`) is seeded and deterministic given
  a seed; no silent fallback to unseeded global RNG state.

## Migration

- Delete `turbo-quant/turboquant.py`, `turbo-quant/compressors.py`,
  `turbo-quant/lloyd_max.py`, `turbo-quant/test_algorithm.py`,
  `turbo-quant/evaluate.py`.
- Update `turbo-quant/README.md` to document the new package's API and how
  to run `examples/run_benchmark.py`.
- `turbo-quant/turboquant-primer.html` is unaffected — it remains the
  human-facing explainer and the source of truth this spec was built from.
