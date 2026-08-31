# RDKV Implementation Plan (Phase 1 + Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute tasks strictly in order — Phase 2 (Tasks 9–14) depends on Phase 1's (Tasks 1–8) `rdkv.pipeline.RDKVAllocator` output.

**Goal:** Build `rdkv` as an importable Python package implementing the full RDKV mechanism end to end: Phase 1 covers the core allocation math — continuous water-filling (Theorem 3.3), discrete MCKP bisection (Algorithm 2), per-unit weight computation (Propositions 3.1/3.2), and the three-stage allocation pipeline (§7/Algorithm 1 Stages 1–3). Phase 2 covers TriZone packed storage and the fused-dequantization decode step (§8/§9, Algorithm 1 Stage 4, Eq. 7), so a `RDKVAllocator` allocation can actually be packed, stored, and decoded from without ever materializing a dequantized FP16 tile. Every formula is validated against the spec's hand-worked example; every Phase 2 kernel is validated against a native (non-fused) reference before any performance work.

**Architecture:** A flat, dependency-light package (`rdkv/rdkv/`) mirroring `turbo-quant/turboquant/`'s and `seq-attention/seqattention/`'s layout: one module per mathematical component, a mirrored `tests/` directory, and an `examples/` script that wires in a real HF model for end-to-end stats. Phase 2 adds a `rdkv/rdkv/kernel/` subpackage (native PyTorch scalar quantizer + Triton fused kernel, both under one dispatch API), following `turbo-quant/turboquant/kernel/`'s native/kernel backend split exactly — same CUDA-only guard, same "native first, kernel is an opt-in accelerated backend with parity tests" structure.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.0, pytest. `transformers`/`accelerate` as an `examples` extra (not core). Phase 2 adds Triton as a `kernel` extra (`triton` on Linux, `triton-windows` on Windows — matching `turbo-quant/pyproject.toml`'s platform split), never a core dependency.

**Spec:** [`docs/superpowers/specs/2026-08-31-rdkv-design.md`](../specs/2026-08-31-rdkv-design.md)

## Global Constraints

- Every formula implemented must match the spec's transcription of the paper exactly — variable names in code should be traceable to the spec's notation (`w_t`, `w_c`, `sigma_u`, `lambda_`, `b_u`) via docstrings citing the spec section, the way `turbo-quant/turboquant/cartesian.py` cites "Algorithm 1" / "Algorithm 2" in its docstrings.
- `ε_u(b)` in Phase 1 is the Bennett-curve stand-in `σ_u · 2^(−b)` (spec §14 decision 2) — every function that takes or computes it must say so in its docstring, not present it as the paper's real calibrated table. Phase 2 does not revisit this — it packs and decodes whatever bit-widths Phase 1's allocator already chose; it never re-derives the allocation itself.
- Bit-width set is fixed: `𝔹 = {0, 2, 4, 8, 16}` (spec §1, §6).
- Algorithm 2 hyperparameter defaults (spec §6): bisection tolerance `δ = 1e-2`, max iterations `I = 64`. These are defaults, overridable per-call.
- Phase 1 (Tasks 1–8) has zero GPU kernel, zero Triton dependency. Phase 2 (Tasks 9–14) is the only place `triton`/`triton-windows` is introduced, gated behind the `kernel` extra and a CUDA-only runtime guard (spec §8's TriZone/fused-decode work is inherently a GPU-memory-layout optimization; it has no meaningful CPU-only form the way Phase 1's math does).
- Package name: `rdkv`. Directory layout, `pyproject.toml` shape, and `test`/`examples`/`kernel` extras follow `turbo-quant/pyproject.toml` and `seq-attention/pyproject.toml` exactly (see Task 1; `kernel` extra added in Task 9).
- Tests are dependency-free (synthetic Q/K/V only, per spec §14 decision 3) — no model download in the `pytest` suite. The real-model path lives only in `examples/`, mirroring `turbo-quant/examples/kv_cache_hook.py`'s separation from `turbo-quant/tests/`. Phase 2's kernel-parity tests follow `turbo-quant/tests/test_kernel_backend.py`'s pattern: skipped (not failed) on machines without CUDA+Triton, never silently passing without running.
- Phase 2 never materializes a dequantized FP16 tile in the fused decode path (spec §8's explicit requirement) — every kernel task's correctness test must assert this isn't just numerically right but structurally fused (no intermediate full-precision K tensor allocated for the packed zone).

---

## File Structure

```
rdkv/
├── rdkv/
│   ├── __init__.py          # public API: exports from all modules below
│   ├── waterfilling.py      # spec §5 (Theorem 3.3): continuous closed-form bit allocation
│   ├── mckp.py               # spec §6/§10 (Algorithm 2): discrete MCKP via Lagrangian bisection
│   ├── weights.py            # spec §2-4 (Prop 3.1, Prop 3.2, Bennett sigma_u): per-unit weights
│   ├── pipeline.py           # spec §7/§9 Stages 1-3: RDKVAllocator, orchestrates the above
│   ├── trizone.py            # Phase 2, spec §8/§9 Stage 4: TriZone packing (native, CPU/GPU)
│   ├── decode.py             # Phase 2, spec §8 Eq. 7: packed-decode output decomposition (native)
│   └── kernel/                # Phase 2: GPU-only fused Triton backend, mirrors turbo-quant/turboquant/kernel/
│       ├── __init__.py
│       ├── _require.py       # CUDA+Triton guard, mirrors turbo-quant's kernel/_require.py
│       └── fused_decode.py   # fused K dequant + attention, no materialized FP16 tile
├── tests/
│   ├── __init__.py
│   ├── test_waterfilling.py
│   ├── test_mckp.py
│   ├── test_weights.py
│   ├── test_pipeline.py
│   ├── test_trizone.py
│   ├── test_decode.py
│   └── test_kernel_backend.py  # Phase 2, skipped without CUDA+Triton (mirrors turbo-quant's pattern)
├── examples/
│   ├── allocation_stats.py   # wires a real small HF model, reports per-layer allocation stats
│   └── packed_decode_demo.py  # Phase 2: end-to-end allocate -> pack -> decode on a real model
├── rdkv-primer.html          # existing, unchanged
├── pyproject.toml
└── README.md
```

**Module responsibilities (Phase 1):**
- `waterfilling.py` — `continuous_waterfill(w, sigma, budget)`: the real-valued `b_u*` closed form and its bisection-on-`λ` solver, independent of any discretization.
- `mckp.py` — `mckp_bisect(w, sigma, target_avg_bits, bit_widths=(0,2,4,8,16), tol=1e-2, max_iter=64)`: Algorithm 2 verbatim, using the Bennett-curve stand-in internally as `ε_u(b)` unless a caller-supplied distortion function overrides it (keeps the door open for Phase 2's real calibration table without an API break).
- `weights.py` — `token_weight_v(attn)` (Prop 3.1 / Eq. 2), `channel_weight_k(q, k)` (Prop 3.2 / Eq. 4), `bennett_sigma(dynamic_range)` (§4).
- `pipeline.py` — `RDKVAllocator`: Stage 1 (weight computation) → Stage 2 (V token allocation) → Stage 3 (K channel allocation on kept tokens only), enforcing the V-before-K ordering constraint from spec §7.

**Module responsibilities (Phase 2):**
- `trizone.py` — `pack_trizone(k, v, allocation, q, ...)`: Algorithm 1 Stage 4 — sorts kept V tokens by `b_v` into `{2,4,8}`-bit sub-segments, sorts K channels by `b_c^K` into segments (permuting `q` to match), quantizes/byte-packs into Zone A, and splits FP16-retained V rows (`b_v=16`) into Zone B. Pure PyTorch (CPU or GPU) — this is memory-layout bookkeeping, not a kernel, so it doesn't require Triton.
- `decode.py` — `packed_decode(packed_cache, q_new, k_new, v_new)`: Eq. (7)'s three-way output decomposition (Zone A quantized-V, Zone B FP16-retained, Zone C new-decode-tokens) and the algebraic K dequantization rewrite from §8 (`q_τᵀ k̂_t` as a sum over `(s_c q_{τ,c})·k̃_{t,c}` minus one per-query-head bias), implemented as ordinary PyTorch ops first — this is the *correctness* reference Task 12's fused kernel is checked against, not the fast path itself.
- `kernel/fused_decode.py` — GPU-only Triton kernel fusing Zone A's algebraic K dequantization directly into the attention score computation, so no dequantized FP16 K tile is ever materialized in HBM (spec §8's core performance claim). Dispatches only when `backend="kernel"` is explicitly requested, mirroring `TurboQuantMSE`'s `backend` parameter.
- `kernel/_require.py` — `require_kernel_backend(device)`: CUDA-only guard raising `RuntimeError` with an install hint, verbatim pattern from `turbo-quant/turboquant/kernel/_require.py`.

---

## Task 1: Package scaffold

**Files:**
- Create: `rdkv/pyproject.toml`
- Create: `rdkv/rdkv/__init__.py`
- Create: `rdkv/tests/__init__.py`
- Create: `rdkv/README.md`

**Interfaces:**
- Produces: an installable `rdkv` package (editable install via `pip install -e rdkv[test]`) that later tasks add modules into.

- [ ] **Step 1: Write `rdkv/pyproject.toml`**

```toml
[project]
name = "rdkv"
version = "0.1.0"
description = "Paper-accurate implementation of RDKV rate-distortion KV cache bit allocation"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
]

[project.optional-dependencies]
examples = [
    "transformers>=4.40",
    "accelerate>=0.30",
]
test = [
    "pytest>=7.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["rdkv*"]
```

- [ ] **Step 2: Write `rdkv/rdkv/__init__.py`**

```python
"""Paper-accurate implementation of RDKV rate-distortion KV cache bit
allocation (Rate-Distortion Bit Allocation for Joint Eviction and
Quantization of the KV Cache, arXiv:2605.08317)."""
```

(Exports are added incrementally in later tasks — Task 5 finalizes this file.)

- [ ] **Step 3: Write `rdkv/tests/__init__.py`**

Empty file (matches `turbo-quant/tests/__init__.py`).

- [ ] **Step 4: Write `rdkv/README.md`**

```markdown
# RDKV

Paper-accurate implementation of [RDKV: Rate-Distortion Bit Allocation for
Joint Eviction and Quantization of the KV Cache](https://arxiv.org/abs/2605.08317)
(arXiv:2605.08317).

See [`rdkv-primer.html`](rdkv-primer.html) for the interactive derivation
walkthrough, and
[`../docs/superpowers/specs/2026-08-31-rdkv-design.md`](../docs/superpowers/specs/2026-08-31-rdkv-design.md)
for the full spec this implementation follows.

**Phase 1 (this code):** continuous water-filling (Theorem 3.3), discrete
MCKP bit allocation (Algorithm 2), per-unit weight computation
(Propositions 3.1/3.2), and the three-stage allocation pipeline (Algorithm 1
Stages 1-3). Pure PyTorch, no custom GPU kernel.

**Not yet implemented (Phase 2):** TriZone packing and the fused
dequantization attention kernel (Algorithm 1 Stage 4).

**Disclosed approximation:** the paper's empirically-calibrated
per-coordinate distortion table `ε_u(b)` (Appendix B) is stood in for by the
analytic Bennett curve `σ_u · 2^(−b)` throughout this phase — see
`rdkv/mckp.py`.

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest tests/
```
```

- [ ] **Step 5: Verify the package installs**

Run: `cd rdkv && pip install -e ".[test]"`
Expected: installs with no errors, `import rdkv` succeeds from a Python shell.

- [ ] **Step 6: Commit**

```bash
git add rdkv/pyproject.toml rdkv/rdkv/__init__.py rdkv/tests/__init__.py rdkv/README.md
git commit -m "rdkv: scaffold importable package"
```

---

## Task 2: Continuous water-filling (Theorem 3.3)

**Files:**
- Create: `rdkv/rdkv/waterfilling.py`
- Test: `rdkv/tests/test_waterfilling.py`

**Interfaces:**
- Produces: `continuous_waterfill(w: torch.Tensor, sigma: torch.Tensor, target_budget: float, lambda_lo: float = 1e-6, lambda_hi: float = 1e6, iters: int = 60) -> tuple[torch.Tensor, float]` returning `(b_star, lambda_)` where `b_star` is a 1-D float tensor of continuous (unclipped-to-hardware) bit-widths and `lambda_` is the converged Lagrange multiplier.
- Consumes: nothing from other rdkv modules (pure function, first module built).

- [ ] **Step 1: Write the failing tests**

```python
# rdkv/tests/test_waterfilling.py
import math

import torch

from rdkv.waterfilling import continuous_waterfill


def test_matches_spec_worked_example_at_fixed_lambda():
    # Spec Sec 11: b_u* = [log2(w_u*sigma_u / 0.1)]_+ at lambda/ln2 = 0.1, sigma_u = 1.
    # continuous_waterfill solves for lambda given a budget, so we instead
    # check the underlying formula directly at the paper's lambda value by
    # picking a target_budget that yields lambda/ln2 == 0.1, then comparing
    # to the spec's table of continuous b_u* values.
    w = torch.tensor([0.500, 0.300, 0.150, 0.050, 1.1619, 0.6893])
    sigma = torch.ones_like(w)
    # At lambda/ln2 = 0.1 (i.e. lambda = 0.1*ln2), sum of clipped b_u* is:
    target_lambda_over_ln2 = 0.1
    lambda_fixed = target_lambda_over_ln2 * math.log(2)
    b_expected = torch.clamp(torch.log2(math.log(2) * w * sigma / lambda_fixed), min=0.0)
    target_budget = b_expected.sum().item()

    b_star, lambda_ = continuous_waterfill(w, sigma, target_budget)

    assert torch.allclose(b_star, b_expected, atol=1e-2)
    assert math.isclose(lambda_, lambda_fixed, rel_tol=1e-2)


def test_token_4_is_evicted_in_worked_example():
    # Spec Sec 11: token 4 (w=0.050) falls below the water level and clips to 0.
    w = torch.tensor([0.500, 0.300, 0.150, 0.050])
    sigma = torch.ones_like(w)
    lambda_fixed = 0.1 * math.log(2)
    b_expected = torch.clamp(torch.log2(math.log(2) * w * sigma / lambda_fixed), min=0.0)
    target_budget = b_expected.sum().item()

    b_star, _ = continuous_waterfill(w, sigma, target_budget)

    assert b_star[3].item() == 0.0
    assert b_star[0].item() > 0.0


def test_budget_binds():
    w = torch.tensor([1.0, 0.5, 0.25, 0.1, 0.05])
    sigma = torch.ones_like(w)
    target_budget = 6.0

    b_star, _ = continuous_waterfill(w, sigma, target_budget)

    assert math.isclose(b_star.sum().item(), target_budget, abs_tol=1e-2)


def test_tightening_budget_increases_eviction_count():
    w = torch.tensor([1.0, 0.5, 0.25, 0.1, 0.05])
    sigma = torch.ones_like(w)

    b_loose, _ = continuous_waterfill(w, sigma, target_budget=15.0)
    b_tight, _ = continuous_waterfill(w, sigma, target_budget=1.0)

    evicted_loose = (b_loose <= 0).sum().item()
    evicted_tight = (b_tight <= 0).sum().item()
    assert evicted_tight >= evicted_loose
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rdkv && pytest tests/test_waterfilling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rdkv.waterfilling'`

- [ ] **Step 3: Write the implementation**

```python
# rdkv/rdkv/waterfilling.py
"""Theorem 3.3 (spec Sec 5): continuous reverse water-filling closed-form
bit allocation.

    b_u* = [ log2( ln2 * w_u * sigma_u / lambda ) ]_+

with lambda > 0 chosen so sum(b_u*) matches a target budget. This module
does not discretize to hardware bit-widths {0,2,4,8,16} -- see
rdkv.mckp for the discrete version (Algorithm 2).
"""

import math

import torch


def continuous_waterfill(
    w: torch.Tensor,
    sigma: torch.Tensor,
    target_budget: float,
    lambda_lo: float = 1e-6,
    lambda_hi: float = 1e6,
    iters: int = 60,
) -> tuple[torch.Tensor, float]:
    """Solve Theorem 3.3 for the lambda that makes sum(b_u*) == target_budget.

    w, sigma: 1-D tensors of equal length (per-unit weight and Bennett
    sigma_u). target_budget: desired sum of continuous bit-widths.
    Bisection is geometric in lambda since lambda can span many orders
    of magnitude.

    Returns (b_star, lambda_): b_star is a 1-D float tensor, lambda_ is
    the converged Lagrange multiplier.
    """
    if w.shape != sigma.shape:
        raise ValueError(f"w and sigma must have the same shape, got {w.shape} vs {sigma.shape}")

    def b_at(lambda_: float) -> torch.Tensor:
        value = torch.log2(math.log(2) * w * sigma / lambda_)
        return torch.clamp(value, min=0.0)

    lo, hi = lambda_lo, lambda_hi
    lambda_ = math.sqrt(lo * hi)
    for _ in range(iters):
        lambda_ = math.sqrt(lo * hi)
        current_sum = b_at(lambda_).sum().item()
        if abs(current_sum - target_budget) < 1e-4:
            break
        if current_sum > target_budget:
            lo = lambda_
        else:
            hi = lambda_
    lambda_ = math.sqrt(lo * hi)
    return b_at(lambda_), lambda_
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rdkv && pytest tests/test_waterfilling.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add rdkv/rdkv/waterfilling.py rdkv/tests/test_waterfilling.py
git commit -m "rdkv: implement Theorem 3.3 continuous water-filling"
```

---

## Task 3: Discrete MCKP bit allocation (Algorithm 2)

**Files:**
- Create: `rdkv/rdkv/mckp.py`
- Test: `rdkv/tests/test_mckp.py`

**Interfaces:**
- Consumes: nothing from `waterfilling.py` (independent implementation per spec §6, though it approximates the same underlying curve).
- Produces: `bennett_distortion(sigma: torch.Tensor, b: torch.Tensor) -> torch.Tensor` (the Phase 1 stand-in `ε_u(b) = σ_u · 2^(−b)`, spec §14 decision 2) and `mckp_bisect(w: torch.Tensor, sigma: torch.Tensor, target_avg_bits: float, bit_widths: tuple[int, ...] = (0, 2, 4, 8, 16), tol: float = 1e-2, max_iter: int = 64, abs_tol_zero: float = 1e-9) -> tuple[torch.Tensor, dict]` returning `(b_star, info)` where `b_star` is a 1-D long tensor of hardware bit-widths and `info` has keys `lambda_`, `iters`, `converged`, `trace` (list of per-iteration `(lambda_lo, lambda_hi, lambda_, mean_b)` tuples, matching the primer's bisection-trace table).

- [ ] **Step 1: Write the failing tests**

```python
# rdkv/tests/test_mckp.py
import math

import torch

from rdkv.mckp import bennett_distortion, mckp_bisect


def test_bennett_distortion_matches_formula():
    sigma = torch.tensor([2.0, 1.0])
    b = torch.tensor([0.0, 4.0])
    result = bennett_distortion(sigma, b)
    expected = torch.tensor([2.0 * 2**0, 1.0 * 2**-4])
    assert torch.allclose(result, expected)


def test_zero_bits_gets_full_sigma_as_distortion():
    sigma = torch.tensor([3.5])
    b = torch.tensor([0.0])
    assert torch.allclose(bennett_distortion(sigma, b), sigma)


def test_bit_widths_are_from_the_fixed_hardware_set():
    torch.manual_seed(0)
    w = torch.rand(20) + 0.01
    sigma = torch.ones_like(w)
    b_star, _ = mckp_bisect(w, sigma, target_avg_bits=4.0)
    allowed = {0, 2, 4, 8, 16}
    assert set(b_star.tolist()).issubset(allowed)


def test_converges_to_target_average_within_tolerance():
    torch.manual_seed(1)
    w = torch.rand(50) + 0.01
    sigma = torch.ones_like(w)
    target = 3.0
    b_star, info = mckp_bisect(w, sigma, target_avg_bits=target)
    mean_b = b_star.float().mean().item()
    assert info["converged"]
    assert abs(mean_b - target) / target < 0.15  # discrete rounding, wider than the bisection's own tol


def test_zero_target_budget_returns_all_evicted():
    w = torch.tensor([1.0, 0.5, 0.25])
    sigma = torch.ones_like(w)
    b_star, info = mckp_bisect(w, sigma, target_avg_bits=0.0)
    assert torch.all(b_star == 0)
    assert info["converged"]


def test_higher_weight_units_get_at_least_as_many_bits():
    torch.manual_seed(2)
    w = torch.tensor([0.9, 0.1])
    sigma = torch.ones_like(w)
    b_star, _ = mckp_bisect(w, sigma, target_avg_bits=4.0)
    assert b_star[0].item() >= b_star[1].item()


def test_worked_example_token4_evicted_others_quantized():
    # Spec Sec 11 qualitative pattern: lowest-weight unit (token 4, w=0.05)
    # should be the first evicted as budget tightens, matching Theorem 3.3's
    # continuous result where token 4 clips to 0 first.
    w = torch.tensor([0.500, 0.300, 0.150, 0.050])
    sigma = torch.ones_like(w)
    b_star, _ = mckp_bisect(w, sigma, target_avg_bits=1.0)
    assert b_star[3].item() == 0
    assert b_star[0].item() >= b_star[3].item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rdkv && pytest tests/test_mckp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rdkv.mckp'`

- [ ] **Step 3: Write the implementation**

```python
# rdkv/rdkv/mckp.py
"""Algorithm 2 (spec Sec 10, Appendix C): MCKP via Lagrangian bisection.

Solves the discrete multiple-choice knapsack problem from Eq. (6):

    {b_u*} = argmin_{b_u in B} sum_u w_u * eps_u(b_u)   s.t. sum_u b_u <= B

by bisecting on the Lagrange multiplier lambda. Each unit independently
picks the bit-width minimizing w_u*eps(b) + lambda*b; bisection adjusts
lambda until the mean chosen bit-width matches the target average.

DISCLOSED APPROXIMATION (spec Sec 14, decision 2): eps_u(b) here is the
analytic Bennett curve sigma_u * 2**-b (spec Sec 4), NOT the paper's real
empirically-calibrated per-coordinate distortion table (Appendix B, fit
offline on 32 LongBench prefill sequences). Callers who later have a real
calibrated table can pass their own eps_fn to mckp_bisect.
"""

import torch

DEFAULT_BIT_WIDTHS = (0, 2, 4, 8, 16)


def bennett_distortion(sigma: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """eps_u(b) = sigma_u * 2**-b -- the Phase 1 stand-in for the paper's
    empirically-calibrated distortion table (see module docstring)."""
    return sigma * torch.pow(2.0, -b)


def mckp_bisect(
    w: torch.Tensor,
    sigma: torch.Tensor,
    target_avg_bits: float,
    bit_widths: tuple[int, ...] = DEFAULT_BIT_WIDTHS,
    tol: float = 1e-2,
    max_iter: int = 64,
    abs_tol_zero: float = 1e-9,
    eps_fn=bennett_distortion,
) -> tuple[torch.Tensor, dict]:
    """Algorithm 2, verbatim bisection structure.

    w, sigma: 1-D tensors of equal length. target_avg_bits: b-bar, the
    desired mean bit-width across all units. eps_fn(sigma, b) -> distortion,
    defaulting to the Bennett-curve stand-in (see module docstring).

    Implementation note (not in the paper, spec Sec 10): when
    target_avg_bits == 0, the paper's relative-tolerance check
    |mean_b - target| / target is undefined (division by zero). This is
    special-cased to converge as soon as mean_b <= abs_tol_zero, returning
    all-zero bit-widths.

    Returns (b_star, info) where info has keys: lambda_, iters, converged,
    trace (list of dicts with keys it, lambda_lo, lambda_hi, lambda_, mean_b).
    """
    if w.shape != sigma.shape:
        raise ValueError(f"w and sigma must have the same shape, got {w.shape} vs {sigma.shape}")

    bit_options = torch.tensor(bit_widths, dtype=torch.float32)
    n_units = w.shape[0]
    n_options = bit_options.shape[0]

    lambda_lo = 0.0
    lambda_hi = max(w.max().item(), 1.0)

    trace = []
    b_star = torch.zeros(n_units, dtype=torch.float32)
    lambda_ = 0.0
    converged = False
    iters_used = 0

    for it in range(1, max_iter + 1):
        lambda_ = (lambda_lo + lambda_hi) / 2.0
        # cost[u, k] = w_u * eps(sigma_u, bit_options[k]) + lambda * bit_options[k]
        distortion = eps_fn(sigma.unsqueeze(1), bit_options.unsqueeze(0))  # (n_units, n_options)
        cost = w.unsqueeze(1) * distortion + lambda_ * bit_options.unsqueeze(0)
        best_idx = torch.argmin(cost, dim=1)
        b_star = bit_options[best_idx]

        mean_b = b_star.mean().item()
        iters_used = it

        if target_avg_bits <= 0.0:
            converged = mean_b <= abs_tol_zero
        else:
            converged = abs(mean_b - target_avg_bits) / target_avg_bits < tol

        trace.append(
            {"it": it, "lambda_lo": lambda_lo, "lambda_hi": lambda_hi, "lambda_": lambda_, "mean_b": mean_b}
        )

        if converged:
            break
        if mean_b > target_avg_bits:
            lambda_lo = lambda_
        else:
            lambda_hi = lambda_

    return b_star.long(), {
        "lambda_": lambda_,
        "iters": iters_used,
        "converged": converged,
        "trace": trace,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rdkv && pytest tests/test_mckp.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add rdkv/rdkv/mckp.py rdkv/tests/test_mckp.py
git commit -m "rdkv: implement Algorithm 2 MCKP Lagrangian bisection"
```

---

## Task 4: Per-unit weight computation (Propositions 3.1, 3.2)

**Files:**
- Create: `rdkv/rdkv/weights.py`
- Test: `rdkv/tests/test_weights.py`

**Interfaces:**
- Consumes: nothing from other rdkv modules.
- Produces: `token_weight_v(attn: torch.Tensor) -> torch.Tensor` (Eq. 2, `w_t`), `channel_weight_k(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor` (Eq. 4, `w_c`), `bennett_sigma(dynamic_range: torch.Tensor) -> torch.Tensor` (§4, `σ_u = R_u / (2√3)`), `total_variation_after_eviction(attn_row: torch.Tensor, evict_idx: int) -> float` (Prop 3.1 verification helper, used only in tests to reproduce the spec's hand-worked TV check).

- [ ] **Step 1: Write the failing tests**

```python
# rdkv/tests/test_weights.py
import math

import torch

from rdkv.weights import bennett_sigma, channel_weight_k, token_weight_v, total_variation_after_eviction


def test_token_weight_is_sum_over_queries():
    # Eq. (2): w_t := sum_tau a_{tau,t}. attn shape (n_queries, T).
    attn = torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.2, 0.2, 0.3, 0.3]])
    w_t = token_weight_v(attn)
    expected = torch.tensor([0.7, 0.5, 0.45, 0.35])
    assert torch.allclose(w_t, expected)


def test_token_weight_single_query_collapses_to_attn_row():
    # Spec Sec 11: with only one query, w_t = a_{tau,t} directly.
    attn = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    w_t = token_weight_v(attn)
    assert torch.allclose(w_t, attn[0])


def test_proposition_3_1_total_variation_equals_evicted_attention_mass():
    # Spec Sec 11 worked example: evicting token 1 (a=0.5) from
    # a_tau = [0.5, 0.3, 0.15, 0.05] gives TV distance exactly 0.5.
    a_tau = torch.tensor([0.5, 0.3, 0.15, 0.05])
    tv = total_variation_after_eviction(a_tau, evict_idx=0)
    assert math.isclose(tv, 0.5, rel_tol=1e-6)
    assert math.isclose(tv, a_tau[0].item(), rel_tol=1e-6)


def test_channel_weight_matches_worked_example():
    # Spec Sec 11: d=2, T=4.
    q = torch.tensor([[1.0, 0.3], [0.0, 0.9], [1.0, 0.2], [0.5, 0.1]])  # (T, d)
    k = torch.tensor([[0.8, 0.5], [0.6, 0.5], [0.4, 0.5], [0.2, 0.5]])  # (T, d)
    w_c = channel_weight_k(q, k)
    expected = torch.tensor([1.1619, 0.6893])
    assert torch.allclose(w_c, expected, atol=1e-3)


def test_bennett_sigma_formula():
    # sigma_u := R_u / (2*sqrt(3))
    dynamic_range = torch.tensor([2.0 * 2 * math.sqrt(3), 6.0])
    sigma = bennett_sigma(dynamic_range)
    expected = torch.tensor([2.0, 6.0 / (2 * math.sqrt(3))])
    assert torch.allclose(sigma, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rdkv && pytest tests/test_weights.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rdkv.weights'`

- [ ] **Step 3: Write the implementation**

```python
# rdkv/rdkv/weights.py
"""Propositions 3.1, 3.2, and Bennett's approximation (spec Sec 2-4):
per-unit weight computation for the V-cache token weight w_t, K-cache
channel weight w_c, and quantization hardness sigma_u.
"""

import math

import torch


def token_weight_v(attn: torch.Tensor) -> torch.Tensor:
    """Eq. (2): w_t := sum_tau a_{tau,t}.

    attn: (n_queries, T) attention weights (post-softmax), one row per
    query tau, one column per V-cache token t. Returns a (T,) tensor.
    With a single query (n_queries == 1), this collapses to w_t = a_{tau,t}
    directly (spec Sec 11).
    """
    return attn.sum(dim=0)


def total_variation_after_eviction(attn_row: torch.Tensor, evict_idx: int) -> float:
    """Verifies Proposition 3.1: TV(a_tau, a_hat_tau) == a_{tau,t} exactly,
    for a single query's attention row, when token evict_idx is evicted
    and the rest renormalize per Eq. (1).

    This is a verification helper (used to reproduce the spec Sec 11 hand
    check), not part of the production weight-computation path -- w_t
    (token_weight_v) already IS the TV distance per unit, by the theorem.
    """
    evicted_mass = attn_row[evict_idx].item()
    renorm = attn_row.clone()
    renorm[evict_idx] = 0.0
    renorm = renorm / (1.0 - evicted_mass)
    tv = 0.5 * torch.abs(attn_row - renorm).sum().item()
    return tv


def channel_weight_k(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Eq. (4): w_c := (1/sqrt(d)) * ||Q[:,c]||_2 * ||K[:,c]||_2.

    q, k: (T, d) query and key matrices for one head. Returns a (d,)
    tensor, one weight per channel.
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} vs {k.shape}")
    d = q.shape[1]
    q_norms = torch.norm(q, dim=0)  # (d,)
    k_norms = torch.norm(k, dim=0)  # (d,)
    return (q_norms * k_norms) / math.sqrt(d)


def bennett_sigma(dynamic_range: torch.Tensor) -> torch.Tensor:
    """sigma_u := R_u / (2*sqrt(3)) (spec Sec 4, Bennett's high-rate
    approximation). dynamic_range is R_u, any shape."""
    return dynamic_range / (2.0 * math.sqrt(3.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rdkv && pytest tests/test_weights.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add rdkv/rdkv/weights.py rdkv/tests/test_weights.py
git commit -m "rdkv: implement Prop 3.1/3.2 weight computation and Bennett sigma"
```

---

## Task 5: Three-stage allocation pipeline (§7, Algorithm 1 Stages 1-3)

**Files:**
- Create: `rdkv/rdkv/pipeline.py`
- Modify: `rdkv/rdkv/__init__.py`
- Test: `rdkv/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `token_weight_v`, `channel_weight_k`, `bennett_sigma` from `rdkv.weights`; `mckp_bisect` from `rdkv.mckp`.
- Produces: `RDKVAllocator` class with method `allocate(attn: torch.Tensor, q: torch.Tensor, k: torch.Tensor, b_tok: float, mckp_kwargs: dict | None = None) -> AllocationResult`, and a `AllocationResult` dataclass with fields `b_v: torch.Tensor` (per-token V bit-widths, shape `(T,)`), `b_k: torch.Tensor` (per-channel K bit-widths, shape `(d,)`), `kept_tokens: torch.Tensor` (long tensor of indices where `b_v > 0`), `w_t: torch.Tensor`, `w_c: torch.Tensor` (the raw weights, exposed for inspection/plotting).

- [ ] **Step 1: Write the failing tests**

```python
# rdkv/tests/test_pipeline.py
import torch

from rdkv.pipeline import RDKVAllocator


def test_allocate_returns_correct_shapes():
    torch.manual_seed(0)
    T, d = 16, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result = allocator.allocate(attn, q, k, b_tok=4.0)

    assert result.b_v.shape == (T,)
    assert result.b_k.shape == (d,)
    assert result.w_t.shape == (T,)
    assert result.w_c.shape == (d,)


def test_kept_tokens_matches_nonzero_b_v():
    torch.manual_seed(1)
    T, d = 20, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result = allocator.allocate(attn, q, k, b_tok=2.0)

    expected_kept = torch.nonzero(result.b_v > 0, as_tuple=True)[0]
    assert torch.equal(result.kept_tokens, expected_kept)


def test_v_bit_widths_are_from_hardware_set():
    torch.manual_seed(2)
    T, d = 16, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result = allocator.allocate(attn, q, k, b_tok=3.0)

    assert set(result.b_v.tolist()).issubset({0, 2, 4, 8, 16})
    assert set(result.b_k.tolist()).issubset({0, 2, 4, 8, 16})


def test_tighter_budget_evicts_more_tokens():
    torch.manual_seed(3)
    T, d = 32, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result_loose = allocator.allocate(attn, q, k, b_tok=8.0)
    result_tight = allocator.allocate(attn, q, k, b_tok=0.5)

    assert len(result_tight.kept_tokens) <= len(result_loose.kept_tokens)


def test_k_allocation_uses_only_kept_token_count_as_denominator():
    # Spec Sec 7 Stage 3: B_bar_K := B_K / |T_kept|. If we force heavy V
    # eviction (tiny b_tok), the K budget denominator shrinks, so each
    # surviving channel should tend to get a higher or equal average
    # bit-width than under a looser V budget with the same B_K.
    torch.manual_seed(4)
    T, d = 32, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result_tight_v = allocator.allocate(attn, q, k, b_tok=0.5)
    result_loose_v = allocator.allocate(attn, q, k, b_tok=8.0)

    # Tighter V eviction leaves fewer kept tokens -> K's per-channel budget
    # denominator (|T_kept|) shrinks -> average K bit-width should not decrease.
    assert result_tight_v.b_k.float().mean().item() >= result_loose_v.b_k.float().mean().item() - 1e-6


def test_all_tokens_evicted_zeros_out_k_allocation_gracefully():
    # Degenerate case: an extremely tight budget evicts every V token,
    # so |T_kept| == 0 and Stage 3's denominator would divide by zero.
    # The pipeline must handle this without raising.
    torch.manual_seed(5)
    T, d = 8, 4
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result = allocator.allocate(attn, q, k, b_tok=1e-6)

    assert result.b_v.sum().item() >= 0  # no exception; near-zero budget mostly evicts
    assert result.b_k.shape == (d,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rdkv && pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rdkv.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# rdkv/rdkv/pipeline.py
"""Sec 7 / Algorithm 1 Stages 1-3 (spec): the three-stage allocation
pipeline, run once per layer-head pair immediately after prefill.

Stage 4 (TriZone packing, Algorithm 1's final stage) is out of scope for
this phase -- see spec Sec 14, decision 1. This module stops at producing
the per-token V bit-widths and per-channel K bit-widths; packing them into
TriZone storage is Phase 2.
"""

from dataclasses import dataclass

import torch

from .mckp import mckp_bisect
from .weights import bennett_sigma, channel_weight_k, token_weight_v


@dataclass
class AllocationResult:
    """Output of RDKVAllocator.allocate for one (layer, head) pair."""

    b_v: torch.Tensor  # (T,) per-token V bit-widths, hardware set {0,2,4,8,16}
    b_k: torch.Tensor  # (d,) per-channel K bit-widths, hardware set {0,2,4,8,16}
    kept_tokens: torch.Tensor  # long tensor of token indices where b_v > 0
    w_t: torch.Tensor  # (T,) raw token weights (Eq. 2)
    w_c: torch.Tensor  # (d,) raw channel weights (Eq. 4)


class RDKVAllocator:
    """Orchestrates Stage 1 (weighting) -> Stage 2 (V allocation) ->
    Stage 3 (K allocation on kept tokens only), per spec Sec 7.

    Uses a fixed sigma_u = 1 for all units by default (uniform dynamic
    range assumption) since Phase 1 has no real per-unit dynamic-range
    estimation wired in yet; pass sigma_v / sigma_k to override.
    """

    def allocate(
        self,
        attn: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        b_tok: float,
        sigma_v: torch.Tensor | None = None,
        sigma_k: torch.Tensor | None = None,
        mckp_kwargs: dict | None = None,
    ) -> AllocationResult:
        """attn: (n_queries, T) post-softmax attention weights. q, k: (T, d).
        b_tok: per-head budget in FP16-equivalent tokens (spec Sec 7).
        """
        mckp_kwargs = mckp_kwargs or {}
        T = attn.shape[1]
        d = q.shape[1]

        # Stage 1: weight computation
        w_t = token_weight_v(attn)
        w_c = channel_weight_k(q, k)
        if sigma_v is None:
            sigma_v = torch.ones_like(w_t)
        if sigma_k is None:
            sigma_k = torch.ones_like(w_c)

        # Stage 2: V-side token allocation. B_head = 2*b_tok*d*16 (Sec 7);
        # B_V = B_K = B_head/2; B_bar_V = B_V/d in summed-bit-width units.
        b_head = 2.0 * b_tok * d * 16.0
        b_v_budget = b_head / 2.0
        b_v_bar = b_v_budget / d
        target_avg_v = b_v_bar / T
        b_v, _ = mckp_bisect(w_t, sigma_v, target_avg_bits=target_avg_v, **mckp_kwargs)
        kept_tokens = torch.nonzero(b_v > 0, as_tuple=True)[0]

        # Stage 3: K-side channel allocation, denominator rescaled by
        # |T_kept| (Sec 7). If every token was evicted, there is nothing
        # left to allocate K bits for; return an all-zero K allocation
        # rather than dividing by zero.
        n_kept = kept_tokens.shape[0]
        if n_kept == 0:
            b_k = torch.zeros(d, dtype=torch.long)
        else:
            b_k_budget = b_head / 2.0
            k_avg = b_k_budget / (n_kept * d)
            b_k, _ = mckp_bisect(w_c, sigma_k, target_avg_bits=k_avg, **mckp_kwargs)

        return AllocationResult(b_v=b_v, b_k=b_k, kept_tokens=kept_tokens, w_t=w_t, w_c=w_c)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rdkv && pytest tests/test_pipeline.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Update `rdkv/rdkv/__init__.py` with the full public API**

```python
# rdkv/rdkv/__init__.py
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
```

- [ ] **Step 6: Verify the public API imports cleanly**

Run: `cd rdkv && python -c "from rdkv import RDKVAllocator, continuous_waterfill, mckp_bisect; print('ok')"`
Expected: prints `ok` with no errors.

- [ ] **Step 7: Run the full test suite**

Run: `cd rdkv && pytest tests/ -v`
Expected: all tests across all four test files pass.

- [ ] **Step 8: Commit**

```bash
git add rdkv/rdkv/pipeline.py rdkv/rdkv/__init__.py rdkv/tests/test_pipeline.py
git commit -m "rdkv: implement Sec 7 three-stage allocation pipeline, finalize public API"
```

---

## Task 6: Real-model example script

**Files:**
- Create: `rdkv/examples/allocation_stats.py`

**Interfaces:**
- Consumes: `rdkv.RDKVAllocator` (public API from Task 5).
- Produces: a standalone script, not imported by the test suite (matches `turbo-quant/examples/run_benchmark.py`'s role — runnable manually, exercised by a thin test only if it can run on CPU with a tiny model within test-suite time budget).

- [ ] **Step 1: Write `rdkv/examples/allocation_stats.py`**

```python
"""Runs RDKV's Phase 1 allocation pipeline against a real HF model's
prefill attention, reporting per-layer eviction/bit-width statistics.

This is a demonstration and sanity-check script, not a production
compressed-cache integration (Phase 2 -- TriZone packing and the fused
kernel -- would be needed for that). Requires the `examples` extra:

    pip install -e ".[examples]"

Usage:
    python examples/allocation_stats.py --model sshleifer/tiny-gpt2 --b-tok 4.0
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rdkv import RDKVAllocator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--b-tok", type=float, default=4.0, help="per-head budget in FP16-equivalent tokens")
    parser.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog. " * 8,
        help="prefill text to build the attention/Q/K statistics from",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager", output_attentions=True)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (batch, n_heads, T, T) per layer
    allocator = RDKVAllocator()

    print(f"model={args.model} n_layers={len(attentions)} b_tok={args.b_tok}")
    for layer_idx, layer_attn in enumerate(attentions):
        n_heads = layer_attn.shape[1]
        T = layer_attn.shape[-1]
        d = model.config.hidden_size // model.config.n_head if hasattr(model.config, "n_head") else 64

        head0_attn = layer_attn[0, 0]  # (T, T): row=query, col=key/token
        q = torch.randn(T, d)  # placeholder Q/K -- real per-head Q/K extraction
        k = torch.randn(T, d)  # requires model-specific hook, left for a follow-up script

        result = allocator.allocate(head0_attn, q, k, b_tok=args.b_tok)
        n_kept = result.kept_tokens.shape[0]
        mean_b_v = result.b_v.float().mean().item()
        mean_b_k = result.b_k.float().mean().item()
        print(
            f"  layer {layer_idx:2d}: T={T:4d} kept={n_kept:4d} "
            f"({100 * n_kept / T:5.1f}%) mean_b_v={mean_b_v:5.2f} mean_b_k={mean_b_k:5.2f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs (manual, not part of automated suite)**

Run: `cd rdkv && pip install -e ".[examples]" && python examples/allocation_stats.py --model sshleifer/tiny-gpt2`
Expected: prints one line per layer with kept-token percentage and mean bit-widths, no exceptions.

Note the placeholder Q/K extraction in the script's docstring/comment — real per-head Q/K requires a model-specific forward hook (attention module varies by architecture), which is flagged here as a follow-up rather than solved in this task, since it doesn't block validating the allocator's own math (already covered by Tasks 2-5's tests against synthetic and worked-example data).

- [ ] **Step 3: Commit**

```bash
git add rdkv/examples/allocation_stats.py
git commit -m "rdkv: add real-model example script for allocation stats"
```

---

## Task 7: Update root README

**Files:**
- Modify: `README.md:52-58` (the RDKV entry added previously, marked "in progress")

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Update the RDKV section to reflect Phase 1 completion**

Replace the existing "in progress" RDKV entry (added in an earlier session) with:

```markdown
### RDKV -- Joint Eviction and Quantization of the KV Cache ([`rdkv/`](rdkv/))

**Paper**: [Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache](https://arxiv.org/abs/2605.08317) (arXiv:2605.08317)

RDKV treats KV cache eviction and quantization as the same operation — bit-width assignment — evaluated at different depths (0 bits = evicted). Phase 1 (implemented): closed-form continuous water-filling (Theorem 3.3), discrete MCKP bit allocation via Lagrangian bisection (Algorithm 2), per-unit weight computation (Propositions 3.1/3.2), and the three-stage allocation pipeline (Algorithm 1, Stages 1-3). Pure PyTorch, no custom GPU kernel.

**Not yet implemented (Phase 2)**: TriZone packing and the fused-dequantization attention kernel (Algorithm 1, Stage 4).

**Disclosed approximation**: Phase 1 uses the analytic Bennett curve `σ_u · 2^(−b)` as a stand-in for the paper's empirically-calibrated per-coordinate distortion table (Appendix B); see [`rdkv/README.md`](rdkv/README.md).

See [`rdkv/rdkv-primer.html`](rdkv/rdkv-primer.html) for the math derivation, [`docs/superpowers/specs/2026-08-31-rdkv-design.md`](docs/superpowers/specs/2026-08-31-rdkv-design.md) for the full spec, and [`rdkv/README.md`](rdkv/README.md) for install/test instructions.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mark RDKV Phase 1 as implemented in root README"
```

---

## Task 8: Full suite verification

**Files:**
- None (verification only).

- [ ] **Step 1: Run the complete rdkv test suite**

Run: `cd rdkv && pytest tests/ -v`
Expected: every test in `test_waterfilling.py`, `test_mckp.py`, `test_weights.py`, `test_pipeline.py` passes.

- [ ] **Step 2: Verify the package's public API matches the spec's terminology**

Run: `cd rdkv && python -c "
from rdkv import RDKVAllocator, continuous_waterfill, mckp_bisect, token_weight_v, channel_weight_k, bennett_sigma, bennett_distortion, AllocationResult
print('all public symbols import cleanly')
"`
Expected: prints the confirmation line, no `ImportError`.

- [ ] **Step 3: Confirm no GPU kernel / Triton dependency leaked in**

Run: `cd rdkv && python -c "import rdkv; import sys; assert 'triton' not in sys.modules; print('no triton import')"`
Expected: prints `no triton import` (Phase 1 must have zero Triton dependency, per Global Constraints).

This task has no commit — it's a checkpoint confirming Tasks 1-7 together satisfy Phase 1 of the plan's Goal statement before starting Phase 2.

---

# Phase 2: TriZone packing and fused decode (spec §8/§9, Algorithm 1 Stage 4)

Phase 2 takes a Phase 1 `AllocationResult` (which token/channel gets which bit-width) and actually does something with it: pack the cache into the three storage zones spec §8 defines, and decode from that packed representation without ever materializing a dequantized FP16 tile. Tasks 9–11 are pure PyTorch (packing is bookkeeping; the *reference* decode implementation is ordinary tensor ops) and run on CPU or GPU. Tasks 12–13 add the GPU-only Triton kernel that fuses K dequantization into the attention score computation — the actual performance claim of the paper (§8's core mechanism) — verified for exact parity against Task 11's reference before any perf work happens. Task 14 is the same kind of full-suite checkpoint as Task 8, now covering both phases.

---

## Task 9: Kernel extra and CUDA/Triton guard

**Files:**
- Modify: `rdkv/pyproject.toml`
- Create: `rdkv/rdkv/kernel/__init__.py`
- Create: `rdkv/rdkv/kernel/_require.py`
- Test: `rdkv/tests/test_kernel_require.py`

**Interfaces:**
- Produces: `require_kernel_backend(device: str) -> None`, raising `RuntimeError` with an install hint when `device != "cuda"` or `triton`/`triton-windows` is not installed. Verbatim pattern from `turbo-quant/turboquant/kernel/_require.py`.

- [ ] **Step 1: Add the `kernel` extra to `rdkv/pyproject.toml`**

Add this section (after `[project.optional-dependencies]`'s existing `examples` and `test` keys):

```toml
# triton has no Windows wheels on PyPI; triton-windows is a Windows-wheel fork with the same API.
kernel = [
    "triton>=3.0; sys_platform == 'linux'",
    "triton-windows>=3.7.0; sys_platform == 'win32'",
]
```

- [ ] **Step 2: Write the failing test**

```python
# rdkv/tests/test_kernel_require.py
import pytest

from rdkv.kernel._require import require_kernel_backend


def test_raises_on_non_cuda_device():
    with pytest.raises(RuntimeError, match="requires device='cuda'"):
        require_kernel_backend("cpu")


def test_raises_with_install_hint_when_triton_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "triton", None)  # force ImportError on `import triton`
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("no triton")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="requires the 'triton' package"):
        require_kernel_backend("cuda")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd rdkv && pytest tests/test_kernel_require.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rdkv.kernel'`

- [ ] **Step 4: Write `rdkv/rdkv/kernel/__init__.py`**

```python
"""GPU-only, Triton-based fused decode kernel for the RDKV TriZone packed
cache (spec Sec 8/9, Algorithm 1 Stage 4).

This subpackage is imported lazily by rdkv.decode only when
backend="kernel" is requested. It has no import-time side effects that
would require triton to be installed to use the default native backend.
"""
```

- [ ] **Step 5: Write `rdkv/rdkv/kernel/_require.py`**

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
            "`pip install rdkv[kernel]`."
        ) from exc
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd rdkv && pytest tests/test_kernel_require.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add rdkv/pyproject.toml rdkv/rdkv/kernel/__init__.py rdkv/rdkv/kernel/_require.py rdkv/tests/test_kernel_require.py
git commit -m "rdkv: add kernel extra and CUDA/Triton backend guard"
```

---

## Task 10: TriZone packing (Algorithm 1 Stage 4)

**Files:**
- Create: `rdkv/rdkv/trizone.py`
- Test: `rdkv/tests/test_trizone.py`

**Interfaces:**
- Consumes: `AllocationResult` from `rdkv.pipeline` (Task 5's `b_v`, `b_k`, `kept_tokens` fields).
- Produces: `pack_trizone(k: torch.Tensor, v: torch.Tensor, allocation: AllocationResult) -> PackedCache`, a `PackedCache` dataclass with fields `zone_a_v: dict[int, torch.Tensor]` (bit-width → packed V sub-segment, keys from `{2,4,8}`), `zone_a_k: torch.Tensor` (retained K rows, permuted to match `b_k`-sorted channel order), `zone_b_v: torch.Tensor` (FP16-retained V rows, `b_v==16`), `zone_b_token_idx: torch.Tensor`, `k_channel_perm: torch.Tensor` (the channel permutation applied, needed to permute incoming queries consistently — spec §9's "permute q to match"), `k_scale: torch.Tensor`, `k_zero_point: torch.Tensor` (per-channel affine quantization params for Zone A's K rows, needed by Task 11's dequantization).

- [ ] **Step 1: Write the failing tests**

```python
# rdkv/tests/test_trizone.py
import torch

from rdkv.pipeline import AllocationResult
from rdkv.trizone import pack_trizone


def _make_allocation(b_v, b_k):
    b_v = torch.tensor(b_v, dtype=torch.long)
    b_k = torch.tensor(b_k, dtype=torch.long)
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    return AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32),
        w_c=torch.ones_like(b_k, dtype=torch.float32),
    )


def test_zone_b_holds_exactly_the_16bit_tokens():
    T, d = 6, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 0, 4, 16, 2], b_k=[8, 4, 2, 16])

    packed = pack_trizone(k, v, allocation)

    expected_zone_b_idx = torch.tensor([0, 4])
    assert torch.equal(packed.zone_b_token_idx, expected_zone_b_idx)
    assert packed.zone_b_v.shape == (2, d)
    assert torch.allclose(packed.zone_b_v, v[expected_zone_b_idx])


def test_zone_a_v_subsegments_partition_the_non16bit_kept_tokens():
    T, d = 6, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 0, 4, 16, 2], b_k=[8, 4, 2, 16])

    packed = pack_trizone(k, v, allocation)

    # tokens 1 (b=8), 3 (b=4), 5 (b=2) are kept and not 16-bit -> Zone A(V)
    all_zone_a_tokens = set()
    for bit_width, segment in packed.zone_a_v.items():
        assert bit_width in (2, 4, 8)
        all_zone_a_tokens.update(range(segment.shape[0]))  # just shape sanity below
    assert set(packed.zone_a_v.keys()) == {2, 4, 8}
    assert packed.zone_a_v[8].shape[0] == 1
    assert packed.zone_a_v[4].shape[0] == 1
    assert packed.zone_a_v[2].shape[0] == 1


def test_zone_a_k_has_one_row_per_kept_token():
    T, d = 6, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 0, 4, 16, 2], b_k=[8, 4, 2, 16])

    packed = pack_trizone(k, v, allocation)

    n_kept = allocation.kept_tokens.shape[0]
    assert packed.zone_a_k.shape == (n_kept, d)


def test_channel_permutation_sorts_by_bit_width():
    T, d = 4, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 4, 2], b_k=[2, 16, 4, 8])

    packed = pack_trizone(k, v, allocation)

    sorted_b_k = allocation.b_k[packed.k_channel_perm]
    assert torch.equal(sorted_b_k, torch.sort(allocation.b_k).values)


def test_all_evicted_v_yields_empty_zones():
    T, d = 4, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[0, 0, 0, 0], b_k=[8, 4, 2, 16])

    packed = pack_trizone(k, v, allocation)

    assert packed.zone_b_v.shape[0] == 0
    assert packed.zone_a_k.shape[0] == 0
    for segment in packed.zone_a_v.values():
        assert segment.shape[0] == 0


def test_zone_a_k_dequantizes_back_close_to_original_within_bit_budget():
    # Sanity check: k_scale/k_zero_point should let us approximately
    # reconstruct the original (permuted) K rows for the kept tokens.
    torch.manual_seed(0)
    T, d = 10, 8
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 4, 2, 0, 8, 4, 2, 16, 8], b_k=[8, 4, 16, 2, 8, 4, 16, 2])

    packed = pack_trizone(k, v, allocation)

    dequant = packed.zone_a_k * packed.k_scale + packed.k_zero_point
    original_permuted = k[allocation.kept_tokens][:, packed.k_channel_perm]
    # Loose tolerance -- this only checks the affine params are self-consistent,
    # not tight quantization error (that's covered once decode.py exists in Task 11).
    assert torch.allclose(dequant, original_permuted, atol=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rdkv && pytest tests/test_trizone.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rdkv.trizone'`

- [ ] **Step 3: Write the implementation**

```python
# rdkv/rdkv/trizone.py
"""Algorithm 1 Stage 4 (spec Sec 8/9): TriZone packing.

Packs a prefill K/V cache plus an Sec 7 AllocationResult into three storage
zones:

  Zone A -- packed, quantized (old cache): retained K rows from
    T_kept, plus V rows with b_v in {2,4,8}, grouped into uniform-bit
    sub-segments.
  Zone B -- FP16, retained: V rows with b_v == 16. Their K rows still
    live in Zone A (K bit-widths follow the independent per-channel
    allocation, not b_v).
  Zone C -- FP16, new decode tokens -- NOT produced here. Zone C grows
    one entry per decode step and is owned by the decode loop
    (rdkv.decode), not by this one-shot post-prefill packing step.

This is memory-layout bookkeeping: it runs in plain PyTorch on CPU or GPU
and requires no Triton kernel.
"""

from dataclasses import dataclass, field

import torch

from .pipeline import AllocationResult

_ZONE_A_V_BITS = (2, 4, 8)


@dataclass
class PackedCache:
    zone_a_v: dict[int, torch.Tensor]  # bit_width -> (n_tokens_at_this_bit, d) quantized V rows
    zone_a_k: torch.Tensor  # (n_kept, d) quantized K rows, channel-permuted
    zone_b_v: torch.Tensor  # (n_16bit, d) FP16 V rows
    zone_b_token_idx: torch.Tensor  # original token indices of Zone B rows
    k_channel_perm: torch.Tensor  # (d,) permutation applied to K's channel axis
    k_scale: torch.Tensor  # (d,) per-channel affine quantization scale s_c
    k_zero_point: torch.Tensor  # (d,) per-channel affine quantization zero point z_c


def _affine_quantize_channel(col: torch.Tensor, bits: int) -> tuple[torch.Tensor, float, float]:
    """Per-channel affine (asymmetric) quantization: k_hat = s*(k_tilde - z).
    Returns (quantized_int_tensor, scale, zero_point) such that
    dequant = quantized * scale + zero_point approximately reconstructs col.
    """
    if bits <= 0 or col.numel() == 0:
        return torch.zeros_like(col, dtype=torch.int64), 1.0, 0.0
    lo, hi = col.min().item(), col.max().item()
    if hi - lo < 1e-12:
        return torch.zeros_like(col, dtype=torch.int64), 1.0, lo
    n_levels = 2**bits
    scale = (hi - lo) / (n_levels - 1)
    zero_point = lo
    quantized = torch.clamp(torch.round((col - zero_point) / scale), 0, n_levels - 1).long()
    return quantized, scale, zero_point


def pack_trizone(k: torch.Tensor, v: torch.Tensor, allocation: AllocationResult) -> PackedCache:
    """k, v: (T, d) prefill K/V cache for one (layer, head) pair."""
    d = k.shape[1]
    kept = allocation.kept_tokens
    b_v_kept = allocation.b_v[kept]

    # Zone B: V rows with b_v == 16.
    zone_b_mask = b_v_kept == 16
    zone_b_token_idx = kept[zone_b_mask]
    zone_b_v = v[zone_b_token_idx]

    # Zone A(V): kept, non-16-bit V rows, split into per-bit-width sub-segments.
    zone_a_v: dict[int, torch.Tensor] = {}
    for bits in _ZONE_A_V_BITS:
        mask = b_v_kept == bits
        idx = kept[mask]
        zone_a_v[bits] = v[idx]

    # Zone A(K): every kept token's K row, with channels permuted by b_k ascending
    # (spec Sec 9: "sort channels by b_c^K into segments; permute q to match").
    k_channel_perm = torch.argsort(allocation.b_k)
    k_kept_permuted = k[kept][:, k_channel_perm]  # (n_kept, d)

    b_k_sorted = allocation.b_k[k_channel_perm]
    k_scale = torch.ones(d)
    k_zero_point = torch.zeros(d)
    zone_a_k_int = torch.zeros_like(k_kept_permuted, dtype=torch.int64)
    for c in range(d):
        bits_c = int(b_k_sorted[c].item())
        quantized, scale, zero_point = _affine_quantize_channel(k_kept_permuted[:, c], bits_c)
        zone_a_k_int[:, c] = quantized
        k_scale[c] = scale
        k_zero_point[c] = zero_point

    return PackedCache(
        zone_a_v=zone_a_v,
        zone_a_k=zone_a_k_int,
        zone_b_v=zone_b_v,
        zone_b_token_idx=zone_b_token_idx,
        k_channel_perm=k_channel_perm,
        k_scale=k_scale,
        k_zero_point=k_zero_point,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rdkv && pytest tests/test_trizone.py -v`
Expected: PASS (6 passed)

Note: `test_zone_a_k_dequantizes_back_close_to_original_within_bit_budget` uses `atol=0.5`, loose on purpose — it's checking that `k_scale`/`k_zero_point` are self-consistent affine parameters, not measuring tight quantization error (some channels in the test's synthetic allocation get very few bits). If this test is flaky across `torch.manual_seed` values, widen the tolerance further rather than changing the seed to hide it — the point is structural self-consistency, not a numerical performance target.

- [ ] **Step 5: Commit**

```bash
git add rdkv/rdkv/trizone.py rdkv/tests/test_trizone.py
git commit -m "rdkv: implement Algorithm 1 Stage 4 TriZone packing"
```

---

## Task 11: Native packed-decode reference (Eq. 7)

**Files:**
- Create: `rdkv/rdkv/decode.py`
- Test: `rdkv/tests/test_decode.py`

**Interfaces:**
- Consumes: `PackedCache` from `rdkv.trizone` (Task 10).
- Produces: `packed_decode(packed: PackedCache, q_tau: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, sqrt_d: float) -> torch.Tensor` returning `o_tau`, the attention output for one decode step, computed via Eq. (7)'s three-way sum. This is the *correctness reference* — plain PyTorch ops, dequantizing Zone A's K into a real tensor before the matmul (unlike the fused kernel in Task 12, which must not do that). Task 12's kernel is checked against this function's output, not the other way around.

- [ ] **Step 1: Write the failing tests**

```python
# rdkv/tests/test_decode.py
import math

import torch

from rdkv.pipeline import AllocationResult
from rdkv.trizone import pack_trizone
from rdkv.decode import packed_decode


def _full_precision_reference_output(k_all, v_all, q_tau, sqrt_d):
    """Unquantized ground truth: standard attention over every original
    (non-evicted) token plus the new decode token, no packing at all."""
    scores = (q_tau @ k_all.T) / sqrt_d
    weights = torch.softmax(scores, dim=-1)
    return weights @ v_all


def test_packed_decode_matches_full_precision_within_quantization_noise():
    torch.manual_seed(0)
    T, d = 12, 8
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    # Give every token a generous bit-width so quantization noise is small;
    # this test checks the *decomposition* is correct, not aggressive compression.
    b_v = torch.tensor([16, 8, 16, 8, 16, 8, 16, 8, 16, 8, 16, 8])
    b_k = torch.tensor([16, 8, 16, 8, 16, 8, 16, 8])
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    packed = pack_trizone(k, v, allocation)

    q_tau = torch.randn(d)
    k_new = torch.randn(1, d)
    v_new = torch.randn(1, d)
    sqrt_d = math.sqrt(d)

    output = packed_decode(packed, q_tau, k_new, v_new, sqrt_d)

    k_all = torch.cat([k[kept], k_new], dim=0)
    v_all = torch.cat([v[kept], v_new], dim=0)
    reference = _full_precision_reference_output(k_all, v_all, q_tau, sqrt_d)

    assert output.shape == (d,)
    # Loose tolerance: Zone A's K/V rows went through real quantization noise.
    assert torch.allclose(output, reference, atol=0.5)


def test_output_shape_is_head_dim():
    torch.manual_seed(1)
    T, d = 6, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    b_v = torch.tensor([16, 0, 8, 4, 16, 2])
    b_k = torch.tensor([8, 4, 16, 2])
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    packed = pack_trizone(k, v, allocation)

    q_tau = torch.randn(d)
    k_new = torch.randn(2, d)
    v_new = torch.randn(2, d)
    output = packed_decode(packed, q_tau, k_new, v_new, math.sqrt(d))
    assert output.shape == (d,)


def test_all_evicted_falls_back_to_new_tokens_only():
    torch.manual_seed(2)
    T, d = 4, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    b_v = torch.zeros(T, dtype=torch.long)
    b_k = torch.tensor([8, 4, 16, 2])
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    packed = pack_trizone(k, v, allocation)

    q_tau = torch.randn(d)
    k_new = torch.randn(3, d)
    v_new = torch.randn(3, d)
    output = packed_decode(packed, q_tau, k_new, v_new, math.sqrt(d))

    reference = _full_precision_reference_output(k_new, v_new, q_tau, math.sqrt(d))
    assert torch.allclose(output, reference, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rdkv && pytest tests/test_decode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rdkv.decode'`

- [ ] **Step 3: Write the implementation**

```python
# rdkv/rdkv/decode.py
"""Eq. (7) (spec Sec 8): packed-decode output decomposition.

    o_tau = sum_{t in T_kept \\ T_V16} a_{tau,t} * v_hat_t   (Zone A, quantized V)
          + sum_{t in T_V16}           a_{tau,t} * v_t       (Zone B, FP16 retained)
          + sum_{t in T_new}           a_{tau,t} * v_t       (Zone C, FP16 new)

This module is the NATIVE reference implementation: it dequantizes Zone A's
K rows into a real tensor before computing attention scores. This is
intentional here (correctness reference for Task 12's kernel to be checked
against) but is exactly what the fused kernel (rdkv.kernel.fused_decode)
must NOT do -- the whole point of Sec 8's algebraic rewrite is to never
materialize a dequantized FP16 K tile in the fast path.
"""

import torch

from .trizone import PackedCache


def packed_decode(
    packed: PackedCache,
    q_tau: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    sqrt_d: float,
) -> torch.Tensor:
    """q_tau: (d,) query for this decode step. k_new, v_new: (n_new, d) --
    Zone C, the new tokens generated since the last packing (n_new >= 1,
    the current step's own token at minimum). Returns o_tau: (d,).
    """
    d = q_tau.shape[0]
    n_kept = packed.zone_a_k.shape[0]

    # Reconstruct Zone A's K rows (native reference -- see module docstring
    # for why this is NOT how the fused kernel does it).
    if n_kept > 0:
        k_zone_a_dequant = packed.zone_a_k.float() * packed.k_scale + packed.k_zero_point
        # Undo the channel permutation applied at packing time so scores
        # align with q_tau's original channel order.
        inv_perm = torch.argsort(packed.k_channel_perm)
        k_zone_a = k_zone_a_dequant[:, inv_perm]
    else:
        k_zone_a = torch.empty(0, d)

    # Zone A's V rows: concatenate the {2,4,8}-bit sub-segments back in
    # kept-token order is not required for a decode-step sum (order-
    # independent softmax-weighted sum), so we just concatenate.
    zone_a_v_parts = [seg for seg in packed.zone_a_v.values() if seg.shape[0] > 0]
    v_zone_a = torch.cat(zone_a_v_parts, dim=0) if zone_a_v_parts else torch.empty(0, d)
    # k_zone_a rows correspond to ALL kept tokens (Zone A(K) covers every
    # kept token, per spec Sec 8's Zone A definition), while v_zone_a only
    # covers the non-16-bit subset -- so the K used for Zone A's V-weighted
    # sum must be restricted to the same non-16-bit token subset.
    n_16bit = packed.zone_b_v.shape[0]
    n_non16bit = n_kept - n_16bit
    # zone_a_k's row order is: all kept tokens in original packing order is
    # NOT guaranteed here since trizone.py builds zone_a_k from `kept`
    # directly (see Task 10) -- so we must select the same non-16-bit rows.
    # This selection mirrors trizone.py's zone_b_mask/zone_a_v construction.
    non16_selector = _non16bit_row_selector(packed)
    k_for_zone_a_v = k_zone_a[non16_selector]

    scores_parts = []
    values_parts = []

    if v_zone_a.shape[0] > 0:
        scores_a = (q_tau @ k_for_zone_a_v.T) / sqrt_d
        scores_parts.append(scores_a)
        values_parts.append(v_zone_a)

    if packed.zone_b_v.shape[0] > 0:
        k_zone_b = k_zone_a[~non16_selector]
        scores_b = (q_tau @ k_zone_b.T) / sqrt_d
        scores_parts.append(scores_b)
        values_parts.append(packed.zone_b_v)

    scores_c = (q_tau @ k_new.T) / sqrt_d
    scores_parts.append(scores_c)
    values_parts.append(v_new)

    all_scores = torch.cat(scores_parts, dim=0)
    all_values = torch.cat(values_parts, dim=0)
    all_weights = torch.softmax(all_scores, dim=0)

    return all_weights @ all_values


def _non16bit_row_selector(packed: PackedCache) -> torch.Tensor:
    """Boolean mask over zone_a_k's rows selecting the non-16-bit-V kept
    tokens, matching the row order trizone.pack_trizone used when it built
    zone_a_k from `allocation.kept_tokens` directly."""
    n_kept = packed.zone_a_k.shape[0]
    n_16bit = packed.zone_b_v.shape[0]
    # zone_b_token_idx are the ORIGINAL token indices of the 16-bit rows;
    # kept_tokens (implicitly, via zone_a_k's construction order in
    # trizone.py) is sorted ascending, and zone_b_token_idx is a subset of
    # it in the same relative order -- so we can recover the selector by
    # position once we know which of zone_a_k's n_kept rows are 16-bit.
    # trizone.py does not currently expose the full kept-token order
    # alongside zone_a_k, so this helper recomputes it from what PackedCache
    # does expose: zone_b_token_idx's positions among the kept set.
    selector = torch.ones(n_kept, dtype=torch.bool)
    if n_16bit == 0:
        return selector
    # This relies on trizone.py having built zone_a_k in ascending
    # kept-token order (it does: `k[kept]` with `kept` from
    # AllocationResult.kept_tokens, which pipeline.py builds via
    # torch.nonzero, always ascending). zone_b_token_idx is therefore a
    # sorted subsequence of that same order.
    selector[: 0] = False  # no-op placeholder to keep structure explicit
    return selector
```

- [ ] **Step 4: Run tests and observe the shape/logic gap in `_non16bit_row_selector`**

Run: `cd rdkv && pytest tests/test_decode.py -v`

The `_non16bit_row_selector` sketch above is deliberately incomplete (it returns all-True, which is wrong whenever any 16-bit tokens exist) — implementing it correctly requires trizone.py to expose the boolean 16-bit mask over kept tokens, not just the 16-bit tokens' original indices. Do not paper over this with a placeholder; fix it at the source:

- [ ] **Step 5: Add a `zone_b_mask` field to `PackedCache` and have `pack_trizone` populate it**

Edit `rdkv/rdkv/trizone.py`:

```python
# In the PackedCache dataclass, add:
    zone_b_mask: torch.Tensor  # (n_kept,) bool, True where the kept token is 16-bit (Zone B)

# In pack_trizone, after computing zone_b_mask (already computed as the
# local variable `zone_b_mask` inside the function -- just add it to the
# returned PackedCache):
    return PackedCache(
        zone_a_v=zone_a_v,
        zone_a_k=zone_a_k_int,
        zone_b_v=zone_b_v,
        zone_b_token_idx=zone_b_token_idx,
        zone_b_mask=zone_b_mask,
        k_channel_perm=k_channel_perm,
        k_scale=k_scale,
        k_zero_point=k_zero_point,
    )
```

Then simplify `decode.py`'s helper to use it directly:

```python
def _non16bit_row_selector(packed: PackedCache) -> torch.Tensor:
    """Boolean mask over zone_a_k's rows selecting the non-16-bit-V kept
    tokens (Zone A's K rows cover every kept token; only the subset with
    b_v != 16 pairs with Zone A's V sub-segments)."""
    return ~packed.zone_b_mask
```

Remove the old placeholder `_non16bit_row_selector` body from Step 3 and replace it with this.

- [ ] **Step 6: Update `test_trizone.py`'s allocation-construction tests if they unpack `PackedCache` positionally**

Check `rdkv/tests/test_trizone.py` from Task 10 — all assertions there use keyword/attribute access (`packed.zone_b_v`, etc.), not positional unpacking, so no changes needed. Confirm by re-running:

Run: `cd rdkv && pytest tests/test_trizone.py -v`
Expected: still PASS (6 passed) — the new `zone_b_mask` field is additive.

- [ ] **Step 7: Run `test_decode.py` again to verify it now passes**

Run: `cd rdkv && pytest tests/test_decode.py -v`
Expected: PASS (3 passed)

- [ ] **Step 8: Commit**

```bash
git add rdkv/rdkv/trizone.py rdkv/rdkv/decode.py rdkv/tests/test_decode.py
git commit -m "rdkv: implement Eq. 7 native packed-decode reference"
```

---

## Task 12: Fused Triton decode kernel

**Files:**
- Create: `rdkv/rdkv/kernel/fused_decode.py`
- Modify: `rdkv/rdkv/decode.py`
- Test: `rdkv/tests/test_kernel_backend.py`

**Interfaces:**
- Consumes: `PackedCache` from `rdkv.trizone`; `packed_decode`'s native implementation from Task 11 as the correctness reference.
- Produces: `packed_decode(..., backend="kernel")` — extends Task 11's `packed_decode` with a `backend: str = "native"` parameter; when `"kernel"`, dispatches to `rdkv.kernel.fused_decode.fused_packed_decode`, which fuses Zone A's algebraic K dequantization (spec §8: `q_τᵀ k̂_t` rewritten as `Σ_c (s_c·q_{τ,c})·k̃_{t,c}` minus one per-query-head bias) directly into the score computation — no dequantized FP16 K tile is allocated for Zone A.

- [ ] **Step 1: Write the failing kernel-parity test**

```python
# rdkv/tests/test_kernel_backend.py
"""Correctness parity between the native and fused-kernel packed-decode
backends. Skipped entirely on machines without CUDA or without triton
installed -- the kernel backend is CUDA-only by design (spec Sec 8;
mirrors turbo-quant/tests/test_kernel_backend.py's pattern)."""

import math

import pytest
import torch

from rdkv.pipeline import AllocationResult
from rdkv.trizone import pack_trizone
from rdkv.decode import packed_decode

CUDA_AND_TRITON_AVAILABLE = torch.cuda.is_available()
if CUDA_AND_TRITON_AVAILABLE:
    try:
        import triton  # noqa: F401
    except ImportError:
        CUDA_AND_TRITON_AVAILABLE = False

requires_kernel_backend = pytest.mark.skipif(
    not CUDA_AND_TRITON_AVAILABLE, reason="kernel backend requires CUDA and triton"
)


def _random_packed_cache(T, d, device):
    torch.manual_seed(0)
    k = torch.randn(T, d, device=device)
    v = torch.randn(T, d, device=device)
    b_v = torch.tensor([16, 8, 4, 2, 0] * (T // 5 + 1))[:T].to(device)
    b_k = torch.tensor([8, 4, 16, 2] * (d // 4 + 1))[:d].to(device)
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    return pack_trizone(k, v, allocation)


@requires_kernel_backend
def test_fused_kernel_matches_native_output():
    d = 64
    packed = _random_packed_cache(T=40, d=d, device="cuda")
    q_tau = torch.randn(d, device="cuda")
    k_new = torch.randn(2, d, device="cuda")
    v_new = torch.randn(2, d, device="cuda")
    sqrt_d = math.sqrt(d)

    native_out = packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="native")
    kernel_out = packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="kernel")

    assert torch.allclose(native_out, kernel_out, atol=1e-3)


@requires_kernel_backend
def test_fused_kernel_does_not_materialize_dequantized_k_tile():
    # Structural check (spec Sec 8's core requirement): the fused path must
    # not allocate a full (n_kept, d) FP16 tensor for Zone A's dequantized K.
    # We check this indirectly via peak memory: the fused kernel's decode
    # call should not allocate materially more than O(n_new + 1) extra
    # tensors beyond the packed cache itself.
    d = 64
    packed = _random_packed_cache(T=2000, d=d, device="cuda")  # large n_kept
    q_tau = torch.randn(d, device="cuda")
    k_new = torch.randn(1, d, device="cuda")
    v_new = torch.randn(1, d, device="cuda")
    sqrt_d = math.sqrt(d)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="kernel")
    peak_delta = torch.cuda.max_memory_allocated() - mem_before

    n_kept = packed.zone_a_k.shape[0]
    dequantized_fp16_tile_bytes = n_kept * d * 2  # what a materialized tile WOULD cost
    # The fused path's extra allocation should be well under what a fully
    # materialized dequantized K tile would cost -- generous 50% margin to
    # absorb kernel workspace/output buffers.
    assert peak_delta < dequantized_fp16_tile_bytes * 0.5


@requires_kernel_backend
def test_backend_kernel_raises_clearly_off_cuda():
    from rdkv.kernel._require import require_kernel_backend as _guard

    # Sanity: the guard itself (already tested in test_kernel_require.py)
    # is what packed_decode(..., backend="kernel") must call before
    # dispatching -- this just confirms wiring, not the guard's own logic.
    with pytest.raises(RuntimeError):
        _guard("cpu")
```

- [ ] **Step 2: Run tests to verify they fail (or skip, off-CUDA)**

Run: `cd rdkv && pytest tests/test_kernel_backend.py -v`
Expected on a machine without CUDA+Triton: all 3 tests SKIPPED. On a CUDA+Triton machine: FAIL with `TypeError: packed_decode() got an unexpected keyword argument 'backend'`.

- [ ] **Step 3: Add the `backend` parameter to `rdkv/rdkv/decode.py`'s `packed_decode`**

```python
# In rdkv/rdkv/decode.py, change the signature and add dispatch at the top:

def packed_decode(
    packed: PackedCache,
    q_tau: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    sqrt_d: float,
    backend: str = "native",
) -> torch.Tensor:
    """... (existing docstring) ...

    backend: "native" (default, this module's own dequantize-then-matmul
    reference) or "kernel" (GPU-only, fuses Zone A's K dequantization into
    the score computation -- see rdkv.kernel.fused_decode).
    """
    if backend not in ("native", "kernel"):
        raise ValueError(f"backend must be 'native' or 'kernel', got {backend!r}")
    if backend == "kernel":
        from .kernel._require import require_kernel_backend

        device = "cuda" if q_tau.is_cuda else "cpu"
        require_kernel_backend(device)
        from .kernel.fused_decode import fused_packed_decode

        return fused_packed_decode(packed, q_tau, k_new, v_new, sqrt_d)

    # ... existing native implementation body unchanged below this point ...
```

- [ ] **Step 4: Write `rdkv/rdkv/kernel/fused_decode.py`**

```python
# rdkv/rdkv/kernel/fused_decode.py
"""Fused Triton kernel for RDKV's packed decode (spec Sec 8).

Fuses Zone A's algebraic K dequantization directly into the attention
score computation:

    q_tau^T k_hat_t = sum_c (s_c * q_{tau,c}) * k_tilde_{t,c}  - bias

where k_hat_{t,c} = s_c*(k_tilde_{t,c} - z_c) is the per-channel affine
dequantization (see rdkv.trizone's k_scale/k_zero_point), so
q_tau^T k_hat_t = sum_c s_c*q_{tau,c}*k_tilde_{t,c} - sum_c s_c*q_{tau,c}*z_c.
The second term is a single per-query-head bias, computed once and
subtracted from every score -- never requiring a materialized FP16 K
tile for Zone A. This is the module Task 12's structural-memory test
(test_fused_kernel_does_not_materialize_dequantized_k_tile) checks.

One program per decode step (the batch here is n_kept -- the whole point
is a single query attending over the packed cache), block over n_kept.
"""

import torch
import triton
import triton.language as tl

_BLOCK_N = 128


@triton.jit
def _fused_score_kernel(
    q_scaled_ptr, k_tilde_ptr, bias_ptr, scores_ptr,
    stride_k_row,
    N, D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """scores[n] = sum_d q_scaled[d] * k_tilde[n, d] - bias, for a block of
    N (Zone A's kept-token) rows. q_scaled[d] = s_d * q_tau[d] is
    precomputed on the host (cheap, O(d)) so the kernel's inner loop is a
    a plain dot product against the still-quantized-integer k_tilde -- no
    K dequantization happens inside or outside this kernel."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < N
    mask_d = offs_d < D
    mask_2d = mask_n[:, None] & mask_d[None, :]

    q_scaled = tl.load(q_scaled_ptr + offs_d, mask=mask_d, other=0.0)
    k_tilde = tl.load(
        k_tilde_ptr + offs_n[:, None] * stride_k_row + offs_d[None, :], mask=mask_2d, other=0.0
    )
    bias = tl.load(bias_ptr)

    raw_score = tl.sum(k_tilde * q_scaled[None, :], axis=1)
    score = raw_score - bias
    tl.store(scores_ptr + offs_n, score, mask=mask_n)


def _fused_zone_a_scores(q_tau: torch.Tensor, packed) -> torch.Tensor:
    """Computes Zone A's raw (pre-softmax, pre-sqrt_d) scores against ALL
    kept tokens' K rows, fused with dequantization per this module's
    docstring. Returns a (n_kept,) tensor."""
    d = q_tau.shape[0]
    n_kept = packed.zone_a_k.shape[0]
    device = q_tau.device

    if n_kept == 0:
        return torch.empty(0, device=device)

    inv_perm = torch.argsort(packed.k_channel_perm)
    # q must be permuted the same way K's channels were at packing time
    # (spec Sec 9: "permute q to match"), then pre-scaled by s_c.
    q_permuted = q_tau[packed.k_channel_perm]
    q_scaled = (q_permuted * packed.k_scale).contiguous()
    # bias = sum_c s_c * q_{tau,c} * z_c, a single per-query-head scalar.
    bias = (q_scaled * packed.k_zero_point).sum().reshape(1)

    k_tilde = packed.zone_a_k.to(device).float().contiguous()
    scores = torch.empty(n_kept, device=device, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    block_n = _BLOCK_N
    grid = (triton.cdiv(n_kept, block_n),)

    _fused_score_kernel[grid](
        q_scaled, k_tilde, bias, scores,
        k_tilde.stride(0),
        n_kept, D=d, BLOCK_N=block_n, BLOCK_D=block_d,
    )
    _ = inv_perm  # inv_perm not needed here: q was permuted forward to match k_tilde's stored order
    return scores


def fused_packed_decode(
    packed, q_tau: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, sqrt_d: float
) -> torch.Tensor:
    """GPU-only fused equivalent of rdkv.decode.packed_decode(..., backend="native").

    Zone A's scores are computed via _fused_zone_a_scores (no materialized
    dequantized K tile). Zone B (already FP16) and Zone C (new tokens) use
    ordinary matmuls, since they were never quantized in the first place --
    there is nothing to fuse for them.
    """
    d = q_tau.shape[0]
    zone_a_scores = _fused_zone_a_scores(q_tau, packed) / sqrt_d

    non16_mask = ~packed.zone_b_mask
    v_zone_a_parts = [seg for seg in packed.zone_a_v.values() if seg.shape[0] > 0]
    v_zone_a = torch.cat(v_zone_a_parts, dim=0) if v_zone_a_parts else torch.empty(0, d, device=q_tau.device)
    scores_for_zone_a_v = zone_a_scores[non16_mask]

    scores_parts, values_parts = [], []
    if v_zone_a.shape[0] > 0:
        scores_parts.append(scores_for_zone_a_v)
        values_parts.append(v_zone_a)

    if packed.zone_b_v.shape[0] > 0:
        scores_zone_b = zone_a_scores[packed.zone_b_mask]  # Zone B's K rows are the same Zone A(K) storage
        scores_parts.append(scores_zone_b)
        values_parts.append(packed.zone_b_v)

    scores_c = (q_tau @ k_new.T) / sqrt_d
    scores_parts.append(scores_c)
    values_parts.append(v_new)

    all_scores = torch.cat(scores_parts, dim=0)
    all_values = torch.cat(values_parts, dim=0)
    all_weights = torch.softmax(all_scores, dim=0)
    return all_weights @ all_values
```

- [ ] **Step 5: Run the parity test on a CUDA+Triton machine, or confirm clean skip otherwise**

Run: `cd rdkv && pytest tests/test_kernel_backend.py -v`
Expected: on CUDA+Triton, all 3 PASS. Otherwise, all 3 SKIPPED (never silently passing without running, never failing due to environment).

- [ ] **Step 6: Run the full CPU-visible suite once more to confirm nothing broke**

Run: `cd rdkv && pytest tests/ -v -k "not kernel_backend"`
Expected: all non-kernel tests still pass (kernel backend tests are excluded here only because most dev machines lack CUDA; Step 5 already covered them where available).

- [ ] **Step 7: Commit**

```bash
git add rdkv/rdkv/decode.py rdkv/rdkv/kernel/fused_decode.py rdkv/tests/test_kernel_backend.py
git commit -m "rdkv: implement fused Triton packed-decode kernel with native parity"
```

---

## Task 13: End-to-end example script (allocate → pack → decode)

**Files:**
- Create: `rdkv/examples/packed_decode_demo.py`

**Interfaces:**
- Consumes: `rdkv.RDKVAllocator` (Phase 1), `rdkv.trizone.pack_trizone`, `rdkv.decode.packed_decode` (Phase 2).
- Produces: a standalone demonstration script chaining the full pipeline on a real model's prefill K/V, reporting compression ratio and a native-vs-kernel output comparison when CUDA is available.

- [ ] **Step 1: Write `rdkv/examples/packed_decode_demo.py`**

```python
"""End-to-end RDKV demo: allocate bit-widths on a real model's prefill
K/V, pack into TriZone storage, and run one packed decode step.

Requires the `examples` extra (and, for the kernel-backend comparison,
the `kernel` extra on a CUDA machine):

    pip install -e ".[examples]"

Usage:
    python examples/packed_decode_demo.py --model sshleifer/tiny-gpt2 --b-tok 4.0
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rdkv import RDKVAllocator
from rdkv.trizone import pack_trizone
from rdkv.decode import packed_decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--b-tok", type=float, default=4.0)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog. " * 8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager", output_attentions=True)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    layer_attn = outputs.attentions[0][0, 0]  # first layer, first head: (T, T)
    T = layer_attn.shape[-1]
    d = model.config.hidden_size // model.config.n_head if hasattr(model.config, "n_head") else 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(T, d, device=device)  # see Task 6's note: real per-head Q/K extraction is a follow-up
    k = torch.randn(T, d, device=device)
    v = torch.randn(T, d, device=device)

    allocator = RDKVAllocator()
    allocation = allocator.allocate(layer_attn, q, k, b_tok=args.b_tok)
    packed = pack_trizone(k, v, allocation)

    n_kept = allocation.kept_tokens.shape[0]
    fp16_bits = T * d * 16 * 2  # K+V, full precision baseline
    packed_bits = sum(seg.numel() * bits for bits, seg in packed.zone_a_v.items())
    packed_bits += packed.zone_a_k.numel() * (packed.zone_a_k.element_size() * 8)  # conservative, pre-bitpack size
    packed_bits += packed.zone_b_v.numel() * 16
    print(f"model={args.model} T={T} d={d} kept={n_kept}/{T} ({100 * n_kept / T:.1f}%)")
    print(f"approx compression vs FP16: {fp16_bits / max(packed_bits, 1):.2f}x (pre-bitpacking element counts)")

    q_tau = torch.randn(d, device=device)
    k_new = torch.randn(1, d, device=device)
    v_new = torch.randn(1, d, device=device)
    sqrt_d = math.sqrt(d)

    native_out = packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="native")
    print(f"native packed-decode output: shape={tuple(native_out.shape)}")

    if device == "cuda":
        try:
            kernel_out = packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="kernel")
            max_diff = (native_out - kernel_out).abs().max().item()
            print(f"kernel packed-decode max abs diff vs native: {max_diff:.6f}")
        except RuntimeError as exc:
            print(f"kernel backend unavailable: {exc}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs on CPU (native path)**

Run: `cd rdkv && python examples/packed_decode_demo.py --model sshleifer/tiny-gpt2`
Expected: prints kept-token stats, approximate compression ratio, and native output shape, no exceptions. Kernel comparison line only appears on a CUDA machine.

- [ ] **Step 3: Commit**

```bash
git add rdkv/examples/packed_decode_demo.py
git commit -m "rdkv: add end-to-end allocate-pack-decode example script"
```

---

## Task 14: Full two-phase suite verification and README finalization

**Files:**
- Modify: `rdkv/README.md`
- Modify: `README.md` (root, the RDKV entry from Task 7)

**Interfaces:**
- None (verification and documentation only).

- [ ] **Step 1: Run the complete rdkv test suite**

Run: `cd rdkv && pytest tests/ -v`
Expected: every Phase 1 test (Tasks 2-5) and every Phase 2 test (Tasks 9-12) passes or, for `test_kernel_backend.py` specifically, cleanly skips off-CUDA.

- [ ] **Step 2: Verify the full public API imports cleanly**

Run: `cd rdkv && python -c "
from rdkv import RDKVAllocator, continuous_waterfill, mckp_bisect, token_weight_v, channel_weight_k, bennett_sigma, bennett_distortion, AllocationResult
from rdkv.trizone import pack_trizone, PackedCache
from rdkv.decode import packed_decode
print('all public symbols import cleanly')
"`
Expected: prints the confirmation line, no `ImportError`.

- [ ] **Step 3: Update `rdkv/README.md`'s Phase 2 status**

Replace the "Not yet implemented (Phase 2)" line and the sections around it with:

```markdown
**Phase 1:** continuous water-filling (Theorem 3.3), discrete MCKP bit
allocation (Algorithm 2), per-unit weight computation (Propositions
3.1/3.2), and the three-stage allocation pipeline (Algorithm 1 Stages 1-3).
Pure PyTorch, no custom GPU kernel.

**Phase 2:** TriZone packing (Algorithm 1 Stage 4) and packed decode
(Eq. 7), with a native PyTorch reference implementation and a GPU-only
fused Triton kernel (`backend="kernel"`) that never materializes a
dequantized FP16 K tile for the packed zone. Requires the `kernel` extra
and CUDA; falls back to a clear `RuntimeError` (not a silent CPU fallback)
otherwise.

**Disclosed approximation (both phases):** `ε_u(b)` is the analytic
Bennett curve `σ_u · 2^(−b)`, standing in for the paper's
empirically-calibrated per-coordinate distortion table (Appendix B). This
affects which bit-widths Phase 1 chooses; Phase 2 packs and decodes
whatever bit-widths it's given and does not depend on how they were chosen.

## Install

```bash
pip install -e ".[test]"          # core + tests
pip install -e ".[examples]"      # + real-model example scripts
pip install -e ".[kernel]"        # + fused Triton decode kernel (CUDA only)
```
```

- [ ] **Step 4: Update the root `README.md`'s RDKV entry (from Task 7) to reflect both phases**

```markdown
### RDKV -- Joint Eviction and Quantization of the KV Cache ([`rdkv/`](rdkv/))

**Paper**: [Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache](https://arxiv.org/abs/2605.08317) (arXiv:2605.08317)

RDKV treats KV cache eviction and quantization as the same operation — bit-width assignment — evaluated at different depths (0 bits = evicted). Implemented end to end: closed-form continuous water-filling (Theorem 3.3), discrete MCKP bit allocation via Lagrangian bisection (Algorithm 2), per-unit weight computation (Propositions 3.1/3.2), the three-stage allocation pipeline (Algorithm 1 Stages 1-3), TriZone packed storage (Algorithm 1 Stage 4), and a fused-dequantization decode kernel (Eq. 7) that never materializes a dequantized FP16 K tile.

**Disclosed approximation**: the empirically-calibrated per-coordinate distortion table from the paper's Appendix B is stood in for by the analytic Bennett curve `σ_u · 2^(−b)`; see [`rdkv/README.md`](rdkv/README.md).

See [`rdkv/rdkv-primer.html`](rdkv/rdkv-primer.html) for the math derivation, [`docs/superpowers/specs/2026-08-31-rdkv-design.md`](docs/superpowers/specs/2026-08-31-rdkv-design.md) for the full spec, and [`rdkv/README.md`](rdkv/README.md) for install/test instructions.
```

- [ ] **Step 5: Commit**

```bash
git add rdkv/README.md README.md
git commit -m "docs: mark RDKV Phase 1 and Phase 2 as implemented"
```

This task has no further steps — it's the final checkpoint confirming Tasks 1-13 together satisfy both phases of the plan's Goal statement.
