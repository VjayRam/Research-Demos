# TurboQuant Paper-Accurate Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing `turbo-quant/` flat-file implementation with an installable `turboquant` Python package that implements Algorithm 1 (`TurboQuant_mse`), Algorithm 2 (`TurboQuant_prod`), and PolarQuant exactly as specified in the papers (no Gaussian-approximation shortcuts, no FWHT rotation substitute), plus an `examples/` harness that measures perplexity vs. compression ratio on real HuggingFace models (Qwen2.5, Gemma-2).

**Architecture:** Six focused core modules (`rotation`, `distributions`, `lloyd_max`, `codebook`, `qjl`, `cartesian`, `polar`) each with a single, narrow responsibility and no configuration branches — one correct behavior per module. `cartesian.py` and `polar.py` compose the lower-level primitives into the two paper algorithms and PolarQuant. A separate `examples/` directory (not part of the installable package) hooks the core into a HuggingFace model's KV cache via a `DynamicCache` subclass.

**Tech Stack:** Python 3.10+, PyTorch, SciPy (`scipy.integrate.quad` for Lloyd-Max), pytest. Examples additionally use `transformers` and (implicitly, via `transformers`) `accelerate`.

**Spec:** `docs/superpowers/specs/2026-08-27-turboquant-redesign-design.md`

## Global Constraints

- No Gaussian-approximation Lloyd-Max path — only the exact Beta/sin-power densities (spec: "Lloyd-Max density").
- No FWHT/Hadamard rotation in the core package — only QR-Haar (spec: "Rotation").
- No `qjl_dim` / algorithm-switch parameters inside `TurboQuantMSE`/`TurboQuantProd` — one behavior per class (spec: "cartesian.py").
- Core package (`turboquant/`) must not import anything from `examples/` or contain HF/model-specific code (spec: "Non-goals", "Architecture").
- All randomness (`rotation.py`, `qjl.py`) is seeded and deterministic; no unseeded global RNG fallback (spec: "Error handling").
- Old files `turboquant.py`, `compressors.py`, `lloyd_max.py`, `test_algorithm.py`, `evaluate.py` are deleted, not kept alongside the new package (spec: "Migration").

---

## Task 1: Package scaffolding

**Files:**
- Create: `turbo-quant/pyproject.toml`
- Create: `turbo-quant/turboquant/__init__.py` (empty for now — populated in Task 10)
- Create: `turbo-quant/tests/__init__.py` (empty)

**Interfaces:**
- Produces: an installable package named `turboquant`, importable after `pip install -e turbo-quant/`.

- [ ] **Step 1: Create the package directories**

```bash
mkdir -p "C:/Vijay/PyCode/Research/turbo-quant/turboquant"
mkdir -p "C:/Vijay/PyCode/Research/turbo-quant/tests"
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "turboquant"
version = "0.1.0"
description = "Paper-accurate implementation of TurboQuant and PolarQuant vector quantization"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "scipy>=1.10",
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
include = ["turboquant*"]
```

- [ ] **Step 3: Create empty `__init__.py` files**

`turbo-quant/turboquant/__init__.py`:
```python
```

`turbo-quant/tests/__init__.py`:
```python
```

- [ ] **Step 4: Install the package in editable mode with test deps**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pip install -e ".[test]"`
Expected: installs successfully, `import turboquant` works (even though empty).

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/pyproject.toml turbo-quant/turboquant/__init__.py turbo-quant/tests/__init__.py
git commit -m "$(cat <<'EOF'
Scaffold turboquant as an installable package

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 2: `rotation.py` — Haar-random orthogonal rotation

**Files:**
- Create: `turbo-quant/turboquant/rotation.py`
- Test: `turbo-quant/tests/test_rotation.py`

**Interfaces:**
- Produces: `generate_rotation_matrix(d: int, seed: int) -> torch.Tensor` — a `d x d` orthogonal `float32` matrix, cached per `(d, seed)`.

- [ ] **Step 1: Write the failing tests**

`turbo-quant/tests/test_rotation.py`:
```python
import torch

from turboquant.rotation import generate_rotation_matrix


def test_output_is_orthogonal():
    q = generate_rotation_matrix(d=16, seed=0)
    identity = torch.eye(16)
    assert torch.allclose(q.T @ q, identity, atol=1e-5)


def test_output_shape_and_dtype():
    q = generate_rotation_matrix(d=8, seed=0)
    assert q.shape == (8, 8)
    assert q.dtype == torch.float32


def test_deterministic_given_same_seed():
    q1 = generate_rotation_matrix(d=16, seed=42)
    q2 = generate_rotation_matrix(d=16, seed=42)
    assert torch.equal(q1, q2)


def test_different_seeds_differ():
    q1 = generate_rotation_matrix(d=16, seed=1)
    q2 = generate_rotation_matrix(d=16, seed=2)
    assert not torch.equal(q1, q2)


def test_cache_returns_same_tensor_object():
    q1 = generate_rotation_matrix(d=16, seed=7)
    q2 = generate_rotation_matrix(d=16, seed=7)
    assert q1 is q2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_rotation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turboquant.rotation'`

- [ ] **Step 3: Write `rotation.py`**

```python
"""Haar-random orthogonal rotation via QR decomposition (paper's exact setup step)."""

import torch

_rotation_cache: dict[tuple[int, int], torch.Tensor] = {}


def generate_rotation_matrix(d: int, seed: int) -> torch.Tensor:
    """Haar-distributed random orthogonal d x d matrix, built by QR-decomposing
    a random Gaussian matrix and fixing the sign ambiguity in Q.

    Cached per (d, seed): this is the paper's "setup, once per (d,b)" step,
    data-independent and reused across every call at that (d, seed).
    """
    key = (d, seed)
    if key in _rotation_cache:
        return _rotation_cache[key]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    g = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(g)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = (q * diag_sign.unsqueeze(0)).to(torch.float32)

    _rotation_cache[key] = q
    return q
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_rotation.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/rotation.py turbo-quant/tests/test_rotation.py
git commit -m "$(cat <<'EOF'
Add paper-exact Haar-random rotation via QR decomposition

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 3: `distributions.py` — exact Beta and polar-angle densities

**Files:**
- Create: `turbo-quant/turboquant/distributions.py`
- Test: `turbo-quant/tests/test_distributions.py`

**Interfaces:**
- Consumes: `generate_rotation_matrix` from Task 2 (for the Haar-ness spot check only).
- Produces: `Density` dataclass (`pdf: Callable[[float], float]`, `support: tuple[float, float]`, `name: str`); `beta_coordinate_density(d: int) -> Density`; `polar_angle_density(level: int) -> Density`.

- [ ] **Step 1: Write the failing tests**

`turbo-quant/tests/test_distributions.py`:
```python
import math

import torch
from scipy import integrate, stats

from turboquant.distributions import beta_coordinate_density, polar_angle_density
from turboquant.rotation import generate_rotation_matrix


def test_beta_density_integrates_to_one():
    density = beta_coordinate_density(d=8)
    total, _ = integrate.quad(density.pdf, *density.support)
    assert abs(total - 1.0) < 1e-6


def test_beta_density_matches_closed_form_at_d4():
    # f_X(x; d=4) = (2/pi) * sqrt(1 - x^2); coeff at x=0 is exactly 2/pi.
    density = beta_coordinate_density(d=4)
    assert math.isclose(density.pdf(0.0), 2 / math.pi, rel_tol=1e-6)
    assert density.pdf(1.0) == 0.0
    assert density.pdf(-1.0) == 0.0


def test_beta_density_is_symmetric():
    density = beta_coordinate_density(d=32)
    for x in (0.1, 0.3, 0.7):
        assert math.isclose(density.pdf(x), density.pdf(-x), rel_tol=1e-9)


def test_polar_level1_is_uniform_on_full_circle():
    density = polar_angle_density(level=1)
    assert density.support == (0.0, 2 * math.pi)
    assert math.isclose(density.pdf(1.0), density.pdf(4.0))


def test_polar_level_ge2_integrates_to_one_and_peaks_near_quarter_pi():
    for level in (2, 3, 4):
        density = polar_angle_density(level=level)
        total, _ = integrate.quad(density.pdf, *density.support)
        assert abs(total - 1.0) < 1e-6
        # deeper levels concentrate more sharply around pi/4
        assert density.pdf(math.pi / 4) > density.pdf(math.pi / 4 - 0.5)


def test_polar_density_rejects_invalid_level():
    import pytest

    with pytest.raises(ValueError):
        polar_angle_density(level=0)


def test_rotation_coordinate_matches_beta_density_via_ks_test():
    # Pi @ e1 should be distributed as beta_coordinate_density(d) across seeds.
    d = 8
    e1 = torch.zeros(d)
    e1[0] = 1.0
    samples = []
    for seed in range(400):
        q = generate_rotation_matrix(d=d, seed=seed)
        samples.append((q @ e1)[0].item())

    density = beta_coordinate_density(d)

    def cdf(x):
        val, _ = integrate.quad(density.pdf, -1.0, x)
        return val

    ks_stat, p_value = stats.kstest(samples, cdf)
    assert p_value > 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_distributions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turboquant.distributions'`

- [ ] **Step 3: Write `distributions.py`**

```python
"""Exact coordinate densities used by TurboQuant (Beta) and PolarQuant (sin-power angles)."""

import math
from dataclasses import dataclass
from typing import Callable

from scipy import integrate


@dataclass(frozen=True)
class Density:
    """A probability density with a known, finite support interval."""

    pdf: Callable[[float], float]
    support: tuple[float, float]
    name: str


def beta_coordinate_density(d: int) -> Density:
    """Exact coordinate density of a uniform point on S^(d-1) (paper Eq. 4).

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2),
    x in [-1, 1].
    """
    log_coeff = math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2)
    coeff = math.exp(log_coeff)
    power = (d - 3) / 2

    def pdf(x: float) -> float:
        if abs(x) >= 1.0:
            return 0.0
        return coeff * (1 - x * x) ** power

    return Density(pdf=pdf, support=(-1.0, 1.0), name=f"beta_d{d}")


def polar_angle_density(level: int) -> Density:
    """PolarQuant angle density at a given recursion level.

    Level 1 pairs raw (signed) coordinates: uniform on [0, 2*pi).
    Level >= 2 pairs nonnegative radii from the previous level: density
    proportional to sin(2*theta)^(2^(level-1) - 1) on [0, pi/2].
    """
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")

    if level == 1:
        lo, hi = 0.0, 2 * math.pi
        c = 1.0 / (hi - lo)
        return Density(pdf=lambda theta: c, support=(lo, hi), name="polar_uniform")

    power = 2 ** (level - 1) - 1
    lo, hi = 0.0, math.pi / 2

    def unnormalized(theta: float) -> float:
        return math.sin(2 * theta) ** power

    z, _ = integrate.quad(unnormalized, lo, hi)

    def pdf(theta: float) -> float:
        if theta < lo or theta > hi:
            return 0.0
        return unnormalized(theta) / z

    return Density(pdf=pdf, support=(lo, hi), name=f"polar_level{level}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_distributions.py -v`
Expected: PASS (7 passed). The KS test is randomized over 400 seeds — if it flakes below `p_value > 0.01` on a rerun, that is a real signal to inspect `generate_rotation_matrix`, not just retry.

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/distributions.py turbo-quant/tests/test_distributions.py
git commit -m "$(cat <<'EOF'
Add exact Beta and polar-angle coordinate densities

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 4: `lloyd_max.py` — generic continuous Lloyd-Max solver

**Files:**
- Create: `turbo-quant/turboquant/lloyd_max.py`
- Test: `turbo-quant/tests/test_lloyd_max.py`

**Interfaces:**
- Consumes: `beta_coordinate_density` from Task 3 (for the Theorem 1 reproduction test).
- Produces: `solve_lloyd_max(pdf, support, bits, max_iter=200, tol=1e-10) -> tuple[torch.Tensor, torch.Tensor]` (centroids, boundaries); `expected_distortion(pdf, support, centroids, boundaries) -> float`.

- [ ] **Step 1: Write the failing tests**

`turbo-quant/tests/test_lloyd_max.py`:
```python
import math

from turboquant.distributions import beta_coordinate_density
from turboquant.lloyd_max import expected_distortion, solve_lloyd_max


def test_centroid_and_boundary_counts():
    density = beta_coordinate_density(d=64)
    centroids, boundaries = solve_lloyd_max(density.pdf, density.support, bits=2)
    assert centroids.shape == (4,)
    assert boundaries.shape == (3,)


def test_centroids_are_sorted():
    density = beta_coordinate_density(d=64)
    centroids, _ = solve_lloyd_max(density.pdf, density.support, bits=3)
    assert list(centroids) == sorted(centroids.tolist())


def test_b1_centroid_matches_exact_half_normal_formula():
    # For b=1, the optimal centroid is exactly E[X | X > 0] = sqrt(2/pi)/sqrt(d).
    d = 128
    density = beta_coordinate_density(d)
    centroids, _ = solve_lloyd_max(density.pdf, density.support, bits=1)
    expected = math.sqrt(2 / math.pi) / math.sqrt(d)
    assert math.isclose(centroids[1].item(), expected, rel_tol=1e-3)
    assert math.isclose(centroids[0].item(), -expected, rel_tol=1e-3)


def test_reproduces_paper_theorem1_table_at_d128():
    # Table: b -> d * C(f_X, b) approx 0.360, 0.117, 0.030, 0.009
    expected_by_bits = {1: 0.360, 2: 0.117, 3: 0.030, 4: 0.009}
    d = 128
    density = beta_coordinate_density(d)
    for bits, expected in expected_by_bits.items():
        centroids, boundaries = solve_lloyd_max(density.pdf, density.support, bits=bits)
        distortion = expected_distortion(density.pdf, density.support, centroids, boundaries)
        d_mse = d * distortion
        assert abs(d_mse - expected) < 0.02, f"bits={bits}: got {d_mse}, want {expected}"


def test_distortion_decreases_with_more_bits():
    density = beta_coordinate_density(d=64)
    distortions = []
    for bits in (1, 2, 3, 4):
        centroids, boundaries = solve_lloyd_max(density.pdf, density.support, bits=bits)
        distortions.append(expected_distortion(density.pdf, density.support, centroids, boundaries))
    assert distortions == sorted(distortions, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_lloyd_max.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turboquant.lloyd_max'`

- [ ] **Step 3: Write `lloyd_max.py`**

```python
"""Generic continuous Lloyd-Max scalar quantizer solver.

Parameterized entirely by a (pdf, support) pair -- no knowledge of any
specific density baked in. Used by both cartesian.py (Beta density) and
polar.py (sin-power angle densities).
"""

from typing import Callable

import torch
from scipy import integrate


def solve_lloyd_max(
    pdf: Callable[[float], float],
    support: tuple[float, float],
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous 1-D Lloyd-Max quantizer for an arbitrary density on a finite support.

    Returns (centroids, boundaries): centroids has 2**bits entries (sorted
    ascending), boundaries has 2**bits - 1 entries (midpoints between
    neighboring centroids).
    """
    n_levels = 2 ** bits
    lo, hi = support

    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo] + boundaries + [hi]

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = integrate.quad(pdf, a, b)
            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


def expected_distortion(
    pdf: Callable[[float], float],
    support: tuple[float, float],
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> float:
    """Expected squared quantization error under pdf: C(f, b) from the paper."""
    lo, hi = support
    edges = [lo] + boundaries.tolist() + [hi]
    total = 0.0
    for i in range(len(centroids)):
        a, b = edges[i], edges[i + 1]
        c = centroids[i].item()
        dist, _ = integrate.quad(lambda x: (x - c) ** 2 * pdf(x), a, b)
        total += dist
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_lloyd_max.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/lloyd_max.py turbo-quant/tests/test_lloyd_max.py
git commit -m "$(cat <<'EOF'
Add generic continuous Lloyd-Max solver over exact densities

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 5: `codebook.py` — cached Codebook with argmin quantize

**Files:**
- Create: `turbo-quant/turboquant/codebook.py`
- Test: `turbo-quant/tests/test_codebook.py`

**Interfaces:**
- Consumes: `Density` and `beta_coordinate_density` from Task 3; `solve_lloyd_max`, `expected_distortion` from Task 4.
- Produces: `Codebook` dataclass with fields `centroids: torch.Tensor`, `boundaries: torch.Tensor`, `distortion: float`; classmethod `Codebook.for_density(density: Density, bits: int) -> Codebook`; instance methods `quantize(x: torch.Tensor) -> torch.Tensor` (indices), `dequantize(indices: torch.Tensor) -> torch.Tensor`.

- [ ] **Step 1: Write the failing tests**

`turbo-quant/tests/test_codebook.py`:
```python
import torch

from turboquant.codebook import Codebook
from turboquant.distributions import beta_coordinate_density


def test_quantize_picks_nearest_centroid_exactly():
    codebook = Codebook.for_density(beta_coordinate_density(d=32), bits=2)
    # Feeding a centroid's exact value back in must return its own index.
    for i, c in enumerate(codebook.centroids):
        idx = codebook.quantize(c.unsqueeze(0))
        assert idx.item() == i


def test_dequantize_looks_up_centroid_values():
    codebook = Codebook.for_density(beta_coordinate_density(d=32), bits=2)
    indices = torch.tensor([0, 1, 2, 3])
    values = codebook.dequantize(indices)
    assert torch.equal(values, codebook.centroids)


def test_quantize_dequantize_shapes_for_batched_vectors():
    codebook = Codebook.for_density(beta_coordinate_density(d=16), bits=3)
    x = torch.randn(5, 16) * 0.1
    indices = codebook.quantize(x)
    assert indices.shape == (5, 16)
    reconstructed = codebook.dequantize(indices)
    assert reconstructed.shape == (5, 16)


def test_for_density_caches_by_name_and_bits():
    c1 = Codebook.for_density(beta_coordinate_density(d=64), bits=2)
    c2 = Codebook.for_density(beta_coordinate_density(d=64), bits=2)
    assert c1 is c2


def test_for_density_distinguishes_different_bits():
    c1 = Codebook.for_density(beta_coordinate_density(d=64), bits=1)
    c2 = Codebook.for_density(beta_coordinate_density(d=64), bits=2)
    assert c1 is not c2
    assert c1.centroids.numel() == 2
    assert c2.centroids.numel() == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_codebook.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turboquant.codebook'`

- [ ] **Step 3: Write `codebook.py`**

```python
"""A Lloyd-Max codebook: solved centroids/boundaries plus quantize/dequantize."""

from dataclasses import dataclass

import torch

from .distributions import Density
from .lloyd_max import expected_distortion, solve_lloyd_max

_codebook_cache: dict[tuple[str, int], "Codebook"] = {}


@dataclass
class Codebook:
    centroids: torch.Tensor
    boundaries: torch.Tensor
    distortion: float

    @classmethod
    def for_density(cls, density: Density, bits: int) -> "Codebook":
        key = (density.name, bits)
        if key in _codebook_cache:
            return _codebook_cache[key]

        centroids, boundaries = solve_lloyd_max(density.pdf, density.support, bits)
        distortion = expected_distortion(density.pdf, density.support, centroids, boundaries)
        codebook = cls(centroids=centroids, boundaries=boundaries, distortion=distortion)
        _codebook_cache[key] = codebook
        return codebook

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Literal per-element argmin over centroids (paper's Algorithm 1 line)."""
        diffs = x.unsqueeze(-1) - self.centroids.to(x.device)
        return diffs.abs().argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(indices.device)[indices]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_codebook.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/codebook.py turbo-quant/tests/test_codebook.py
git commit -m "$(cat <<'EOF'
Add cached Codebook with literal argmin quantize/dequantize

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 6: `qjl.py` — QJL sign projection

**Files:**
- Create: `turbo-quant/turboquant/qjl.py`
- Test: `turbo-quant/tests/test_qjl.py`

**Interfaces:**
- Produces: `generate_qjl_matrix(d: int, seed: int) -> torch.Tensor` (`d x d`, cached per `(d, seed)`); `sign_quantize(x: torch.Tensor) -> torch.Tensor` (entries in `{-1.0, +1.0}`, zero maps to `+1.0`).

- [ ] **Step 1: Write the failing tests**

`turbo-quant/tests/test_qjl.py`:
```python
import torch

from turboquant.qjl import generate_qjl_matrix, sign_quantize


def test_qjl_matrix_shape_and_determinism():
    s1 = generate_qjl_matrix(d=16, seed=5)
    s2 = generate_qjl_matrix(d=16, seed=5)
    assert s1.shape == (16, 16)
    assert torch.equal(s1, s2)


def test_qjl_matrix_different_seeds_differ():
    s1 = generate_qjl_matrix(d=16, seed=1)
    s2 = generate_qjl_matrix(d=16, seed=2)
    assert not torch.equal(s1, s2)


def test_sign_quantize_only_returns_plus_minus_one():
    x = torch.tensor([-2.0, -0.001, 0.0, 0.001, 3.0])
    signs = sign_quantize(x)
    assert torch.equal(signs.abs(), torch.ones_like(signs))


def test_sign_quantize_zero_maps_to_plus_one():
    x = torch.tensor([0.0])
    signs = sign_quantize(x)
    assert signs.item() == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_qjl.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turboquant.qjl'`

- [ ] **Step 3: Write `qjl.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_qjl.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/qjl.py turbo-quant/tests/test_qjl.py
git commit -m "$(cat <<'EOF'
Add QJL sign projection for Algorithm 2's residual stage

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 7: `cartesian.py` — `TurboQuantMSE` (Algorithm 1)

**Files:**
- Create: `turbo-quant/turboquant/cartesian.py`
- Test: `turbo-quant/tests/test_cartesian_mse.py`

**Interfaces:**
- Consumes: `generate_rotation_matrix` (Task 2), `beta_coordinate_density` (Task 3), `Codebook` (Task 5).
- Produces: `TurboQuantMSE(d: int, bits: int, seed: int = 0)` with `.rotate(x) -> Tensor`, `.unrotate(y) -> Tensor`, `.quantize(x: Tensor) -> tuple[Tensor, Tensor]` (indices, norm), `.dequantize(indices: Tensor, norm: Tensor) -> Tensor`. `bits < 1` raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

`turbo-quant/tests/test_cartesian_mse.py`:
```python
import math

import pytest
import torch

from turboquant.cartesian import TurboQuantMSE


def test_rejects_invalid_bits():
    with pytest.raises(ValueError):
        TurboQuantMSE(d=4, bits=0)


def test_round_trip_reduces_error_with_more_bits():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    errors = []
    for bits in (1, 2, 3, 4):
        q = TurboQuantMSE(d=32, bits=bits, seed=1)
        indices, norm = q.quantize(x)
        x_hat = q.dequantize(indices, norm)
        errors.append(((x - x_hat) ** 2).sum().item())
    assert errors == sorted(errors, reverse=True)


def test_rotation_is_orthogonal_round_trip():
    q = TurboQuantMSE(d=16, bits=3, seed=2)
    x = torch.randn(3, 16)
    assert torch.allclose(q.unrotate(q.rotate(x)), x, atol=1e-4)


def test_worked_example_d4_b1_matches_primer():
    # Primer's hand-worked example (#worked-simple): input x=(1,0,0,0), a
    # concrete (normalized) Hadamard matrix as the orthogonal transform,
    # b=1. We inject that exact Hadamard rotation in place of the package's
    # random QR rotation so the deterministic worked numbers are
    # reproducible: after rotation every coordinate is 0.5, centroids are
    # +/- sqrt(2/pi)/sqrt(4) = +/-0.39894, reconstruction is
    # (0.79788, 0, 0, 0), and the squared error is ~0.20212.
    q = TurboQuantMSE(d=4, bits=1, seed=0)
    hadamard = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    ) * 0.5
    q.rotation = hadamard

    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rotated = q.rotate(x / x.norm())
    assert torch.allclose(rotated, torch.full((1, 4), 0.5), atol=1e-6)

    expected_centroid = math.sqrt(2 / math.pi) / math.sqrt(4)
    assert math.isclose(q.codebook.centroids[1].item(), expected_centroid, rel_tol=1e-3)

    indices, norm = q.quantize(x)
    x_hat = q.dequantize(indices, norm)
    assert torch.allclose(x_hat, torch.tensor([[0.79788, 0.0, 0.0, 0.0]]), atol=1e-3)

    squared_error = ((x - x_hat) ** 2).sum().item()
    assert math.isclose(squared_error, 0.20212, abs_tol=2e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_cartesian_mse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turboquant.cartesian'`

- [ ] **Step 3: Write `cartesian.py`** (Algorithm 1 only for this task; `TurboQuantProd` is added in Task 8)

```python
"""Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod), verbatim."""

from .codebook import Codebook
from .distributions import beta_coordinate_density
from .rotation import generate_rotation_matrix

import torch


class TurboQuantMSE:
    """Algorithm 1: rotate, per-coordinate Lloyd-Max quantize, unrotate."""

    def __init__(self, d: int, bits: int, seed: int = 0):
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        self.d = d
        self.bits = bits
        self.rotation = generate_rotation_matrix(d, seed)
        self.codebook = Codebook.for_density(beta_coordinate_density(d), bits)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.rotation.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.rotation

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: any nonzero vector(s), shape (..., d). Returns (indices, norm)."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        unit = x / norm.clamp_min(1e-12)
        y = self.rotate(unit)
        indices = self.codebook.quantize(y)
        return indices, norm.squeeze(-1)

    def dequantize(self, indices: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        y_hat = self.codebook.dequantize(indices)
        x_hat = self.unrotate(y_hat)
        return x_hat * norm.unsqueeze(-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_cartesian_mse.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/cartesian.py turbo-quant/tests/test_cartesian_mse.py
git commit -m "$(cat <<'EOF'
Add TurboQuantMSE implementing Algorithm 1 verbatim

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 8: `cartesian.py` — `TurboQuantProd` (Algorithm 2)

**Files:**
- Modify: `turbo-quant/turboquant/cartesian.py` (append `TurboQuantProd`)
- Test: `turbo-quant/tests/test_cartesian_prod.py`

**Interfaces:**
- Consumes: `TurboQuantMSE` (this file, Task 7), `generate_qjl_matrix`, `sign_quantize` (Task 6).
- Produces: `TurboQuantProd(d: int, bits: int, seed: int = 0)` with `.quantize(x: Tensor) -> dict` (`indices`, `norm`, `qjl_signs`, `residual_norm`), `.dequantize(compressed: dict) -> Tensor`, `.inner_product(y: Tensor, compressed: dict) -> Tensor`. `bits < 2` raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

`turbo-quant/tests/test_cartesian_prod.py`:
```python
import pytest
import torch

from turboquant.cartesian import TurboQuantProd


def test_rejects_bits_below_two():
    with pytest.raises(ValueError):
        TurboQuantProd(d=8, bits=1)


def test_quantize_dequantize_round_trip_shapes():
    q = TurboQuantProd(d=16, bits=3, seed=0)
    x = torch.randn(5, 16)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert x_hat.shape == x.shape


def test_inner_product_is_empirically_unbiased():
    torch.manual_seed(0)
    d = 64
    q = TurboQuantProd(d=d, bits=2, seed=3)

    estimates = []
    truths = []
    for _ in range(300):
        x = torch.randn(d)
        y = torch.randn(d)
        compressed = q.quantize(x.unsqueeze(0))
        estimate = q.inner_product(y.unsqueeze(0), compressed).item()
        estimates.append(estimate)
        truths.append(torch.dot(x, y).item())

    mean_estimate = sum(estimates) / len(estimates)
    mean_truth = sum(truths) / len(truths)
    # Both should be close to 0 in expectation (independent random x, y);
    # check the estimator doesn't introduce the paper's ~36% shrink bias
    # by comparing average absolute deviation instead.
    bias = abs(mean_estimate - mean_truth)
    assert bias < 0.15


def test_inner_product_matches_dequantized_dot_for_pure_mse_term():
    # With residual_norm forced to 0, inner_product's QJL term vanishes and
    # it must equal a plain dot product against the MSE reconstruction.
    q = TurboQuantProd(d=16, bits=2, seed=1)
    x = torch.randn(1, 16)
    y = torch.randn(1, 16)
    compressed = q.quantize(x)
    compressed["residual_norm"] = torch.zeros_like(compressed["residual_norm"])

    estimate = q.inner_product(y, compressed)
    x_hat_mse = q.mse.dequantize(compressed["indices"], compressed["norm"])
    expected = (y * x_hat_mse).sum(dim=-1)
    assert torch.allclose(estimate, expected, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_cartesian_prod.py -v`
Expected: FAIL — `ImportError: cannot import name 'TurboQuantProd' from 'turboquant.cartesian'`

- [ ] **Step 3: Append `TurboQuantProd` to `cartesian.py`**

Add these imports at the top of `turbo-quant/turboquant/cartesian.py`:

```python
import math

from .qjl import generate_qjl_matrix, sign_quantize
```

Append this class at the end of the file:

```python
class TurboQuantProd:
    """Algorithm 2: (bits-1)-bit MSE stage + 1-bit QJL sign-quantized residual,
    for unbiased inner-product estimation."""

    def __init__(self, d: int, bits: int, seed: int = 0):
        if bits < 2:
            raise ValueError(f"bits must be >= 2 for TurboQuantProd, got {bits}")
        self.d = d
        self.bits = bits
        self.mse = TurboQuantMSE(d, bits - 1, seed=seed)
        self.qjl_matrix = generate_qjl_matrix(d, seed=seed + 1)

    def quantize(self, x: torch.Tensor) -> dict:
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
        correction_scale = math.sqrt(math.pi / 2) / self.d
        return (
            compressed["residual_norm"].unsqueeze(-1)
            * correction_scale
            * (compressed["qjl_signs"] @ self.qjl_matrix)
        )

    def dequantize(self, compressed: dict) -> torch.Tensor:
        x_hat_mse = self.mse.dequantize(compressed["indices"], compressed["norm"])
        return x_hat_mse + self._qjl_correction(compressed)

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Unbiased estimate of <x, y> using compressed x (Algorithm 2's payoff)."""
        x_hat_mse = self.mse.dequantize(compressed["indices"], compressed["norm"])
        term1 = (y * x_hat_mse).sum(dim=-1)

        y_projected = y @ self.qjl_matrix.T
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)
        correction_scale = math.sqrt(math.pi / 2) / self.d
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip

        return term1 + term2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_cartesian_prod.py tests/test_cartesian_mse.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/cartesian.py turbo-quant/tests/test_cartesian_prod.py
git commit -m "$(cat <<'EOF'
Add TurboQuantProd implementing Algorithm 2's QJL correction

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 9: `polar.py` — `PolarQuant`

**Files:**
- Create: `turbo-quant/turboquant/polar.py`
- Test: `turbo-quant/tests/test_polar.py`

**Interfaces:**
- Consumes: `generate_rotation_matrix` (Task 2), `polar_angle_density` (Task 3), `Codebook` (Task 5).
- Produces: `PolarQuant(d: int, bits: int, seed: int = 0)` with `.quantize(x: Tensor) -> dict` (`angle_indices: list[Tensor]`, `final_radius: Tensor`), `.dequantize(compressed: dict) -> Tensor`. `d` not a power of 2 (or `< 2`) raises `ValueError`; `bits < 1` raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

`turbo-quant/tests/test_polar.py`:
```python
import pytest
import torch

from turboquant.polar import PolarQuant


def test_rejects_non_power_of_two_dimension():
    with pytest.raises(ValueError):
        PolarQuant(d=6, bits=2)


def test_rejects_invalid_bits():
    with pytest.raises(ValueError):
        PolarQuant(d=8, bits=0)


def test_number_of_levels_and_angle_indices():
    q = PolarQuant(d=8, bits=2, seed=0)
    assert q.n_levels == 3  # log2(8)
    x = torch.randn(2, 8)
    compressed = q.quantize(x)
    assert len(compressed["angle_indices"]) == 3
    assert compressed["angle_indices"][0].shape == (2, 4)  # d/2 angles at level 1
    assert compressed["angle_indices"][1].shape == (2, 2)  # d/4 at level 2
    assert compressed["angle_indices"][2].shape == (2, 1)  # d/8 at level 3
    assert compressed["final_radius"].shape == (2,)


def test_round_trip_shape():
    q = PolarQuant(d=16, bits=3, seed=1)
    x = torch.randn(4, 16)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert x_hat.shape == x.shape


def test_round_trip_error_decreases_with_more_bits():
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    errors = []
    for bits in (1, 2, 3, 4):
        q = PolarQuant(d=16, bits=bits, seed=2)
        compressed = q.quantize(x)
        x_hat = q.dequantize(compressed)
        errors.append(((x - x_hat) ** 2).sum().item())
    assert errors == sorted(errors, reverse=True)


def test_norm_is_approximately_preserved():
    # Rotation preserves norm exactly; quantization error should keep the
    # reconstructed norm reasonably close to the original at moderate bits.
    q = PolarQuant(d=16, bits=4, seed=3)
    x = torch.randn(1, 16)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert torch.allclose(x.norm(), x_hat.norm(), rtol=0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_polar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turboquant.polar'`

- [ ] **Step 3: Write `polar.py`**

```python
"""PolarQuant: recursive Cartesian -> polar decomposition with per-level
Lloyd-Max codebooks on the sin-power angle densities."""

import math

import torch

from .codebook import Codebook
from .distributions import polar_angle_density
from .rotation import generate_rotation_matrix


class PolarQuant:
    def __init__(self, d: int, bits: int, seed: int = 0):
        if d < 2 or (d & (d - 1)) != 0:
            raise ValueError(f"d must be a power of 2 and >= 2, got {d}")
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")

        self.d = d
        self.bits = bits
        self.n_levels = int(math.log2(d))
        self.rotation = generate_rotation_matrix(d, seed)
        self.codebooks = [
            Codebook.for_density(polar_angle_density(level), bits)
            for level in range(1, self.n_levels + 1)
        ]

    @staticmethod
    def _decompose(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pair up the last dimension of v: (..., 2k) -> radii (..., k), angles (..., k)."""
        evens = v[..., 0::2]
        odds = v[..., 1::2]
        radii = torch.sqrt(evens ** 2 + odds ** 2)
        angles = torch.atan2(odds, evens) % (2 * math.pi)
        return radii, angles

    def quantize(self, x: torch.Tensor) -> dict:
        v = x @ self.rotation.T
        angle_indices = []
        for level in range(self.n_levels):
            v, angles = self._decompose(v)
            angle_indices.append(self.codebooks[level].quantize(angles))
        return {"angle_indices": angle_indices, "final_radius": v.squeeze(-1)}

    def dequantize(self, compressed: dict) -> torch.Tensor:
        v = compressed["final_radius"].unsqueeze(-1)
        for level in reversed(range(self.n_levels)):
            angles = self.codebooks[level].dequantize(compressed["angle_indices"][level])
            evens = v * torch.cos(angles)
            odds = v * torch.sin(angles)
            v = torch.stack([evens, odds], dim=-1).flatten(start_dim=-2)
        return v @ self.rotation
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_polar.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/polar.py turbo-quant/tests/test_polar.py
git commit -m "$(cat <<'EOF'
Add PolarQuant recursive polar decomposition

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 10: Public package API and full-suite smoke test

**Files:**
- Modify: `turbo-quant/turboquant/__init__.py`
- Test: `turbo-quant/tests/test_public_api.py`

**Interfaces:**
- Produces: `from turboquant import TurboQuantMSE, TurboQuantProd, PolarQuant` works directly from the package root.

- [ ] **Step 1: Write the failing test**

`turbo-quant/tests/test_public_api.py`:
```python
def test_public_classes_importable_from_package_root():
    from turboquant import PolarQuant, TurboQuantMSE, TurboQuantProd

    assert TurboQuantMSE is not None
    assert TurboQuantProd is not None
    assert PolarQuant is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_public_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'TurboQuantMSE' from 'turboquant'`

- [ ] **Step 3: Write `__init__.py`**

```python
"""turboquant: paper-accurate TurboQuant (Algorithms 1 & 2) and PolarQuant."""

from .cartesian import TurboQuantMSE, TurboQuantProd
from .polar import PolarQuant

__all__ = ["TurboQuantMSE", "TurboQuantProd", "PolarQuant"]
```

- [ ] **Step 4: Run the full test suite**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/ -v`
Expected: PASS — every test from Tasks 2-10 passes.

- [ ] **Step 5: Commit**

```bash
git add turbo-quant/turboquant/__init__.py turbo-quant/tests/test_public_api.py
git commit -m "$(cat <<'EOF'
Export TurboQuantMSE, TurboQuantProd, PolarQuant from package root

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 11: Remove old implementation, update README

**Files:**
- Delete: `turbo-quant/turboquant.py`, `turbo-quant/compressors.py`, `turbo-quant/lloyd_max.py`, `turbo-quant/test_algorithm.py`, `turbo-quant/evaluate.py`
- Modify: `turbo-quant/README.md`

**Interfaces:**
- None (documentation and cleanup only).

- [ ] **Step 1: Verify nothing outside `turbo-quant/` imports the old modules**

Run: `cd "C:/Vijay/PyCode/Research" && grep -rl "from turboquant import\|import turboquant\b" --include="*.py" . | grep -v "^./turbo-quant/turboquant/" | grep -v "^./turbo-quant/tests/"`
Expected: no output (old flat-module imports, e.g. `import compressors`, only existed inside `turbo-quant/` itself and are about to be deleted).

- [ ] **Step 2: Delete the old files**

```bash
cd "C:/Vijay/PyCode/Research"
git rm turbo-quant/turboquant.py turbo-quant/compressors.py turbo-quant/lloyd_max.py turbo-quant/test_algorithm.py turbo-quant/evaluate.py
rm -rf turbo-quant/__pycache__
```

- [ ] **Step 3: Rewrite the README's usage section**

Read the current `turbo-quant/README.md` first (`Read turbo-quant/README.md`) to see what structure/tone to preserve, then replace its algorithm-usage / API section with:

```markdown
## Installation

    cd turbo-quant
    pip install -e ".[test]"

## Usage

    from turboquant import TurboQuantMSE, TurboQuantProd, PolarQuant

    # Algorithm 1: MSE-optimal quantizer
    q = TurboQuantMSE(d=128, bits=4, seed=0)
    indices, norm = q.quantize(x)          # x: (..., 128)
    x_hat = q.dequantize(indices, norm)

    # Algorithm 2: unbiased inner-product quantizer
    q = TurboQuantProd(d=128, bits=4, seed=0)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    estimate = q.inner_product(y, compressed)   # unbiased estimate of <x, y>

    # PolarQuant: recursive polar-coordinate alternative (d must be a power of 2)
    q = PolarQuant(d=128, bits=4, seed=0)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)

Every numerical choice in `turboquant/` is exact per the papers: true
Haar-random rotation via QR decomposition (no Hadamard-transform shortcut),
Lloyd-Max solved against the exact Beta / sin-power densities (no Gaussian
approximation). See `turboquant-primer.html` for a full interactive
walkthrough of the math, and `docs/superpowers/specs/2026-08-27-turboquant-redesign-design.md`
for the design rationale.

## Testing against real models

    pip install -e ".[examples]"
    python examples/run_benchmark.py --smoke-test
    python examples/run_benchmark.py --model Qwen/Qwen2.5-0.5B --algorithm mse prod --bits 1 2 3 4
```

Keep whatever pre-existing top-of-file description/motivation section in the README makes sense to retain; replace only the parts that document the old `turboquant.py`/`compressors.py` API.

- [ ] **Step 4: Verify the package still imports and tests still pass after deletion**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/ -v`
Expected: PASS — deleting the old files must not have broken the new package (it doesn't import from them).

- [ ] **Step 5: Commit**

```bash
cd "C:/Vijay/PyCode/Research"
git add turbo-quant/README.md
git commit -m "$(cat <<'EOF'
Remove superseded flat-file implementation, document new package

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 12: `examples/kv_cache_hook.py` — HF KV-cache round-trip

**Files:**
- Create: `turbo-quant/examples/kv_cache_hook.py`
- Test: `turbo-quant/tests/test_kv_cache_hook.py`

**Interfaces:**
- Consumes: any object with `.quantize(x) -> tuple | dict` and `.dequantize(*args) -> Tensor` matching `TurboQuantMSE`/`TurboQuantProd`/`PolarQuant`'s interfaces from Tasks 7-9.
- Produces: `QuantizingCache(key_quantizer, value_quantizer)`, a `transformers.cache_utils.DynamicCache` subclass overriding `.update(...)` to round-trip every key/value tensor through the given quantizers before caching.

- [ ] **Step 1: Install the `examples` extra (needed for `transformers`)**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pip install -e ".[examples,test]"`
Expected: installs successfully.

- [ ] **Step 2: Write the failing test**

`turbo-quant/tests/test_kv_cache_hook.py`:
```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from kv_cache_hook import QuantizingCache
from turboquant import TurboQuantMSE


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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_kv_cache_hook.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kv_cache_hook'`

- [ ] **Step 4: Write `examples/kv_cache_hook.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant" && pytest tests/test_kv_cache_hook.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add turbo-quant/examples/kv_cache_hook.py turbo-quant/tests/test_kv_cache_hook.py
git commit -m "$(cat <<'EOF'
Add QuantizingCache to round-trip HF KV cache through turboquant

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

---

## Task 13: `examples/run_benchmark.py` — perplexity vs. compression on real models

**Files:**
- Create: `turbo-quant/examples/run_benchmark.py`

**Interfaces:**
- Consumes: `QuantizingCache` (Task 12), `TurboQuantMSE`/`TurboQuantProd`/`PolarQuant` (package root).
- Produces: a runnable CLI script; no importable interface consumed by later tasks (this is the final task).

- [ ] **Step 1: Write `examples/run_benchmark.py`**

```python
"""Benchmark turboquant algorithms against real LLMs' KV caches.

Usage:
    python run_benchmark.py --smoke-test
    python run_benchmark.py --model Qwen/Qwen2.5-0.5B --algorithm mse prod --bits 1 2 3 4
    python run_benchmark.py --model google/gemma-2-2b --algorithm mse polar --bits 2 4
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache_hook import QuantizingCache
from turboquant import PolarQuant, TurboQuantMSE, TurboQuantProd

ALGORITHMS = {
    "mse": TurboQuantMSE,
    "prod": TurboQuantProd,
    "polar": PolarQuant,
}

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. " * 20


def compression_ratio(head_dim: int, bits: int, algorithm: str) -> float:
    """Analytical compression ratio (index bits vs. fp16), not actual bit-packing."""
    fp16_bits = head_dim * 16
    if algorithm == "prod":
        packed_bits = head_dim * (bits - 1) + head_dim  # (bits-1)-bit MSE + 1 QJL bit/coord
    else:
        packed_bits = head_dim * bits
    packed_bits += 16  # one fp16 norm/radius scalar per vector
    return fp16_bits / packed_bits


@torch.no_grad()
def measure_perplexity(model, tokenizer, text: str, cache=None) -> float:
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model(input_ids, past_key_values=cache, labels=input_ids, use_cache=cache is not None)
    return math.exp(outputs.loss.item())


def head_dim_of(model) -> int:
    config = model.config
    if hasattr(config, "head_dim") and config.head_dim:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def run(model_name: str, algorithms: list[str], bits_list: list[int], repeat: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    text = SAMPLE_TEXT * repeat
    d = head_dim_of(model)

    baseline_ppl = measure_perplexity(model, tokenizer, text)
    print(f"{model_name} (head_dim={d}) baseline perplexity: {baseline_ppl:.3f}")

    for algorithm in algorithms:
        cls = ALGORITHMS[algorithm]
        for bits in bits_list:
            if algorithm == "prod" and bits < 2:
                print(f"  {algorithm} b={bits}: skipped (prod requires bits >= 2)")
                continue
            key_q = cls(d, bits, seed=1)
            val_q = cls(d, bits, seed=2)
            cache = QuantizingCache(key_quantizer=key_q, value_quantizer=val_q)
            ppl = measure_perplexity(model, tokenizer, text, cache=cache)
            ratio = compression_ratio(d, bits, algorithm)
            print(
                f"  {algorithm} b={bits}: perplexity={ppl:.3f} "
                f"(+{ppl - baseline_ppl:+.3f} vs baseline), compression={ratio:.2f}x"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--algorithm", nargs="+", default=["mse", "prod"], choices=list(ALGORITHMS))
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--repeat", type=int, default=10, help="repeat the sample text N times")
    parser.add_argument("--smoke-test", action="store_true", help="tiny model, one config, for CI-free verification")
    args = parser.parse_args()

    if args.smoke_test:
        run("sshleifer/tiny-gpt2", ["mse"], [2], repeat=1)
    else:
        run(args.model, args.algorithm, args.bits, repeat=args.repeat)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant/examples" && python run_benchmark.py --smoke-test`
Expected: downloads the tiny `sshleifer/tiny-gpt2` checkpoint (small, fast) and prints a baseline perplexity line plus one `mse b=2` line without raising an exception. If `DynamicCache.update`'s signature differs for this model's attention implementation (transformers version skew), the traceback will point at the exact mismatch — fix `QuantizingCache.update`'s signature/behavior to match the installed `transformers` version's `Cache.update` contract before proceeding.

- [ ] **Step 3: Commit**

```bash
git add turbo-quant/examples/run_benchmark.py
git commit -m "$(cat <<'EOF'
Add run_benchmark.py to measure perplexity vs compression on real LLMs

Co-Authored-By: WOZCODE <contact@withwoz.com>
EOF
)"
```

- [ ] **Step 4: (Manual, not automated) Run against Qwen2.5-0.5B and Gemma-2-2b**

Run: `cd "C:/Vijay/PyCode/Research/turbo-quant/examples" && python run_benchmark.py --model Qwen/Qwen2.5-0.5B --algorithm mse prod polar --bits 1 2 3 4`
Run: `cd "C:/Vijay/PyCode/Research/turbo-quant/examples" && python run_benchmark.py --model google/gemma-2-2b --algorithm mse prod polar --bits 1 2 3 4`

These download multi-hundred-MB to multi-GB checkpoints and are not part of the automated test suite — run them manually to get the actual perplexity/compression numbers the user asked for, and report the results.
