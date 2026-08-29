# Sequential Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a paper-accurate `seqattention` package (Sequential Attention feature selection, ICLR 2023) inside `seq-attention/`, with a numerical OMP-equivalence demo and a benchmark reproducing the paper's Table 2 results on MNIST/Fashion-MNIST/ISOLET.

**Architecture:** A core algorithm library (`seqattention/`: `mask.py`, `models.py`, `selector.py`, `onepass.py`, `omp.py`) generalized over any `nn.Module` with a `.mask` attribute, so the same selection code drives both the linear-regression OMP-equivalence demo and the MLP image/audio-feature classification benchmark. `examples/` holds two runnable scripts (`run_omp_equivalence.py`, `run_benchmark.py`) plus their shared data/logging utilities.

**Tech Stack:** PyTorch (matches repo convention), `torchvision` for MNIST/Fashion-MNIST, `requests` for the ISOLET download, `pytest` for tests. Runs locally on the project's RTX 4070 Laptop GPU (or CPU — nothing here requires CUDA).

**Spec:** `docs/superpowers/specs/2026-08-28-sequential-attention-design.md`

## Global Constraints

- All new files live under `seq-attention/` (or edits to the root `pyproject.toml` to wire it into the workspace) — nothing outside that folder is touched, per the spec's scope.
- No transformer/LLM attention code, no SequentialAttention++, no hyperparameter search framework, no multi-GPU (spec's Non-Goals).
- Package name: `seqattention`. Module/file layout matches the spec's Architecture section exactly.
- Style follows `turbo-quant/`'s conventions: module docstring at top of each file, `torch.Tensor` type hints, small focused files, `tests/` mirrors `turboquant/tests/`'s flat pytest style (no test classes).
- `python -m pytest` must pass for every test file added, run from `seq-attention/` before each commit that touches it.

---

### Task 1: Project scaffolding and workspace wiring

**Files:**
- Create: `seq-attention/pyproject.toml`
- Create: `seq-attention/seqattention/__init__.py`
- Create: `seq-attention/tests/__init__.py`
- Create: `seq-attention/examples/__init__.py`
- Modify: `pyproject.toml:20` (`[tool.uv.workspace] members`) and `pyproject.toml:23-25` (`[tool.uv.sources]`)
- Modify: `.gitignore`

**Interfaces:**
- Produces: an installable `seqattention` package (empty for now) importable from `seq-attention/`, wired into the uv workspace so `uv sync` picks it up.

- [ ] **Step 1: Create `seq-attention/pyproject.toml`**

```toml
[project]
name = "seqattention"
version = "0.1.0"
description = "Paper-accurate implementation of Sequential Attention feature selection"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
]

[project.optional-dependencies]
examples = [
    "torchvision>=0.20",
    "requests>=2.30",
]
test = [
    "pytest>=7.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["seqattention*"]
```

- [ ] **Step 2: Create empty package/test/example init files**

`seq-attention/seqattention/__init__.py`:
```python
"""Paper-accurate implementation of Sequential Attention feature selection
(Yasuda, Bateni, Chen, Fahrbach, Fu, Mirrokni, ICLR 2023, arXiv:2209.14881)."""
```

`seq-attention/tests/__init__.py`: empty file.

`seq-attention/examples/__init__.py`: empty file.

- [ ] **Step 3: Wire `seq-attention` into the root workspace**

In `pyproject.toml`, change:
```toml
[tool.uv.workspace]
members = ["turbo-quant"]
```
to:
```toml
[tool.uv.workspace]
members = ["turbo-quant", "seq-attention"]
```

And change:
```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
turboquant = { workspace = true }
```
to:
```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
turboquant = { workspace = true }
seqattention = { workspace = true }
```

- [ ] **Step 4: Add cache/output dirs to `.gitignore`**

Append to `.gitignore`:
```
# seq-attention example artifacts
seq-attention/examples/data_cache/
```

- [ ] **Step 5: Verify the workspace resolves**

Run: `cd "C:/Vijay/PyCode/Research" && uv sync`
Expected: completes without error; `seqattention` appears as an installed workspace member (`uv pip list | grep -i seqattention` shows it, editable).

- [ ] **Step 6: Commit**

```bash
git add seq-attention/pyproject.toml seq-attention/seqattention/__init__.py seq-attention/tests/__init__.py seq-attention/examples/__init__.py pyproject.toml .gitignore uv.lock
git commit -m "seq-attention: scaffold seqattention package and wire into workspace"
```

---

### Task 2: Sequential Attention mask (`mask.py`)

**Files:**
- Create: `seq-attention/seqattention/mask.py`
- Test: `seq-attention/tests/test_mask.py`

**Interfaces:**
- Produces: `SequentialAttentionMask(num_features: int, seed: int = 0)`, an `nn.Module` with:
  - `.attention_logits: nn.Parameter` shape `(num_features,)`
  - `.overparam_weight: nn.Parameter` shape `(num_features,)`
  - `.selected: torch.BoolTensor` buffer shape `(num_features,)`
  - `.softmax_mask() -> torch.Tensor` shape `(num_features,)`
  - `.gate() -> torch.Tensor` shape `(num_features,)`
  - `.forward(x: torch.Tensor) -> torch.Tensor` — elementwise `x * self.gate()`, broadcasting over any leading batch dims
  - `.select(idx: int) -> None` — pins feature `idx` into `selected`
  - `.reset_logits(seed: int | None = None) -> None` — reinitializes `attention_logits` in place, used by `onepass.py`

- [ ] **Step 1: Write the failing tests**

`seq-attention/tests/test_mask.py`:
```python
import torch

from seqattention.mask import SequentialAttentionMask


def test_unselected_softmax_sums_to_one():
    mask = SequentialAttentionMask(num_features=5, seed=0)
    m = mask.softmax_mask()
    assert torch.isclose(m.sum(), torch.tensor(5.0), atol=1e-4) is False  # not all weight-1 yet
    assert torch.isclose(m[~mask.selected].sum(), torch.tensor(1.0), atol=1e-5)


def test_selected_features_pinned_to_one():
    mask = SequentialAttentionMask(num_features=5, seed=0)
    mask.select(2)
    m = mask.softmax_mask()
    assert m[2].item() == 1.0
    assert torch.isclose(m[~mask.selected].sum(), torch.tensor(1.0), atol=1e-5)


def test_gate_is_hadamard_product_of_mask_and_weight():
    mask = SequentialAttentionMask(num_features=4, seed=0)
    with torch.no_grad():
        mask.overparam_weight.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]))
    m = mask.softmax_mask()
    g = mask.gate()
    assert torch.allclose(g, m * mask.overparam_weight)


def test_forward_multiplies_input_by_gate():
    mask = SequentialAttentionMask(num_features=3, seed=0)
    x = torch.ones(2, 3)
    out = mask(x)
    assert torch.allclose(out, mask.gate().unsqueeze(0).expand(2, 3))


def test_select_is_idempotent_and_excludes_from_softmax_domain():
    mask = SequentialAttentionMask(num_features=4, seed=0)
    mask.select(0)
    mask.select(0)
    assert mask.selected.sum().item() == 1
    m = mask.softmax_mask()
    assert torch.isclose(m[[1, 2, 3]].sum(), torch.tensor(1.0), atol=1e-5)


def test_reset_logits_changes_values_deterministically_by_seed():
    mask = SequentialAttentionMask(num_features=4, seed=0)
    before = mask.attention_logits.clone()
    mask.reset_logits(seed=1)
    after_seed1 = mask.attention_logits.clone()
    mask.reset_logits(seed=1)
    after_seed1_again = mask.attention_logits.clone()
    assert not torch.allclose(before, after_seed1)
    assert torch.allclose(after_seed1, after_seed1_again)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd seq-attention && python -m pytest tests/test_mask.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seqattention.mask'`

- [ ] **Step 3: Implement `mask.py`**

```python
"""The Sequential Attention mask: selected features get a fixed weight of 1,
unselected features compete via softmax over their attention logits, and the
result is combined with a second learned weight vector via a Hadamard
product -- this overparameterization is what induces implicit L1-style
sparsity (Yasuda et al., ICLR 2023, arXiv:2209.14881, Section 3)."""

import torch


class SequentialAttentionMask(torch.nn.Module):
    def __init__(self, num_features: int, seed: int = 0):
        super().__init__()
        self.num_features = num_features
        generator = torch.Generator().manual_seed(seed)
        self.attention_logits = torch.nn.Parameter(
            torch.randn(num_features, generator=generator) * 0.01
        )
        self.overparam_weight = torch.nn.Parameter(torch.ones(num_features))
        self.register_buffer("selected", torch.zeros(num_features, dtype=torch.bool))

    def softmax_mask(self) -> torch.Tensor:
        m = torch.zeros_like(self.attention_logits)
        m = torch.where(self.selected, torch.ones_like(m), m)
        unselected = ~self.selected
        if unselected.any():
            softmaxed = torch.softmax(self.attention_logits[unselected], dim=0)
            m = m.masked_scatter(unselected, softmaxed)
        return m

    def gate(self) -> torch.Tensor:
        return self.softmax_mask() * self.overparam_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate()

    def select(self, idx: int) -> None:
        self.selected[idx] = True

    def reset_logits(self, seed: int | None = None) -> None:
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        with torch.no_grad():
            self.attention_logits.copy_(
                torch.randn(self.num_features, generator=generator) * 0.01
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd seq-attention && python -m pytest tests/test_mask.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add seq-attention/seqattention/mask.py seq-attention/tests/test_mask.py
git commit -m "seq-attention: implement the Sequential Attention mask"
```

---

### Task 3: Model wrappers (`models.py`)

**Files:**
- Create: `seq-attention/seqattention/models.py`
- Test: `seq-attention/tests/test_models.py`

**Interfaces:**
- Consumes: `SequentialAttentionMask` from Task 2.
- Produces: two `nn.Module` classes, both exposing a `.mask: SequentialAttentionMask` attribute (this is the contract `selector.py`/`onepass.py` in Tasks 4-5 rely on):
  - `LinearRegressionModel(num_features: int, seed: int = 0)` — `.forward(x) -> torch.Tensor` shape `(batch,)`
  - `AttentionGatedMLP(num_features: int, hidden_dim: int, num_classes: int, seed: int = 0)` — `.forward(x) -> torch.Tensor` shape `(batch, num_classes)`, `.body: nn.Sequential` (the non-mask parameters)

- [ ] **Step 1: Write the failing tests**

`seq-attention/tests/test_models.py`:
```python
import torch

from seqattention.models import AttentionGatedMLP, LinearRegressionModel


def test_linear_regression_model_shape_and_mask_attribute():
    model = LinearRegressionModel(num_features=6, seed=0)
    x = torch.randn(10, 6)
    out = model(x)
    assert out.shape == (10,)
    assert hasattr(model, "mask")
    assert model.mask.num_features == 6


def test_attention_gated_mlp_shape_and_mask_attribute():
    model = AttentionGatedMLP(num_features=8, hidden_dim=16, num_classes=3, seed=0)
    x = torch.randn(5, 8)
    out = model(x)
    assert out.shape == (5, 3)
    assert hasattr(model, "mask")
    assert model.mask.num_features == 8


def test_attention_gated_mlp_body_excludes_mask_parameters():
    model = AttentionGatedMLP(num_features=4, hidden_dim=8, num_classes=2, seed=0)
    body_params = set(model.body.parameters())
    assert model.mask.attention_logits not in body_params
    assert model.mask.overparam_weight not in body_params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd seq-attention && python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seqattention.models'`

- [ ] **Step 3: Implement `models.py`**

```python
"""Model wrappers whose input layer is gated by a SequentialAttentionMask.
Both classes expose a `.mask` attribute -- the contract selector.py and
onepass.py rely on to find and update the attention logits / selected set
regardless of what the rest of the model looks like."""

import torch

from .mask import SequentialAttentionMask


class LinearRegressionModel(torch.nn.Module):
    """Mask-gated linear regression, used for the OMP-equivalence demo where
    Theorem 1.1/3.3 apply directly."""

    def __init__(self, num_features: int, seed: int = 0):
        super().__init__()
        self.mask = SequentialAttentionMask(num_features, seed=seed)
        self.linear = torch.nn.Linear(num_features, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.mask(x)).squeeze(-1)


class AttentionGatedMLP(torch.nn.Module):
    """Mask-gated MLP: attention-gated input layer followed by a standard
    MLP body, matching the paper's experimental architecture for the
    MNIST / Fashion-MNIST / ISOLET benchmarks."""

    def __init__(self, num_features: int, hidden_dim: int, num_classes: int, seed: int = 0):
        super().__init__()
        self.mask = SequentialAttentionMask(num_features, seed=seed)
        self.body = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.mask(x))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd seq-attention && python -m pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add seq-attention/seqattention/models.py seq-attention/tests/test_models.py
git commit -m "seq-attention: add mask-gated linear regression and MLP models"
```

---

### Task 4: Algorithm 1 — naive per-phase greedy selection (`selector.py`)

**Files:**
- Create: `seq-attention/seqattention/selector.py`
- Test: `seq-attention/tests/test_selector.py`

**Interfaces:**
- Consumes: any `nn.Module` with a `.mask: SequentialAttentionMask` attribute (Task 3's contract).
- Produces: `select_features_naive(model_factory: Callable[[int], torch.nn.Module], loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], X: torch.Tensor, y: torch.Tensor, k: int, train_steps: int = 200, lr: float = 0.05, seed: int = 0) -> list[int]` — used directly by Task 6's equivalence tests/demo, and by Task 5's onepass test as the "ground truth" baseline it must match.

- [ ] **Step 1: Write the failing test**

`seq-attention/tests/test_selector.py`:
```python
import torch

from seqattention.models import LinearRegressionModel
from seqattention.selector import select_features_naive


def _make_sparse_regression_problem(seed=0, n=200, d=10, true_idx=(1, 4, 6)):
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=generator)
    true_coef = torch.zeros(d)
    for i, idx in enumerate(true_idx):
        true_coef[idx] = 3.0 - i
    y = X @ true_coef
    return X, y, set(true_idx)


def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def test_select_features_naive_recovers_ground_truth_support():
    X, y, true_support = _make_sparse_regression_problem()
    selected = select_features_naive(
        model_factory=lambda seed: LinearRegressionModel(num_features=X.shape[1], seed=seed),
        loss_fn=mse_loss,
        X=X, y=y, k=3, train_steps=300, lr=0.1, seed=0,
    )
    assert set(selected) == true_support


def test_select_features_naive_grows_by_one_per_phase_no_repeats():
    X, y, _ = _make_sparse_regression_problem()
    selected = select_features_naive(
        model_factory=lambda seed: LinearRegressionModel(num_features=X.shape[1], seed=seed),
        loss_fn=mse_loss,
        X=X, y=y, k=4, train_steps=50, lr=0.1, seed=0,
    )
    assert len(selected) == 4
    assert len(set(selected)) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd seq-attention && python -m pytest tests/test_selector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seqattention.selector'`

- [ ] **Step 3: Implement `selector.py`**

```python
"""Algorithm 1 (Yasuda et al., ICLR 2023), applied literally: each phase
trains a fresh model from scratch, with previously-selected features
pre-pinned into the mask, then greedily adds the argmax unselected
attention logit to the selected set. This is the "k separate models"
reading of the algorithm; onepass.py implements the paper's more efficient
single-model variant and is tested against this function's output."""

from typing import Callable

import torch


def select_features_naive(
    model_factory: Callable[[int], torch.nn.Module],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    k: int,
    train_steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> list[int]:
    selected: list[int] = []
    for phase in range(k):
        model = model_factory(seed + phase)
        for idx in selected:
            model.mask.select(idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(train_steps):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            unselected = (~model.mask.selected).nonzero(as_tuple=True)[0]
            best = unselected[torch.argmax(model.mask.attention_logits[unselected])].item()
        selected.append(best)
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd seq-attention && python -m pytest tests/test_selector.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add seq-attention/seqattention/selector.py seq-attention/tests/test_selector.py
git commit -m "seq-attention: implement Algorithm 1 naive per-phase selection"
```

---

### Task 5: One-pass training trick (`onepass.py`)

**Files:**
- Create: `seq-attention/seqattention/onepass.py`
- Test: `seq-attention/tests/test_onepass.py`

**Interfaces:**
- Consumes: `select_features_naive` from Task 4 (as the comparison baseline in the test only, not a runtime dependency), `SequentialAttentionMask.reset_logits` from Task 2.
- Produces: `select_features_onepass(model_factory, loss_fn, X, y, k, train_steps_per_phase: int = 200, lr: float = 0.05, seed: int = 0) -> list[int]` — used by Task 7's OMP-equivalence demo/tests and Task 10's benchmark script.

- [ ] **Step 1: Write the failing test**

`seq-attention/tests/test_onepass.py`:
```python
import torch

from seqattention.models import LinearRegressionModel
from seqattention.onepass import select_features_onepass
from seqattention.selector import select_features_naive


def _make_sparse_regression_problem(seed=0, n=200, d=10, true_idx=(1, 4, 6)):
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=generator)
    true_coef = torch.zeros(d)
    for i, idx in enumerate(true_idx):
        true_coef[idx] = 3.0 - i
    y = X @ true_coef
    return X, y, set(true_idx)


def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def test_select_features_onepass_recovers_ground_truth_support():
    X, y, true_support = _make_sparse_regression_problem()
    selected = select_features_onepass(
        model_factory=lambda seed: LinearRegressionModel(num_features=X.shape[1], seed=seed),
        loss_fn=mse_loss,
        X=X, y=y, k=3, train_steps_per_phase=300, lr=0.1, seed=0,
    )
    assert set(selected) == true_support


def test_select_features_onepass_matches_naive_baseline_selected_set():
    X, y, _ = _make_sparse_regression_problem(true_idx=(0, 3, 5, 7))
    naive = select_features_naive(
        model_factory=lambda seed: LinearRegressionModel(num_features=X.shape[1], seed=seed),
        loss_fn=mse_loss, X=X, y=y, k=4, train_steps=300, lr=0.1, seed=0,
    )
    onepass = select_features_onepass(
        model_factory=lambda seed: LinearRegressionModel(num_features=X.shape[1], seed=seed),
        loss_fn=mse_loss, X=X, y=y, k=4, train_steps_per_phase=300, lr=0.1, seed=0,
    )
    assert set(onepass) == set(naive)


def test_select_features_onepass_persists_model_weights_across_phases():
    X, y, _ = _make_sparse_regression_problem()
    captured_models = []

    def factory(seed):
        model = LinearRegressionModel(num_features=X.shape[1], seed=seed)
        captured_models.append(model)
        return model

    select_features_onepass(
        model_factory=factory, loss_fn=mse_loss, X=X, y=y, k=3,
        train_steps_per_phase=50, lr=0.1, seed=0,
    )
    assert len(captured_models) == 1  # one persistent model, not one per phase
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd seq-attention && python -m pytest tests/test_onepass.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seqattention.onepass'`

- [ ] **Step 3: Implement `onepass.py`**

```python
"""The paper's one-pass training trick: instead of training k independent
models (selector.py's select_features_naive), train a single model across
k phases, resetting only the attention logits (not the model's other
weights, and not the growing selected set) between phases."""

from typing import Callable

import torch


def select_features_onepass(
    model_factory: Callable[[int], torch.nn.Module],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    k: int,
    train_steps_per_phase: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> list[int]:
    model = model_factory(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    selected: list[int] = []
    for phase in range(k):
        for _ in range(train_steps_per_phase):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            unselected = (~model.mask.selected).nonzero(as_tuple=True)[0]
            best = unselected[torch.argmax(model.mask.attention_logits[unselected])].item()
            model.mask.select(best)
            selected.append(best)
            model.mask.reset_logits(seed=seed + phase + 1)
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd seq-attention && python -m pytest tests/test_onepass.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add seq-attention/seqattention/onepass.py seq-attention/tests/test_onepass.py
git commit -m "seq-attention: implement the one-pass training trick"
```

---

### Task 6: OMP / Sequential LASSO reference implementations and equivalence test (`omp.py`)

**Files:**
- Create: `seq-attention/seqattention/omp.py`
- Test: `seq-attention/tests/test_omp_equivalence.py`

**Interfaces:**
- Consumes: `select_features_onepass` (Task 5), `LinearRegressionModel` (Task 3).
- Produces: `orthogonal_matching_pursuit(X: torch.Tensor, y: torch.Tensor, k: int) -> list[int]`, `sequential_lasso(X: torch.Tensor, y: torch.Tensor, k: int, lam: float = 0.01, steps: int = 500, lr: float = 0.05) -> list[int]` — both used again by Task 7's example script.

- [ ] **Step 1: Write the failing test**

`seq-attention/tests/test_omp_equivalence.py`:
```python
import torch

from seqattention.models import LinearRegressionModel
from seqattention.omp import orthogonal_matching_pursuit, sequential_lasso
from seqattention.onepass import select_features_onepass


def _orthonormal_design_problem(seed=0, n=64, d=16, true_idx=(1, 4, 6)):
    """An orthonormal-column design matrix, where Theorem 1.1/3.3 guarantee
    OMP, Sequential LASSO (as lambda -> 0), and Regularized Linear Sequential
    Attention select the same feature sequence."""
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(n, d, generator=generator)
    Q, _ = torch.linalg.qr(A)  # orthonormal columns
    true_coef = torch.zeros(d)
    for i, idx in enumerate(true_idx):
        true_coef[idx] = 4.0 - i
    y = Q @ true_coef
    return Q, y, list(true_idx)


def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def test_omp_recovers_ground_truth_support_on_orthonormal_design():
    X, y, true_idx = _orthonormal_design_problem()
    selected = orthogonal_matching_pursuit(X, y, k=3)
    assert set(selected) == set(true_idx)


def test_sequential_lasso_recovers_ground_truth_support_on_orthonormal_design():
    X, y, true_idx = _orthonormal_design_problem()
    selected = sequential_lasso(X, y, k=3, lam=1e-4)
    assert set(selected) == set(true_idx)


def test_omp_sequential_lasso_and_sequential_attention_agree_on_orthonormal_design():
    X, y, true_idx = _orthonormal_design_problem()
    omp_selected = orthogonal_matching_pursuit(X, y, k=3)
    lasso_selected = sequential_lasso(X, y, k=3, lam=1e-4)
    attention_selected = select_features_onepass(
        model_factory=lambda seed: LinearRegressionModel(num_features=X.shape[1], seed=seed),
        loss_fn=mse_loss, X=X, y=y, k=3, train_steps_per_phase=300, lr=0.1, seed=0,
    )
    assert set(omp_selected) == set(lasso_selected) == set(attention_selected) == set(true_idx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd seq-attention && python -m pytest tests/test_omp_equivalence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seqattention.omp'`

- [ ] **Step 3: Implement `omp.py`**

```python
"""Reference implementations of Orthogonal Matching Pursuit and Sequential
LASSO, independent of seqattention's selector/onepass code path, used to
numerically demonstrate the paper's proven equivalence between Regularized
Linear Sequential Attention, Sequential LASSO, and OMP (Theorem 1.1/3.3)
rather than merely asserting it."""

import torch


def orthogonal_matching_pursuit(X: torch.Tensor, y: torch.Tensor, k: int) -> list[int]:
    """Greedily picks the unselected column of X most correlated with the
    current residual, then re-solves least squares over all selected
    columns to update the residual before the next pick."""
    selected: list[int] = []
    residual = y.clone()
    for _ in range(k):
        correlations = (X.T @ residual).abs()
        correlations[selected] = -float("inf")
        best = torch.argmax(correlations).item()
        selected.append(best)
        X_s = X[:, selected]
        coef = torch.linalg.lstsq(X_s, y.unsqueeze(1)).solution.squeeze(1)
        residual = y - X_s @ coef
    return selected


def sequential_lasso(
    X: torch.Tensor, y: torch.Tensor, k: int, lam: float = 0.01, steps: int = 500, lr: float = 0.05
) -> list[int]:
    """At each phase, fits an L1-regularized regression over all features
    (already-selected features excluded from the penalty), then greedily
    adds the largest-magnitude unselected coefficient to the selected set."""
    n, d = X.shape
    selected: list[int] = []
    for _ in range(k):
        coef = torch.zeros(d, requires_grad=True)
        optimizer = torch.optim.Adam([coef], lr=lr)
        unselected_mask = torch.ones(d, dtype=torch.bool)
        unselected_mask[selected] = False
        for _ in range(steps):
            optimizer.zero_grad()
            y_hat = X @ coef
            penalty = lam * coef[unselected_mask].abs().sum()
            loss = torch.mean((y_hat - y) ** 2) + penalty
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            candidates = coef.abs().clone()
            candidates[selected] = -float("inf")
            best = torch.argmax(candidates).item()
        selected.append(best)
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd seq-attention && python -m pytest tests/test_omp_equivalence.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add seq-attention/seqattention/omp.py seq-attention/tests/test_omp_equivalence.py
git commit -m "seq-attention: implement OMP/Sequential LASSO and equivalence tests"
```

---

### Task 7: OMP-equivalence example script (`examples/run_omp_equivalence.py`)

**Files:**
- Create: `seq-attention/examples/run_omp_equivalence.py`

**Interfaces:**
- Consumes: `orthogonal_matching_pursuit`, `sequential_lasso` (Task 6), `select_features_onepass` (Task 5), `LinearRegressionModel` (Task 3).
- Produces: a standalone runnable demo (no return value consumed by later tasks).

- [ ] **Step 1: Implement the script**

```python
"""Standalone demo: builds a synthetic sparse linear regression problem and
runs plain OMP, Sequential LASSO, and Regularized Linear Sequential
Attention side by side, printing whether they select the same feature
sequence -- a numerical demonstration of Theorem 1.1/3.3's equivalence.

Run: python examples/run_omp_equivalence.py
"""

import torch

from seqattention.models import LinearRegressionModel
from seqattention.omp import orthogonal_matching_pursuit, sequential_lasso
from seqattention.onepass import select_features_onepass

SEED = 20220914
N, D, K = 64, 16, 3
TRUE_IDX = (1, 4, 6)
TRUE_COEF = (3.0, -2.0, 1.5)


def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def build_problem():
    generator = torch.Generator().manual_seed(SEED)
    A = torch.randn(N, D, generator=generator)
    X, _ = torch.linalg.qr(A)
    true_coef = torch.zeros(D)
    for idx, coef in zip(TRUE_IDX, TRUE_COEF):
        true_coef[idx] = coef
    y = X @ true_coef
    return X, y


def main():
    X, y = build_problem()
    print(f"True generating features: {sorted(TRUE_IDX)}\n")

    omp_selected = orthogonal_matching_pursuit(X, y, k=K)
    print(f"OMP selected:                 {omp_selected}")

    lasso_selected = sequential_lasso(X, y, k=K, lam=1e-4)
    print(f"Sequential LASSO selected:    {lasso_selected}")

    attention_selected = select_features_onepass(
        model_factory=lambda seed: LinearRegressionModel(num_features=D, seed=seed),
        loss_fn=mse_loss, X=X, y=y, k=K, train_steps_per_phase=300, lr=0.1, seed=0,
    )
    print(f"Sequential Attention selected: {attention_selected}")

    all_match = set(omp_selected) == set(lasso_selected) == set(attention_selected)
    print(f"\nAll three agree: {all_match}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify output**

Run: `cd seq-attention && python examples/run_omp_equivalence.py`
Expected: prints selected feature lists for all three algorithms and `All three agree: True`

- [ ] **Step 3: Commit**

```bash
git add seq-attention/examples/run_omp_equivalence.py
git commit -m "seq-attention: add OMP-equivalence demo script"
```

---

### Task 8: Dataset loaders (`examples/data.py`)

**Files:**
- Create: `seq-attention/examples/data.py`
- Test: `seq-attention/tests/test_data.py`

**Interfaces:**
- Produces: `load_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]`, `load_fashion_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]`, `load_isolet(train: bool) -> tuple[torch.Tensor, torch.Tensor]` — each returns `(X, y)` with `X` shape `(n, num_features)` float in `[0, 1]`-ish range and `y` shape `(n,)` int64 class labels. Used by Task 10's `run_benchmark.py`.

- [ ] **Step 1: Write the failing test (covers the pure-parsing logic only — no network access in tests)**

`seq-attention/tests/test_data.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from data import _parse_isolet_file  # noqa: E402


def test_parse_isolet_file_splits_features_and_zero_indexes_labels():
    text = "1.0,2.0,3.0,1.\n4.0,5.0,6.0,26.\n"
    X, y = _parse_isolet_file(text)
    assert X.shape == (2, 3)
    assert X[0].tolist() == [1.0, 2.0, 3.0]
    assert y.tolist() == [0, 25]


def test_parse_isolet_file_skips_blank_lines():
    text = "1.0,2.0,1.\n\n3.0,4.0,2.\n"
    X, y = _parse_isolet_file(text)
    assert X.shape == (2, 2)
    assert y.tolist() == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd seq-attention && python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data'`

- [ ] **Step 3: Implement `examples/data.py`**

```python
"""Dataset loading for the benchmark reproduction. MNIST and Fashion-MNIST
come from torchvision; ISOLET isn't in torchvision/torchtext, so it's
fetched once from the UCI ML Repository and cached locally."""

import zipfile
from pathlib import Path

import requests
import torch
import torchvision

CACHE_DIR = Path(__file__).parent / "data_cache"
ISOLET_URL = "https://archive.ics.uci.edu/static/public/54/isolet.zip"


def load_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(str(CACHE_DIR), train=train, download=True)
    X = ds.data.reshape(len(ds), -1).float() / 255.0
    return X, ds.targets


def load_fashion_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.FashionMNIST(str(CACHE_DIR), train=train, download=True)
    X = ds.data.reshape(len(ds), -1).float() / 255.0
    return X, ds.targets


def _parse_isolet_file(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [line.strip().split(",") for line in text.strip().splitlines() if line.strip()]
    X = torch.tensor([[float(v) for v in row[:-1]] for row in rows])
    y = torch.tensor([int(float(row[-1])) - 1 for row in rows])  # labels are 1..26
    return X, y


def load_isolet(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "isolet.zip"
    if not zip_path.exists():
        response = requests.get(ISOLET_URL, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as zf:
        name = "isolet1+2+3+4.data" if train else "isolet5.data"
        with zf.open(name) as f:
            text = f.read().decode("utf-8")
    return _parse_isolet_file(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd seq-attention && python -m pytest tests/test_data.py -v`
Expected: 2 passed

- [ ] **Step 5: Manually verify the network-dependent loaders once**

Run: `cd seq-attention && python -c "from examples.data import load_mnist, load_isolet; X, y = load_mnist(True); print(X.shape, y.shape); X2, y2 = load_isolet(True); print(X2.shape, y2.shape)"`
Expected: prints `torch.Size([60000, 784]) torch.Size([60000])` and an ISOLET shape around `torch.Size([6238, 617]) torch.Size([6238])`. If the ISOLET zip's internal filenames differ from `isolet1+2+3+4.data`/`isolet5.data`, inspect `zipfile.ZipFile(zip_path).namelist()` and adjust the `name` lookup in Step 3 accordingly, then re-run.

- [ ] **Step 6: Commit**

```bash
git add seq-attention/examples/data.py seq-attention/tests/test_data.py
git commit -m "seq-attention: add MNIST/Fashion-MNIST/ISOLET dataset loaders"
```

---

### Task 9: Results CSV logger (`examples/results_logger.py`)

**Files:**
- Create: `seq-attention/examples/results_logger.py`

**Interfaces:**
- Produces: `write_csv(rows: list[dict], path: str) -> None`, `default_output_path(prefix: str) -> str` — identical contract to `turbo-quant/examples/results_logger.py`, used by Task 10's `run_benchmark.py`.

- [ ] **Step 1: Implement `results_logger.py`** (mirrors `turbo-quant/examples/results_logger.py` exactly)

```python
"""Shared CSV result-logging helper for the seqattention benchmark script."""

import csv
from datetime import datetime, timezone
from pathlib import Path


def write_csv(rows: list[dict], path: str) -> None:
    """Write a list of flat dict rows to a CSV file, creating parent dirs as
    needed. All rows must share the same set of keys. Existing files at
    `path` are overwritten, not appended."""
    if not rows:
        raise ValueError("write_csv called with an empty rows list")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_path(prefix: str) -> str:
    """A timestamped default path under seq-attention/examples/results/."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return str(Path(__file__).parent / "results" / f"{prefix}_{timestamp}.csv")
```

- [ ] **Step 2: Smoke-test it**

Run: `cd seq-attention && python -c "from examples.results_logger import write_csv, default_output_path; p = default_output_path('smoke'); write_csv([{'a': 1, 'b': 2}], p); print(p)"`
Expected: prints a path under `examples/results/`, and that CSV file exists with header `a,b` and row `1,2`. Delete the smoke-test file afterward: `rm seq-attention/examples/results/smoke_*.csv`

- [ ] **Step 3: Commit**

```bash
git add seq-attention/examples/results_logger.py
git commit -m "seq-attention: add CSV results logger"
```

---

### Task 10: Benchmark reproduction script (`examples/run_benchmark.py`)

**Files:**
- Create: `seq-attention/examples/run_benchmark.py`

**Interfaces:**
- Consumes: `AttentionGatedMLP` (Task 3), `select_features_onepass` (Task 5), `load_mnist`/`load_fashion_mnist`/`load_isolet` (Task 8), `write_csv`/`default_output_path` (Task 9).
- Produces: a runnable CLI script; no return value consumed by later tasks (Task 11's real run is the consumer, via its CSV output and printed accuracies).

- [ ] **Step 1: Implement the script**

```python
"""Reproduces Table 2 of the Sequential Attention paper: baseline (all
features) vs Sequential Attention-selected top-k features, on MNIST,
Fashion-MNIST, and ISOLET, with a small MLP. Logs results to CSV.

Run: python examples/run_benchmark.py [--dataset mnist|fashion_mnist|isolet|all] [--output PATH]
"""

import argparse

import torch

from seqattention.models import AttentionGatedMLP
from seqattention.onepass import select_features_onepass

from data import load_fashion_mnist, load_isolet, load_mnist
from results_logger import default_output_path, write_csv

# (loader, num_features, num_classes, k, hidden_dim)
DATASETS = {
    "mnist": (load_mnist, 784, 10, 50, 256),
    "fashion_mnist": (load_fashion_mnist, 784, 10, 50, 256),
    "isolet": (load_isolet, 617, 26, 50, 256),
}


def train_classifier(model, X, y, steps=2000, lr=1e-3):
    optimizer = torch.optim.Adam(model.body.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def evaluate(model, X, y) -> float:
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def pin_selected_features(model, selected: list[int]):
    with torch.no_grad():
        model.mask.overparam_weight.zero_()
        for idx in selected:
            model.mask.select(idx)
            model.mask.overparam_weight[idx] = 1.0
    model.mask.attention_logits.requires_grad_(False)
    model.mask.overparam_weight.requires_grad_(False)
    return model


def run_dataset(name, loader, num_features, num_classes, k, hidden_dim, seed=0):
    X_train, y_train = loader(train=True)
    X_test, y_test = loader(train=False)

    torch.manual_seed(seed)
    baseline = AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed)
    baseline = train_classifier(baseline, X_train, y_train)
    baseline_acc = evaluate(baseline, X_test, y_test)

    y_train_float = y_train.float()
    selector_model_fn = lambda seed_: AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed_)
    selected = select_features_onepass(
        model_factory=selector_model_fn,
        loss_fn=lambda y_pred, y_true: torch.nn.functional.cross_entropy(y_pred, y_true.long()),
        X=X_train, y=y_train, k=k, train_steps_per_phase=200, lr=1e-3, seed=seed,
    )

    torch.manual_seed(seed)
    selected_model = AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed)
    selected_model = pin_selected_features(selected_model, selected)
    selected_model = train_classifier(selected_model, X_train, y_train)
    selected_acc = evaluate(selected_model, X_test, y_test)

    return {
        "dataset": name,
        "k": k,
        "baseline_accuracy": round(baseline_acc, 4),
        "selected_accuracy": round(selected_acc, 4),
        "selected_features": ";".join(str(i) for i in selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    rows = []
    for name in names:
        loader, num_features, num_classes, k, hidden_dim = DATASETS[name]
        row = run_dataset(name, loader, num_features, num_classes, k, hidden_dim)
        print(row)
        rows.append(row)

    output = args.output or default_output_path("run_benchmark")
    write_csv(rows, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on the fastest dataset with a tiny step count**

Run: `cd seq-attention/examples && python -c "
import run_benchmark as rb
from data import load_mnist
row = rb.run_dataset('mnist', load_mnist, 784, 10, k=5, hidden_dim=32)
print(row)
"`
Expected: prints a dict with `dataset='mnist'`, `k=5`, and both accuracy fields as floats between 0 and 1 (values will be low since this smoke test uses far fewer training steps than the real run — that's expected and fine here).

- [ ] **Step 3: Commit**

```bash
git add seq-attention/examples/run_benchmark.py
git commit -m "seq-attention: add Table 2 benchmark reproduction script"
```

---

### Task 11: `seq-attention/README.md`

**Files:**
- Create: `seq-attention/README.md`
- Modify: `README.md:33` (root README's "Sequential Attention" section)

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Write `seq-attention/README.md`**

```markdown
# Sequential Attention

A paper-accurate PyTorch implementation of Google's **Sequential Attention**
feature-selection algorithm, based on *"Sequential Attention for Feature
Selection"* (Yasuda, Bateni, Chen, Fahrbach, Fu, Mirrokni, ICLR 2023,
[arXiv:2209.14881](https://arxiv.org/abs/2209.14881)).

This is a greedy input-feature-selection method built on a softmax attention
mask over candidate features — not a transformer/LLM attention mechanism.
See [`sequential-attention.html`](sequential-attention.html) for an
interactive walkthrough of the math.

## What is implemented

- The softmax attention mask + Hadamard-product overparameterization
  (`seqattention/mask.py`)
- Algorithm 1: greedy sequential feature selection, both the naive
  per-phase-retrain reading (`seqattention/selector.py`) and the paper's
  more efficient one-pass training trick (`seqattention/onepass.py`)
- Reference Orthogonal Matching Pursuit and Sequential LASSO
  implementations (`seqattention/omp.py`), used to numerically demonstrate
  their proven equivalence to Regularized Linear Sequential Attention
  (Theorem 1.1/3.3) — see `examples/run_omp_equivalence.py`
- A benchmark reproduction of the paper's Table 2 results on MNIST,
  Fashion-MNIST, and ISOLET with a small MLP (`examples/run_benchmark.py`)

## Results

<!-- Filled in by Task 12 after a real run on the project's RTX 4070. -->

## File structure

```
seqattention/
├── mask.py       # softmax attention mask + Hadamard overparameterization
├── selector.py   # Algorithm 1, naive per-phase retrain
├── onepass.py    # Algorithm 1, one-pass training trick
├── omp.py        # OMP + Sequential LASSO reference implementations
└── models.py     # mask-gated linear regression and MLP models
examples/
├── run_omp_equivalence.py   # OMP/LASSO/Sequential Attention equivalence demo
├── run_benchmark.py         # Table 2 reproduction (MNIST/Fashion-MNIST/ISOLET)
├── data.py                  # dataset loading, incl. ISOLET fetch/cache
└── results_logger.py        # CSV result logging
```

## Installation

From the repo root:
```bash
uv sync
```

## Usage

```bash
cd seq-attention
python examples/run_omp_equivalence.py
python examples/run_benchmark.py --dataset all
python -m pytest tests/ -v
```
```

- [ ] **Step 2: Update the root README's Sequential Attention section**

In `README.md`, replace:
```markdown
### Sequential Attention

**Paper**: [Sequential Attention: Making AI models leaner and faster without sacrificing accuracy](https://research.google/blog/sequential-attention-making-ai-models-leaner-and-faster-without-sacrificing-accuracy/)

*Implementation pending.*
```
with:
```markdown
### Sequential Attention -- Feature Selection ([`seq-attention/`](seq-attention/))

**Paper**: [Sequential Attention for Feature Selection](https://arxiv.org/abs/2209.14881) (ICLR 2023, Yasuda et al.)

Paper-accurate PyTorch implementation of Algorithm 1 (greedy sequential
selection, naive and one-pass variants), the softmax attention mask with
Hadamard-product overparameterization, and a numerical demonstration of the
paper's proven OMP/Sequential-LASSO equivalence. Benchmarked against the
paper's Table 2 on MNIST, Fashion-MNIST, and ISOLET.

See [`seq-attention/README.md`](seq-attention/README.md) for results and
how to reproduce them.
```

- [ ] **Step 3: Commit**

```bash
git add seq-attention/README.md README.md
git commit -m "seq-attention: add README and update root README status"
```

---

### Task 12: Real benchmark run and results documentation

**Files:**
- Modify: `seq-attention/README.md` (Results section)
- Create: `seq-attention/examples/results/run_benchmark_<timestamp>.csv` (generated, not hand-written)

**Interfaces:**
- None (this task's deliverable is validated data, not new code).

- [ ] **Step 1: Run the full benchmark on the project's RTX 4070 (or CPU if unavailable)**

Run: `cd seq-attention && python examples/run_benchmark.py --dataset all`
Expected: completes without error (MNIST/Fashion-MNIST download automatically via `torchvision`; ISOLET downloads once and caches under `examples/data_cache/`), prints one result dict per dataset, and writes a timestamped CSV under `examples/results/`.

- [ ] **Step 2: Compare against the paper's Table 2 targets**

Check the printed `baseline_accuracy`/`selected_accuracy` against:

| Dataset | Baseline | Sequential Attention |
|---|---|---|
| MNIST | 0.944 | 0.956 |
| Fashion-MNIST | 0.843 | 0.854 |
| ISOLET | 0.866 | 0.920 |

If any result is far off (not just normal run-to-run noise — e.g., `selected_accuracy` below `baseline_accuracy`, which would indicate a bug rather than variance), treat it as a bug: re-check `pin_selected_features`, the mask's `overparam_weight` zeroing, and `select_features_onepass`'s loss function before re-running. Do not adjust the target numbers to match a bad run.

- [ ] **Step 3: Record the real results in `seq-attention/README.md`**

Replace the `## Results` placeholder with a table like:
```markdown
## Results

Run on the project's RTX 4070 Laptop GPU, MLP with `hidden_dim=256`,
`k=50` selected features per dataset.

| Dataset | Baseline (all features) | Sequential Attention (k=50) | Paper (Table 2) |
|---|---|---|---|
| MNIST | <measured> | <measured> | 0.944 -> 0.956 |
| Fashion-MNIST | <measured> | <measured> | 0.843 -> 0.854 |
| ISOLET | <measured> | <measured> | 0.866 -> 0.920 |

Reproduce with: `python examples/run_benchmark.py --dataset all`. Full CSV:
[`examples/results/run_benchmark_<timestamp>.csv`](examples/results/).
```
(Fill in `<measured>` with the actual numbers from Step 1's run, and the real timestamp/filename.)

- [ ] **Step 4: Commit**

```bash
git add seq-attention/README.md seq-attention/examples/results/
git commit -m "seq-attention: record real benchmark results vs paper Table 2"
```
