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


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)


def build_problem() -> tuple[torch.Tensor, torch.Tensor]:
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
