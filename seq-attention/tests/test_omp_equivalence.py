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
