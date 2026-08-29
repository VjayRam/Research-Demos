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
