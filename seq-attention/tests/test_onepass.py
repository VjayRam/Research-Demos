import torch

from seqattention.models import LinearRegressionModel
from seqattention.onepass import select_features_onepass
from seqattention.selector import select_features_naive


def _make_sparse_regression_problem(seed=0, n=200, d=10, true_idx=(1, 4, 6), base=3.0):
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=generator)
    true_coef = torch.zeros(d)
    for i, idx in enumerate(true_idx):
        true_coef[idx] = base - i
    y = X @ true_coef
    return X, y, set(true_idx)


def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def test_select_features_onepass_recovers_ground_truth_support():
    X, y, true_support = _make_sparse_regression_problem()
    selected = select_features_onepass(
        model_factory=lambda seed: LinearRegressionModel(num_features=X.shape[1], seed=seed),
        loss_fn=mse_loss,
        X=X, y=y, k=3, train_steps_per_phase=300, lr=0.01, seed=0,
    )
    assert set(selected) == true_support


def test_select_features_onepass_matches_naive_baseline_selected_set():
    X, y, _ = _make_sparse_regression_problem(true_idx=(0, 3, 5, 7), base=6.0)
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
