"""Diagnostic experiment, NOT the paper-accurate benchmark (see run_benchmark.py
for that). The early-stopping experiment (run_benchmark_early_stopping.py)
falsified the "baseline over-trains" hypothesis. Next candidate: the
baseline's first layer has num_features * hidden_dim parameters, vastly more
than the selected model's k * hidden_dim -- e.g. on MNIST, 784*256 vs
50*256, a 15.7x difference. Tests whether the baseline's advantage is simply
having far more first-layer capacity per input than the selected model,
independent of which/how many features it sees, by shrinking the baseline's
hidden_dim so its first layer has (approximately) the SAME parameter count
as the selected model's first layer: hidden_dim_baseline = round(k *
hidden_dim_selected / num_features). Everything else (training budget,
optimizer, lr) stays identical and symmetric between the two models, as in
run_benchmark.py.

Run: python examples/run_benchmark_capacity_matched.py [--dataset mnist|fashion_mnist|isolet|all] [--output PATH]
"""

import argparse

import torch

from seqattention.models import AttentionGatedMLP
from seqattention.onepass import select_features_onepass

from data import load_fashion_mnist, load_isolet, load_mnist
from results_logger import default_output_path, write_csv
from run_benchmark import DATASETS, DEVICE, evaluate, pin_selected_features, train_classifier


def capacity_matched_hidden_dim(num_features, k, hidden_dim_selected):
    return max(1, round(k * hidden_dim_selected / num_features))


def run_dataset(name, loader, num_features, num_classes, k, hidden_dim, seed=0):
    X_train, y_train = loader(train=True)
    X_test, y_test = loader(train=False)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    hidden_dim_baseline = capacity_matched_hidden_dim(num_features, k, hidden_dim)

    torch.manual_seed(seed)
    baseline = AttentionGatedMLP(num_features, hidden_dim_baseline, num_classes, seed=seed).to(DEVICE)
    baseline = pin_selected_features(baseline)
    baseline = train_classifier(baseline, X_train, y_train)
    baseline_acc = evaluate(baseline, X_test, y_test)

    selector_model_fn = lambda seed_: AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed_).to(DEVICE)
    selected = select_features_onepass(
        model_factory=selector_model_fn,
        loss_fn=lambda y_pred, y_true: torch.nn.functional.cross_entropy(y_pred, y_true.long()),
        X=X_train, y=y_train, k=k, train_steps_per_phase=200, lr=1e-3, seed=seed,
    )

    selected_idx = torch.tensor(selected, dtype=torch.long, device=DEVICE)
    X_train_selected = X_train[:, selected_idx]
    X_test_selected = X_test[:, selected_idx]

    torch.manual_seed(seed)
    selected_model = AttentionGatedMLP(len(selected), hidden_dim, num_classes, seed=seed).to(DEVICE)
    selected_model = pin_selected_features(selected_model)
    selected_model = train_classifier(selected_model, X_train_selected, y_train)
    selected_acc = evaluate(selected_model, X_test_selected, y_test)

    return {
        "dataset": name,
        "k": k,
        "baseline_hidden_dim": hidden_dim_baseline,
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

    output = args.output or default_output_path("run_benchmark_capacity_matched_experiment")
    write_csv(rows, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
