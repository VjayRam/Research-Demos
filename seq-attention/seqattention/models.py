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
