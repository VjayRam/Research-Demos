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
