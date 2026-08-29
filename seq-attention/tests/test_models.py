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
