import torch

from seqattention.mask import SequentialAttentionMask


def test_unselected_softmax_sums_to_one():
    mask = SequentialAttentionMask(num_features=5, seed=0)
    m = mask.softmax_mask()
    assert not torch.isclose(m.sum(), torch.tensor(5.0), atol=1e-4)  # not all weight-1 yet
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
