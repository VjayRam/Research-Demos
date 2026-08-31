"""Runs RDKV's Phase 1 allocation pipeline against a real HF model's
prefill attention, reporting per-layer eviction/bit-width statistics.

This is a demonstration and sanity-check script, not a production
compressed-cache integration (Phase 2 -- TriZone packing and the fused
kernel -- would be needed for that). Requires the `examples` extra:

    pip install -e ".[examples]"

Usage:
    python examples/allocation_stats.py --model sshleifer/tiny-gpt2 --b-tok 4.0
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rdkv import RDKVAllocator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--b-tok", type=float, default=4.0, help="per-head budget in FP16-equivalent tokens")
    parser.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog. " * 8,
        help="prefill text to build the attention/Q/K statistics from",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager", output_attentions=True)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (batch, n_heads, T, T) per layer
    allocator = RDKVAllocator()

    print(f"model={args.model} n_layers={len(attentions)} b_tok={args.b_tok}")
    for layer_idx, layer_attn in enumerate(attentions):
        n_heads = layer_attn.shape[1]
        T = layer_attn.shape[-1]
        d = model.config.hidden_size // model.config.n_head if hasattr(model.config, "n_head") else 64

        head0_attn = layer_attn[0, 0]  # (T, T): row=query, col=key/token
        q = torch.randn(T, d)  # placeholder Q/K -- real per-head Q/K extraction
        k = torch.randn(T, d)  # requires model-specific hook, left for a follow-up script

        result = allocator.allocate(head0_attn, q, k, b_tok=args.b_tok)
        n_kept = result.kept_tokens.shape[0]
        mean_b_v = result.b_v.float().mean().item()
        mean_b_k = result.b_k.float().mean().item()
        print(
            f"  layer {layer_idx:2d}: T={T:4d} kept={n_kept:4d} "
            f"({100 * n_kept / T:5.1f}%) mean_b_v={mean_b_v:5.2f} mean_b_k={mean_b_k:5.2f}"
        )


if __name__ == "__main__":
    main()
