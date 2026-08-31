"""Sweep RDKV's Phase 1 allocator's `b_tok` budget on a real HF model's
prefill attention, showing how eviction rate, mean bit-widths, and
approximate compression ratio trade off against the budget.

This is the "what does the allocator actually do as you turn the knob"
companion to `allocation_stats.py` (which only prints one `--b-tok` value).
Requires the `examples` extra:

    pip install -e ".[examples]"

Usage:
    python run_allocation_sweep.py --smoke-test
    python run_allocation_sweep.py --model sshleifer/tiny-gpt2 \\
        --b-tok-values 0.25 0.5 1 2 4 8 16 --layer 0
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from results_logger import default_output_path, write_csv

from rdkv import RDKVAllocator


def _head_dim(model) -> int:
    cfg = model.config
    if hasattr(cfg, "n_head") and cfg.n_head:
        return cfg.hidden_size // cfg.n_head
    if hasattr(cfg, "num_attention_heads") and cfg.num_attention_heads:
        return cfg.hidden_size // cfg.num_attention_heads
    return 64


def sweep(
    model_name: str,
    text: str,
    layer: int,
    b_tok_values: list[float],
    seed: int,
    output: str | None,
) -> None:
    if output is None:
        output = default_output_path("run_allocation_sweep")

    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", output_attentions=True)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    n_layers = len(outputs.attentions)
    if layer >= n_layers:
        raise ValueError(f"--layer {layer} out of range (model has {n_layers} layers)")

    layer_attn = outputs.attentions[layer][0, 0]  # (T, T): row=query, col=key/token
    T = layer_attn.shape[-1]
    d = _head_dim(model)

    # NOTE: placeholder Q/K (real per-head extraction needs a model-specific
    # forward hook -- see allocation_stats.py's note). Fixed across the sweep
    # via the seed above, so differences between b_tok values below reflect
    # only the budget, not fresh randomness.
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()

    print(f"model={model_name} layer={layer} T={T} d={d}")
    print(f"{'b_tok':>8} {'kept':>10} {'kept_%':>7} {'mean_b_v':>9} {'mean_b_k':>9} {'compress_x':>11}")

    rows = []
    fp16_bits = T * d * 16 * 2  # K+V, full-precision baseline for this head
    for b_tok in b_tok_values:
        result = allocator.allocate(layer_attn, q, k, b_tok=b_tok)
        n_kept = result.kept_tokens.shape[0]
        mean_b_v = result.b_v.float().mean().item()
        mean_b_k = result.b_k.float().mean().item()

        # Same illustrative, pre-bitpacking accounting as packed_decode_demo.py
        # -- see that script's caveat: Zone A(V) isn't actually byte-packed
        # yet, so this is an upper bound on achievable compression, not a
        # measurement of the current storage format.
        allocated_bits = result.b_v.sum().item() * d + result.b_k.sum().item() * n_kept
        compress_x = fp16_bits / max(allocated_bits, 1)

        print(
            f"{b_tok:8.3f} {n_kept:6d}/{T:<4d}{100 * n_kept / T:6.1f}% "
            f"{mean_b_v:9.2f} {mean_b_k:9.2f} {compress_x:10.2f}x"
        )
        rows.append(
            {
                "model": model_name,
                "layer": layer,
                "T": T,
                "d": d,
                "b_tok": b_tok,
                "kept_tokens": n_kept,
                "kept_pct": 100 * n_kept / T,
                "mean_b_v": mean_b_v,
                "mean_b_k": mean_b_k,
                "compression_x_illustrative": compress_x,
            }
        )

    write_csv(rows, output)
    print(f"\nResults written to: {output}")
    print(
        "Note: compress_x is illustrative (assumes Zone A(V)/Zone A(K) are "
        "bit-packed at their target widths, which isn't implemented yet -- "
        "see rdkv/README.md's Disclosed gap)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog. " * 8,
        help="prefill text to build the attention/Q/K statistics from",
    )
    parser.add_argument("--layer", type=int, default=0, help="which layer's attention to allocate over")
    parser.add_argument(
        "--b-tok-values",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        help="per-head budgets (in FP16-equivalent tokens) to sweep",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the placeholder Q/K tensors")
    parser.add_argument(
        "--output",
        default=None,
        help="path to write CSV results (default: timestamped file under examples/results/)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="tiny sweep, for quick CI-free verification")
    args = parser.parse_args()

    if args.smoke_test:
        sweep(args.model, args.text, args.layer, [0.5, 4.0], seed=args.seed, output=args.output)
    else:
        sweep(args.model, args.text, args.layer, args.b_tok_values, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
