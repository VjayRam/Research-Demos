"""End-to-end RDKV demo: allocate bit-widths on a real model's prefill
K/V, pack into TriZone storage, and run one packed decode step.

Requires the `examples` extra (and, for the kernel-backend comparison,
the `kernel` extra on a CUDA machine):

    pip install -e ".[examples]"

Usage:
    python examples/packed_decode_demo.py --model sshleifer/tiny-gpt2 --b-tok 4.0
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rdkv import RDKVAllocator
from rdkv.trizone import pack_trizone
from rdkv.decode import packed_decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--b-tok", type=float, default=4.0)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog. " * 8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager", output_attentions=True)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_attn = outputs.attentions[0][0, 0].to(device)  # first layer, first head: (T, T)
    T = layer_attn.shape[-1]
    d = model.config.hidden_size // model.config.n_head if hasattr(model.config, "n_head") else 64

    q = torch.randn(T, d, device=device)  # see Task 6's note: real per-head Q/K extraction is a follow-up
    k = torch.randn(T, d, device=device)
    v = torch.randn(T, d, device=device)

    allocator = RDKVAllocator()
    allocation = allocator.allocate(layer_attn, q, k, b_tok=args.b_tok)
    packed = pack_trizone(k, v, allocation)

    n_kept = allocation.kept_tokens.shape[0]
    fp16_bits = T * d * 16 * 2  # K+V, full precision baseline
    packed_bits = sum(seg.numel() * bits for bits, seg in packed.zone_a_v.items())
    packed_bits += packed.zone_a_k.numel() * (packed.zone_a_k.element_size() * 8)  # conservative, pre-bitpack size
    packed_bits += packed.zone_b_v.numel() * 16
    print(f"model={args.model} T={T} d={d} kept={n_kept}/{T} ({100 * n_kept / T:.1f}%)")
    print(f"approx compression vs FP16: {fp16_bits / max(packed_bits, 1):.2f}x (pre-bitpacking element counts)")
    print(
        "  (illustrative -- assumes Zone A(V) sub-segments are bit-packed at their "
        "target bit-width; they are not yet, see rdkv/README.md's Phase 2 disclosed "
        "gap. Real storage today is Zone A(V) at full float32 precision, so this "
        "number is aspirational, not a measurement of current on-disk/in-memory size.)"
    )

    q_tau = torch.randn(d, device=device)
    k_new = torch.randn(1, d, device=device)
    v_new = torch.randn(1, d, device=device)
    sqrt_d = math.sqrt(d)

    native_out = packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="native")
    print(f"native packed-decode output: shape={tuple(native_out.shape)}")

    if device == "cuda":
        try:
            kernel_out = packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="kernel")
            max_diff = (native_out - kernel_out).abs().max().item()
            print(f"kernel packed-decode max abs diff vs native: {max_diff:.6f}")
        except RuntimeError as exc:
            print(f"kernel backend unavailable: {exc}")


if __name__ == "__main__":
    main()
