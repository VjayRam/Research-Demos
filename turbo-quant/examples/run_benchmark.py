"""Benchmark turboquant algorithms against real LLMs' KV caches.

Usage:
    python run_benchmark.py --smoke-test
    python run_benchmark.py --model Qwen/Qwen2.5-0.5B --algorithm mse prod --bits 1 2 3 4
    python run_benchmark.py --model google/gemma-2-2b --algorithm mse polar --bits 2 4
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache_hook import QuantizingCache
from turboquant import PolarQuant, TurboQuantMSE, TurboQuantProd

ALGORITHMS = {
    "mse": TurboQuantMSE,
    "prod": TurboQuantProd,
    "polar": PolarQuant,
}

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. " * 20


def compression_ratio(head_dim: int, bits: int, algorithm: str) -> float:
    """Analytical compression ratio (index bits vs. fp16), not actual bit-packing."""
    fp16_bits = head_dim * 16
    if algorithm == "prod":
        packed_bits = head_dim * (bits - 1) + head_dim  # (bits-1)-bit MSE + 1 QJL bit/coord
    else:
        packed_bits = head_dim * bits
    packed_bits += 16  # one fp16 norm/radius scalar per vector
    return fp16_bits / packed_bits


@torch.no_grad()
def measure_perplexity(model, tokenizer, text: str, cache=None, device: str = "cpu") -> float:
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids, past_key_values=cache, labels=input_ids, use_cache=cache is not None)
    return math.exp(outputs.loss.item())


def head_dim_of(model) -> int:
    config = model.config
    if hasattr(config, "head_dim") and config.head_dim:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def run(model_name: str, algorithms: list[str], bits_list: list[int], repeat: int, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    text = SAMPLE_TEXT * repeat
    d = head_dim_of(model)

    baseline_ppl = measure_perplexity(model, tokenizer, text, device=device)
    print(f"{model_name} (head_dim={d}) baseline perplexity: {baseline_ppl:.3f}")

    for algorithm in algorithms:
        cls = ALGORITHMS[algorithm]
        for bits in bits_list:
            if algorithm == "prod" and bits < 2:
                print(f"  {algorithm} b={bits}: skipped (prod requires bits >= 2)")
                continue
            key_q = cls(d, bits, seed=1, device=device)
            val_q = cls(d, bits, seed=2, device=device)
            cache = QuantizingCache(key_quantizer=key_q, value_quantizer=val_q)
            ppl = measure_perplexity(model, tokenizer, text, cache=cache, device=device)
            ratio = compression_ratio(d, bits, algorithm)
            print(
                f"  {algorithm} b={bits}: perplexity={ppl:.3f} "
                f"(+{ppl - baseline_ppl:+.3f} vs baseline), compression={ratio:.2f}x"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--algorithm", nargs="+", default=["mse", "prod"], choices=list(ALGORITHMS))
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--repeat", type=int, default=10, help="repeat the sample text N times")
    parser.add_argument("--smoke-test", action="store_true", help="tiny model, one config, for CI-free verification")
    parser.add_argument(
        "--device", default=None, help="torch device to run on (default: auto-detect CUDA if available, else CPU)"
    )
    args = parser.parse_args()

    if args.smoke_test:
        run("hf-internal-testing/tiny-random-gpt2", ["mse"], [2], repeat=1, device=args.device)
    else:
        run(args.model, args.algorithm, args.bits, repeat=args.repeat, device=args.device)


if __name__ == "__main__":
    main()
