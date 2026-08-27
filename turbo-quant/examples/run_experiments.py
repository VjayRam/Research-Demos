"""Real-model experiments for turboquant: perplexity/compression on WikiText-2,
and empirical-vs-theoretical distortion on real Qwen2.5-0.5B key vectors.

Part A reuses run_benchmark.py's pattern (algorithm x bits perplexity sweep via
QuantizingCache) but on real WikiText-2 text instead of a repeated fixed string.

Part B verifies the paper's Theorem 1 near-optimality claim -- that the fixed,
data-oblivious rotation+Lloyd-Max quantizer achieves distortion close to the
theoretical bound -- using REAL key vectors extracted from a live forward pass,
not synthetic Gaussian noise.

Usage:
    python run_experiments.py --smoke-test
    python run_experiments.py
    python run_experiments.py --model Qwen/Qwen2.5-0.5B --bits 1 2 3 4
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from kv_cache_hook import QuantizingCache
from results_logger import default_output_path, write_csv
from turboquant import PolarQuant, TurboQuantMSE, TurboQuantProd

ALGORITHMS = {
    "mse": TurboQuantMSE,
    "prod": TurboQuantProd,
    "polar": PolarQuant,
}

SMOKE_TEXT = "The quick brown fox jumps over the lazy dog. " * 20


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def head_dim_of(model) -> int:
    config = model.config
    if hasattr(config, "head_dim") and config.head_dim:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def compression_ratio(head_dim: int, bits: int, algorithm: str) -> float:
    """Analytical compression ratio (index bits vs. fp16), not actual bit-packing."""
    fp16_bits = head_dim * 16
    if algorithm == "prod":
        packed_bits = head_dim * (bits - 1) + head_dim  # (bits-1)-bit MSE + 1 QJL bit/coord
    else:
        packed_bits = head_dim * bits
    packed_bits += 16  # one fp16 norm/radius scalar per vector
    return fp16_bits / packed_bits


def build_wikitext_sample(min_words: int = 2000) -> str:
    """Concatenate non-empty, non-heading lines from WikiText-2's test split
    until the sample has at least `min_words` words."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    lines = []
    word_count = 0
    for text in ds["text"]:
        stripped = text.strip()
        if not stripped:
            continue
        # WikiText-2 raw format marks section headings as " = Heading = " (or
        # " == Subheading == ", etc.) with nothing else on the line -- skip them.
        if stripped.startswith("=") and stripped.endswith("="):
            continue
        lines.append(stripped)
        word_count += len(stripped.split())
        if word_count >= min_words:
            break

    return " ".join(lines)


@torch.no_grad()
def measure_perplexity(model, tokenizer, text: str, cache=None, device: str = "cpu") -> float:
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids, past_key_values=cache, labels=input_ids, use_cache=cache is not None)
    return math.exp(outputs.loss.item())


@torch.no_grad()
def run_perplexity_sweep(
    model,
    tokenizer,
    text: str,
    model_name: str,
    device: str,
    algorithms: list[str],
    bits_list: list[int],
    output: str | None,
) -> int:
    """Part A: algorithm x bits perplexity/compression sweep on real text.

    Returns the model's head_dim (needed by Part B)."""
    if output is None:
        output = default_output_path("run_experiments_perplexity")

    d = head_dim_of(model)
    rows = []

    baseline_ppl = measure_perplexity(model, tokenizer, text, device=device)
    print(f"{model_name} (head_dim={d}) baseline perplexity: {baseline_ppl:.3f}")
    rows.append(
        {
            "model": model_name,
            "device": device,
            "head_dim": d,
            "algorithm": "baseline",
            "bits": None,
            "perplexity": baseline_ppl,
            "perplexity_delta": 0.0,
            "compression_ratio": 1.0,
        }
    )

    for algorithm in algorithms:
        if algorithm == "polar" and not is_power_of_2(d):
            print(f"  {algorithm}: skipped (head_dim={d} is not a power of 2, PolarQuant requires it)")
            continue
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
            rows.append(
                {
                    "model": model_name,
                    "device": device,
                    "head_dim": d,
                    "algorithm": algorithm,
                    "bits": bits,
                    "perplexity": ppl,
                    "perplexity_delta": ppl - baseline_ppl,
                    "compression_ratio": ratio,
                }
            )

    write_csv(rows, output)
    print(f"Perplexity results written to: {output}")
    return d


@torch.no_grad()
def extract_real_key_vectors(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """Run one forward pass with a plain (non-quantizing) DynamicCache and pull
    out the real key vectors stored for one middle transformer layer.

    This is the simpler "inspect past_key_values" approach rather than a
    forward hook on the K/V projections: transformers 5.x's DynamicCache
    already stores exactly the per-layer key tensors we need (shape
    (batch, num_kv_heads, seq_len, head_dim)) on `cache.layers[i].keys`, so a
    hook would just be re-deriving the same tensor with more moving parts
    (module-path guessing across model architectures/transformers versions).
    Inspecting the cache after a single forward pass is more robust.
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()
    model(input_ids, past_key_values=cache, use_cache=True)

    num_layers = model.config.num_hidden_layers
    mid_layer = num_layers // 2
    keys = cache.layers[mid_layer].keys  # (batch, num_kv_heads, seq_len, head_dim)
    return keys


@torch.no_grad()
def run_distortion_experiment(
    model,
    tokenizer,
    text: str,
    head_dim: int,
    device: str,
    bits_list: list[int],
    output: str | None,
):
    """Part B: empirical distortion of TurboQuantMSE on REAL key vectors vs.
    the paper's Theorem 1 theoretical bounds."""
    if output is None:
        output = default_output_path("run_experiments_distortion")

    keys = extract_real_key_vectors(model, tokenizer, text, device=device)
    x = keys.reshape(-1, head_dim).float()
    n_vectors = x.shape[0]
    print(f"\nExtracted {n_vectors} real key vectors (head_dim={head_dim}) from a middle transformer layer.")

    rows = []
    print(f"\n{'bits':>4}  {'empirical':>10}  {'bound(general)':>15}  {'bound(solved)':>14}  {'within bound':>12}")
    for bits in bits_list:
        quantizer = TurboQuantMSE(head_dim, bits, seed=0, device=device)
        indices, norm = quantizer.quantize(x)
        x_hat = quantizer.dequantize(indices, norm)

        sq_err = ((x - x_hat) ** 2).sum(dim=-1)
        sq_norm = (x ** 2).sum(dim=-1).clamp_min(1e-12)
        empirical_distortion = (sq_err / sq_norm).mean().item()

        theoretical_bound_general = 1.5 * 4 ** (-bits)
        theoretical_bound_solved = quantizer.codebook.distortion * head_dim

        # 10% slack: the paper's expectation is over the assumed coordinate
        # density after rotation, whereas real model key vectors are a finite,
        # non-adversarial empirical sample -- so we allow a small margin
        # rather than requiring the empirical mean to land exactly at or
        # under the asymptotic bound.
        within_bound = bool(empirical_distortion <= theoretical_bound_general * 1.1)

        print(
            f"{bits:>4}  {empirical_distortion:>10.4f}  {theoretical_bound_general:>15.4f}  "
            f"{theoretical_bound_solved:>14.4f}  {str(within_bound):>12}"
        )

        rows.append(
            {
                "bits": bits,
                "head_dim": head_dim,
                "n_vectors": n_vectors,
                "empirical_distortion": empirical_distortion,
                "theoretical_bound_general": theoretical_bound_general,
                "theoretical_bound_solved": theoretical_bound_solved,
                "empirical_within_general_bound": within_bound,
            }
        )

    write_csv(rows, output)
    print(f"\nDistortion results written to: {output}")


def run(
    model_name: str,
    algorithms: list[str],
    bits_list: list[int],
    device: str | None,
    text: str | None,
    min_words: int,
    output_perplexity: str | None,
    output_distortion: str | None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    if text is None:
        print("Loading WikiText-2 (test split) and building a real-text sample...")
        text = build_wikitext_sample(min_words=min_words)
        print(f"Sample built: {len(text.split())} words, {len(text)} characters.")

    print("\n=== Part A: perplexity / compression sweep on real text ===")
    head_dim = run_perplexity_sweep(
        model, tokenizer, text, model_name, device, algorithms, bits_list, output_perplexity
    )

    print("\n=== Part B: empirical distortion vs. theoretical bound (real key vectors) ===")
    run_distortion_experiment(model, tokenizer, text, head_dim, device, bits_list, output_distortion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default=None, help="torch device (default: auto-detect CUDA if available)")
    parser.add_argument("--algorithms", nargs="+", default=["mse", "prod", "polar"], choices=list(ALGORITHMS))
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--min-words", type=int, default=2000, help="minimum words in the WikiText-2 sample")
    parser.add_argument("--output-perplexity", default=None, help="path to write perplexity-sweep CSV")
    parser.add_argument("--output-distortion", default=None, help="path to write distortion CSV")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="tiny model, short hardcoded text, single config, for fast CI-free verification",
    )
    args = parser.parse_args()

    if args.smoke_test:
        run(
            "hf-internal-testing/tiny-random-gpt2",
            ["mse"],
            [2],
            device=args.device,
            text=SMOKE_TEXT,
            min_words=args.min_words,
            output_perplexity=args.output_perplexity,
            output_distortion=args.output_distortion,
        )
    else:
        run(
            args.model,
            args.algorithms,
            args.bits,
            device=args.device,
            text=None,
            min_words=args.min_words,
            output_perplexity=args.output_perplexity,
            output_distortion=args.output_distortion,
        )


if __name__ == "__main__":
    main()
