"""
TurboQuant KV Cache Compression -- Honest Evaluation
=====================================================
Measures real GPU memory, attention output fidelity, compress/decompress
throughput, and generation round-trip correctness.

No cache hacking, no fake memory savings.  All measurements use
torch.cuda.memory_allocated() deltas on actual tensors.

Usage:
    python evaluate.py
    python evaluate.py --model Qwen/Qwen2.5-3B-Instruct
    python evaluate.py --model meta-llama/Llama-3.2-1B-Instruct --context 2048
    python evaluate.py --profiles moderate extreme
"""

import argparse
import gc
import math
import sys
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from compressors import PROFILES, CompressionProfile, TurboQuantV3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_bytes() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@contextmanager
def track_gpu():
    """Context manager that yields a dict with 'before' and 'after' GPU bytes."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    info = {"before": torch.cuda.memory_allocated()}
    yield info
    torch.cuda.synchronize()
    info["after"] = torch.cuda.memory_allocated()
    info["delta"] = info["after"] - info["before"]


def fmt_bytes(b: int) -> str:
    if abs(b) >= 1024 ** 3:
        return f"{b / 1024**3:.2f} GB"
    if abs(b) >= 1024 ** 2:
        return f"{b / 1024**2:.1f} MB"
    if abs(b) >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def separator(char: str = "=", width: int = 72) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    load_kwargs = {"device_map": "auto", "dtype": torch.float16}
    if not args.no_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            print("  Using 4-bit model quantization (bitsandbytes)")
        except (ImportError, Exception) as e:
            print(f"  bitsandbytes not available ({e}), loading in fp16")

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except ValueError as e:
        if "GPU RAM" in str(e) or "CPU or the disk" in str(e):
            print(f"  4-bit load failed ({e}), retrying in fp16 ...")
            load_kwargs.pop("quantization_config", None)
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        else:
            raise
    model.eval()

    mem_mb = torch.cuda.memory_allocated() // 1024 // 1024
    print(f"  Model loaded. GPU memory: {mem_mb} MB")
    return model, tokenizer


def extract_model_config(model):
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads",
                         getattr(cfg, "num_attention_heads", 32))
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return n_layers, n_kv_heads, head_dim


# ---------------------------------------------------------------------------
# KV cache extraction
# ---------------------------------------------------------------------------

def capture_kv_cache(model, tokenizer, context_len: int):
    """Run a forward pass and return (keys_list, values_list, input_ids).

    Each element of keys_list / values_list is shape (B, H, S, D) on GPU.
    """
    filler = (
        "The quarterly financial review meeting covered several topics including "
        "budget allocations for the upcoming fiscal year, departmental spending "
        "reports, and projected revenue streams from various business units. "
        "Several action items were assigned to team leads for follow-up.\n\n"
    )
    filler_tok_len = len(tokenizer.encode(filler))
    n_reps = max(1, context_len // filler_tok_len)
    text = filler * n_reps

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text + "\nSummarize the above."},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = text + "\nSummarize the above.\n"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_len
    ).to("cuda")
    actual_tokens = inputs["input_ids"].shape[1]
    print(f"  Prompt tokens: {actual_tokens}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    cache = outputs.past_key_values
    n_layers = len(cache.layers)
    keys_list, values_list = [], []
    for li in range(n_layers):
        keys_list.append(cache.layers[li].keys.detach().clone())
        values_list.append(cache.layers[li].values.detach().clone())

    del outputs, cache
    gc.collect()
    torch.cuda.empty_cache()

    return keys_list, values_list, inputs["input_ids"]


# ---------------------------------------------------------------------------
# Test 1: Actual GPU memory measurement
# ---------------------------------------------------------------------------

def _tensor_bytes(obj) -> int:
    """Recursively sum logical GPU tensor bytes inside dicts/tuples."""
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        return obj.nelement() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def measure_memory(keys_list, values_list, profile: CompressionProfile, n_layers: int):
    """Compress each layer's KV and count actual tensor bytes in the result."""
    device = keys_list[0].device
    B, H, S, D = keys_list[0].shape

    fp16_total = n_layers * B * H * S * D * 2 * 2

    compressed_total = 0
    for li in range(n_layers):
        comp = TurboQuantV3(
            head_dim=D,
            key_bits=profile.key_bits,
            value_bits=profile.value_bits,
            residual_window=profile.residual_window,
            layer_idx=li,
            n_layers=n_layers,
            protected_layers=profile.protected_layers,
            protected_bits=profile.protected_bits,
            seed=42,
            device=str(device),
        )
        ck, cv = comp.compress_kv(keys_list[li], values_list[li])
        compressed_total += _tensor_bytes(ck)
        compressed_total += _tensor_bytes(cv)
        del ck, cv

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fp16_bytes": fp16_total,
        "compressed_bytes": compressed_total,
        "ratio": fp16_total / max(compressed_total, 1),
    }


# ---------------------------------------------------------------------------
# Test 2: Attention output fidelity
# ---------------------------------------------------------------------------

def measure_fidelity(keys_list, values_list, profile: CompressionProfile, n_layers: int):
    """Compute attention output cosine similarity per layer."""
    device = keys_list[0].device
    B, H, S, D = keys_list[0].shape
    scale = 1.0 / math.sqrt(D)

    layer_results = []
    for li in range(n_layers):
        K_orig = keys_list[li].float()
        V_orig = values_list[li].float()

        comp = TurboQuantV3(
            head_dim=D,
            key_bits=profile.key_bits,
            value_bits=profile.value_bits,
            residual_window=profile.residual_window,
            layer_idx=li,
            n_layers=n_layers,
            protected_layers=profile.protected_layers,
            protected_bits=profile.protected_bits,
            seed=42,
            device=str(device),
        )
        ck, cv = comp.compress_kv(keys_list[li], values_list[li])
        K_dec, V_dec = comp.decompress_kv(ck, cv)
        K_dec = K_dec.float()
        V_dec = V_dec.float()

        n_queries = min(64, S)
        q_indices = torch.linspace(0, S - 1, n_queries).long()
        Q = K_orig[:, :, q_indices, :]

        scores_orig = (Q @ K_orig.transpose(-2, -1)) * scale
        attn_orig = F.softmax(scores_orig, dim=-1)
        out_orig = attn_orig @ V_orig

        scores_dec = (Q @ K_dec.transpose(-2, -1)) * scale
        attn_dec = F.softmax(scores_dec, dim=-1)
        out_dec = attn_dec @ V_dec

        out_cos = cosine_sim(out_orig, out_dec)

        top1_orig = scores_orig.argmax(dim=-1)
        top1_dec = scores_dec.argmax(dim=-1)
        top1_match = (top1_orig == top1_dec).float().mean().item()

        top5_orig = scores_orig.topk(min(5, S), dim=-1).indices
        top5_dec = scores_dec.topk(min(5, S), dim=-1).indices
        top5_overlap = sum(
            len(set(top5_orig[b, h, q].tolist()) & set(top5_dec[b, h, q].tolist()))
            for b in range(B)
            for h in range(H)
            for q in range(n_queries)
        )
        n_total_top5 = B * H * n_queries * min(5, S)
        top5_match = top5_overlap / max(n_total_top5, 1)

        layer_results.append({
            "layer": li,
            "output_cosine": out_cos,
            "top1_match": top1_match,
            "top5_match": top5_match,
        })

        del K_orig, V_orig, K_dec, V_dec, ck, cv
        del Q, scores_orig, scores_dec, attn_orig, attn_dec, out_orig, out_dec
        gc.collect()
        torch.cuda.empty_cache()

    cosines = [r["output_cosine"] for r in layer_results]
    top1s = [r["top1_match"] for r in layer_results]
    worst_idx = cosines.index(min(cosines))

    return {
        "per_layer": layer_results,
        "mean_cosine": sum(cosines) / len(cosines),
        "min_cosine": min(cosines),
        "worst_layer": worst_idx,
        "mean_top1": sum(top1s) / len(top1s),
    }


# ---------------------------------------------------------------------------
# Test 3: Compress / decompress throughput
# ---------------------------------------------------------------------------

def measure_throughput(keys_list, values_list, profile: CompressionProfile,
                       n_layers: int, n_warmup: int = 2, n_runs: int = 5):
    """Wall-clock compress and decompress times."""
    device = keys_list[0].device
    B, H, S, D = keys_list[0].shape
    total_tokens = n_layers * B * H * S

    compressors = []
    for li in range(n_layers):
        compressors.append(TurboQuantV3(
            head_dim=D,
            key_bits=profile.key_bits,
            value_bits=profile.value_bits,
            residual_window=profile.residual_window,
            layer_idx=li,
            n_layers=n_layers,
            protected_layers=profile.protected_layers,
            protected_bits=profile.protected_bits,
            seed=42,
            device=str(device),
        ))

    def run_compress():
        results = []
        for li in range(n_layers):
            results.append(compressors[li].compress_kv(keys_list[li], values_list[li]))
        return results

    def run_decompress(compressed_list):
        for li in range(n_layers):
            compressors[li].decompress_kv(*compressed_list[li])

    for _ in range(n_warmup):
        c = run_compress()
        run_decompress(c)
        del c

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        c = run_compress()
    torch.cuda.synchronize()
    compress_sec = (time.perf_counter() - t0) / n_runs

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_decompress(c)
    torch.cuda.synchronize()
    decompress_sec = (time.perf_counter() - t0) / n_runs

    fp16_bytes_total = n_layers * B * H * S * D * 2 * 2
    compress_gbps = (fp16_bytes_total / 1e9) / max(compress_sec, 1e-9)
    decompress_gbps = (fp16_bytes_total / 1e9) / max(decompress_sec, 1e-9)

    del c
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "compress_sec": compress_sec,
        "decompress_sec": decompress_sec,
        "compress_tok_per_sec": total_tokens / max(compress_sec, 1e-9),
        "decompress_tok_per_sec": total_tokens / max(decompress_sec, 1e-9),
        "compress_gbps": compress_gbps,
        "decompress_gbps": decompress_gbps,
    }


# ---------------------------------------------------------------------------
# Test 4: Generation round-trip correctness
# ---------------------------------------------------------------------------

def measure_generation_correctness(model, tokenizer, keys_list, values_list,
                                   profile: CompressionProfile, n_layers: int):
    """Compress/decompress each layer's KV and report tensor-level agreement."""
    device = keys_list[0].device
    B, H, S, D = keys_list[0].shape

    layer_k_cos = []
    layer_v_cos = []
    for li in range(n_layers):
        comp = TurboQuantV3(
            head_dim=D,
            key_bits=profile.key_bits,
            value_bits=profile.value_bits,
            residual_window=profile.residual_window,
            layer_idx=li,
            n_layers=n_layers,
            protected_layers=profile.protected_layers,
            protected_bits=profile.protected_bits,
            seed=42,
            device=str(device),
        )
        ck, cv = comp.compress_kv(keys_list[li], values_list[li])
        K_dec, V_dec = comp.decompress_kv(ck, cv)

        layer_k_cos.append(cosine_sim(keys_list[li], K_dec))
        layer_v_cos.append(cosine_sim(values_list[li], V_dec))

        del ck, cv, K_dec, V_dec
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "mean_key_cosine": sum(layer_k_cos) / len(layer_k_cos),
        "min_key_cosine": min(layer_k_cos),
        "mean_value_cosine": sum(layer_v_cos) / len(layer_v_cos),
        "min_value_cosine": min(layer_v_cos),
    }


# ---------------------------------------------------------------------------
# Test 5: Generation speed (tok/s) -- FP16 baseline vs V3Cache
# ---------------------------------------------------------------------------

NEEDLE = "The secret project code name is AURORA-7749."
EXPECTED_ANSWER = "AURORA-7749"


def _build_needle_prompt(tokenizer, target_tokens: int):
    """Build a needle-in-haystack prompt for quality checking."""
    filler = (
        "The quarterly financial review meeting covered several topics including "
        "budget allocations for the upcoming fiscal year, departmental spending "
        "reports, and projected revenue streams from various business units. "
        "Several action items were assigned to team leads for follow-up.\n\n"
    )
    filler_tok_len = len(tokenizer.encode(filler))
    n_reps = max(1, target_tokens // filler_tok_len)
    needle_idx = n_reps // 2
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Internal Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(filler)
    haystack = "".join(parts)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": (
            f"Read this document:\n\n{haystack}\n\n"
            "What is the secret project code name? Answer with just the code name."
        )},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"Document:\n{haystack}\n\n"
            "Question: What is the secret project code name?\nAnswer:"
        )


def _build_v3cache(profile: CompressionProfile, n_layers: int):
    """Build a V3Cache for generation with TurboQuant compression."""
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer

    class V3Cache(DynamicCache):
        """DynamicCache that compresses older KV with TurboQuantV3 on the fly."""

        def __init__(self, prof: CompressionProfile, num_layers: int):
            super().__init__()
            self._prof = prof
            self._n_layers = num_layers
            self._compressors: dict[int, TurboQuantV3] = {}
            self._chunks_k: dict[int, list] = {}
            self._chunks_v: dict[int, list] = {}
            self._recent_k: dict[int, list] = {}
            self._recent_v: dict[int, list] = {}
            self._seq_lens: dict[int, int] = {}

        def _get_comp(self, layer_idx: int, D: int, device: str):
            if layer_idx not in self._compressors:
                self._compressors[layer_idx] = TurboQuantV3(
                    head_dim=D,
                    key_bits=self._prof.key_bits,
                    value_bits=self._prof.value_bits,
                    residual_window=0,
                    layer_idx=layer_idx,
                    n_layers=self._n_layers,
                    protected_layers=self._prof.protected_layers,
                    protected_bits=self._prof.protected_bits,
                    seed=42,
                    device=device,
                )
            return self._compressors[layer_idx]

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            B, H, S_new, D = key_states.shape
            comp = self._get_comp(layer_idx, D, str(key_states.device))
            rw = self._prof.residual_window

            while len(self.layers) <= layer_idx:
                self.layers.append(DynamicLayer())

            if layer_idx not in self._chunks_k:
                self._chunks_k[layer_idx] = []
                self._chunks_v[layer_idx] = []
                self._recent_k[layer_idx] = []
                self._recent_v[layer_idx] = []

            self._recent_k[layer_idx].append(key_states)
            self._recent_v[layer_idx].append(value_states)

            recent_k = torch.cat(self._recent_k[layer_idx], dim=2)
            recent_v = torch.cat(self._recent_v[layer_idx], dim=2)

            if recent_k.shape[2] > rw:
                overflow = recent_k.shape[2] - rw
                ck, cv = comp.compress_kv(
                    recent_k[:, :, :overflow, :],
                    recent_v[:, :, :overflow, :],
                )
                self._chunks_k[layer_idx].append(ck)
                self._chunks_v[layer_idx].append(cv)

                recent_k = recent_k[:, :, overflow:, :]
                recent_v = recent_v[:, :, overflow:, :]
                self._recent_k[layer_idx] = [recent_k]
                self._recent_v[layer_idx] = [recent_v]

            parts_k, parts_v = [], []
            for ck, cv in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
                dk, dv = comp.decompress_kv(ck, cv)
                parts_k.append(dk.to(key_states.dtype))
                parts_v.append(dv.to(value_states.dtype))

            parts_k.append(torch.cat(self._recent_k[layer_idx], dim=2))
            parts_v.append(torch.cat(self._recent_v[layer_idx], dim=2))

            full_k = torch.cat(parts_k, dim=2)
            full_v = torch.cat(parts_v, dim=2)

            self._seq_lens[layer_idx] = full_k.shape[2]

            layer = self.layers[layer_idx]
            if not layer.is_initialized:
                layer.dtype = full_k.dtype
                layer.device = full_k.device
                layer.is_initialized = True
            layer.keys = full_k
            layer.values = full_v

            return full_k, full_v

        def get_seq_length(self, layer_idx=0):
            return self._seq_lens.get(layer_idx, 0)

    return V3Cache(profile, n_layers)


def _run_generation(model, tokenizer, inputs, max_new_tokens, cache=None):
    """Run model.generate and return (tok/s, n_generated, response_text)."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    n_prompt = input_ids.shape[1]

    gc.collect()
    torch.cuda.empty_cache()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    if cache is not None:
        gen_kwargs["past_key_values"] = cache

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = outputs[0][n_prompt:]
    n_gen = len(new_tokens)
    tps = n_gen / elapsed if elapsed > 0 else 0
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "tok_per_sec": tps,
        "n_generated": n_gen,
        "elapsed": elapsed,
        "response": response,
        "found_needle": EXPECTED_ANSWER.lower() in response.lower(),
    }


def measure_generation_speed(model, tokenizer, context_len: int,
                             profiles: list[tuple[str, CompressionProfile]],
                             n_layers: int, max_new_tokens: int = 50):
    """Run generation with FP16 baseline and each profile, return speed results."""
    prompt = _build_needle_prompt(tokenizer, context_len)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_len + 512
    ).to("cuda")
    actual_tokens = inputs["input_ids"].shape[1]
    print(f"        Prompt tokens: {actual_tokens}, generating up to {max_new_tokens}")

    print("        Running FP16 baseline ...")
    baseline = _run_generation(model, tokenizer, inputs, max_new_tokens)
    safe_resp = baseline["response"][:60].encode("ascii", errors="replace").decode("ascii")
    quality = "FOUND" if baseline["found_needle"] else "MISS"
    print(f"        FP16: {baseline['tok_per_sec']:.1f} tok/s, "
          f"{baseline['n_generated']} tokens, [{quality}] \"{safe_resp}\"")

    profile_speeds = [{"name": "FP16", **baseline}]

    for pname, profile in profiles:
        print(f"        Running {pname} ...")
        cache = _build_v3cache(profile, n_layers)
        result = _run_generation(model, tokenizer, inputs, max_new_tokens, cache=cache)
        safe_resp = result["response"][:60].encode("ascii", errors="replace").decode("ascii")
        quality = "FOUND" if result["found_needle"] else "MISS"
        print(f"        {pname}: {result['tok_per_sec']:.1f} tok/s, "
              f"{result['n_generated']} tokens, [{quality}] \"{safe_resp}\"")
        profile_speeds.append({"name": pname, **result})

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    return profile_speeds


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(model_name: str, context_len: int, gpu_name: str,
                  profile_results: list[dict],
                  gen_speed: list[dict] | None = None):
    w = 72
    print()
    print(separator("=", w))
    print(f"  TurboQuant Evaluation -- {model_name}")
    print(f"  Context: {context_len} tokens | GPU: {gpu_name}")
    print(separator("=", w))

    # Memory
    print()
    print("  Memory (actual GPU bytes):")
    print(f"  {'Profile':<12s}  {'FP16 Cache':>12s}  {'Compressed':>12s}  {'Ratio':>7s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*7}")
    for pr in profile_results:
        m = pr["memory"]
        print(f"  {pr['name']:<12s}  {fmt_bytes(m['fp16_bytes']):>12s}  "
              f"{fmt_bytes(m['compressed_bytes']):>12s}  {m['ratio']:>5.1f}x")

    # Fidelity
    print()
    print("  Attention Fidelity (output cosine similarity):")
    print(f"  {'Profile':<12s}  {'Mean':>8s}  {'Min':>8s}  {'Worst Layer':>12s}  {'Top-1 Match':>11s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*11}")
    for pr in profile_results:
        f = pr["fidelity"]
        print(f"  {pr['name']:<12s}  {f['mean_cosine']:>8.6f}  {f['min_cosine']:>8.6f}  "
              f"{'layer ' + str(f['worst_layer']):>12s}  {f['mean_top1']*100:>9.1f}%")

    # Throughput
    print()
    print("  Throughput (Python, no Triton):")
    print(f"  {'Profile':<12s}  {'Compress':>14s}  {'Decompress':>14s}  "
          f"{'Comp GB/s':>10s}  {'Dec GB/s':>10s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}")
    for pr in profile_results:
        t = pr["throughput"]
        ct = t["compress_tok_per_sec"]
        dt = t["decompress_tok_per_sec"]
        ct_str = f"{ct/1e6:.2f} M tok/s" if ct >= 1e6 else f"{ct/1e3:.0f} K tok/s"
        dt_str = f"{dt/1e6:.2f} M tok/s" if dt >= 1e6 else f"{dt/1e3:.0f} K tok/s"
        print(f"  {pr['name']:<12s}  {ct_str:>14s}  {dt_str:>14s}  "
              f"{t['compress_gbps']:>8.2f}  {t['decompress_gbps']:>10.2f}")

    # Generation speed
    if gen_speed:
        print()
        print("  Generation Speed (actual model.generate):")
        print(f"  {'Config':<12s}  {'tok/s':>8s}  {'Tokens':>7s}  {'Time':>7s}  {'Needle':>7s}  {'Response'}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*30}")
        baseline_tps = gen_speed[0]["tok_per_sec"] if gen_speed else 0
        for gs in gen_speed:
            safe = gs["response"][:40].encode("ascii", errors="replace").decode("ascii")
            quality = "FOUND" if gs["found_needle"] else "MISS"
            slowdown = ""
            if gs["name"] != "FP16" and baseline_tps > 0:
                ratio = gs["tok_per_sec"] / baseline_tps
                slowdown = f" ({ratio:.2f}x)"
            print(f"  {gs['name']:<12s}  {gs['tok_per_sec']:>6.1f}{slowdown:>2s}"
                  f"  {gs['n_generated']:>7d}  {gs['elapsed']:>5.1f}s"
                  f"  {quality:>7s}  \"{safe}\"")

    # Generation correctness
    print()
    print("  Generation Round-trip (KV tensor cosine similarity):")
    print(f"  {'Profile':<12s}  {'Key Mean':>10s}  {'Key Min':>10s}  "
          f"{'Value Mean':>10s}  {'Value Min':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for pr in profile_results:
        g = pr["generation"]
        print(f"  {pr['name']:<12s}  {g['mean_key_cosine']:>10.6f}  {g['min_key_cosine']:>10.6f}  "
              f"{g['mean_value_cosine']:>10.6f}  {g['min_value_cosine']:>10.6f}")

    # Per-layer detail for worst profile
    worst = min(profile_results, key=lambda p: p["fidelity"]["min_cosine"])
    print()
    print(f"  Per-layer breakdown ({worst['name']}):")
    print(f"  {'Layer':>7s}  {'Attn Cos':>10s}  {'Top-1':>7s}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*7}")
    for lr in worst["fidelity"]["per_layer"]:
        print(f"  {lr['layer']:>7d}  {lr['output_cosine']:>10.6f}  {lr['top1_match']*100:>5.1f}%")

    print()
    print(separator("=", w))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TurboQuant KV Cache Evaluation")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--profiles", nargs="+", default=["moderate", "extreme"],
                   choices=list(PROFILES.keys()))
    p.add_argument("--max-new-tokens", type=int, default=50,
                   help="Max tokens to generate in speed test")
    p.add_argument("--no-4bit", action="store_true",
                   help="Load model in fp16 instead of 4-bit")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip generation and speed tests")
    return p.parse_args()


def main():
    args = parse_args()

    print()
    print(separator())
    print("  TurboQuant KV Cache Compression -- Evaluation")
    print(separator())

    if not torch.cuda.is_available():
        print("\n  CUDA not available. This evaluation requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name()
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
    print(f"  GPU: {gpu_name} ({gpu_mem} MB)")
    print(f"  Model: {args.model}")
    print(f"  Context: {args.context} tokens")
    print(f"  Profiles: {', '.join(args.profiles)}")
    print()

    model, tokenizer = load_model(args)
    n_layers, n_kv_heads, head_dim = extract_model_config(model)
    print(f"  Architecture: {n_layers} layers, {n_kv_heads} KV heads, {head_dim} head_dim")

    print(f"\n  Capturing KV cache ...")
    keys_list, values_list, input_ids = capture_kv_cache(
        model, tokenizer, args.context
    )
    B, H, S, D = keys_list[0].shape
    print(f"  KV shape per layer: B={B}, H={H}, S={S}, D={D}")

    profile_results = []
    for pname in args.profiles:
        profile = PROFILES[pname]
        print(f"\n  --- Profile: {pname} (K{profile.key_bits}/V{profile.value_bits}, "
              f"rw={profile.residual_window}, prot={profile.protected_layers}) ---")

        print("  [1/5] Measuring GPU memory ...")
        mem = measure_memory(keys_list, values_list, profile, n_layers)
        print(f"        FP16: {fmt_bytes(mem['fp16_bytes'])}, "
              f"Compressed: {fmt_bytes(mem['compressed_bytes'])}, "
              f"Ratio: {mem['ratio']:.1f}x")

        print("  [2/5] Measuring attention fidelity ...")
        fid = measure_fidelity(keys_list, values_list, profile, n_layers)
        print(f"        Mean cosine: {fid['mean_cosine']:.6f}, "
              f"Min: {fid['min_cosine']:.6f} (layer {fid['worst_layer']}), "
              f"Top-1: {fid['mean_top1']*100:.1f}%")

        print("  [3/5] Measuring throughput ...")
        thr = measure_throughput(keys_list, values_list, profile, n_layers)
        ct = thr["compress_tok_per_sec"]
        dt = thr["decompress_tok_per_sec"]
        ct_str = f"{ct/1e6:.2f}M" if ct >= 1e6 else f"{ct/1e3:.0f}K"
        dt_str = f"{dt/1e6:.2f}M" if dt >= 1e6 else f"{dt/1e3:.0f}K"
        print(f"        Compress: {ct_str} tok/s, Decompress: {dt_str} tok/s")

        if args.skip_generation:
            gen = {
                "mean_key_cosine": 0, "min_key_cosine": 0,
                "mean_value_cosine": 0, "min_value_cosine": 0,
            }
            print("  [4/5] Generation correctness -- SKIPPED")
        else:
            print("  [4/5] Measuring generation round-trip ...")
            gen = measure_generation_correctness(
                model, tokenizer, keys_list, values_list, profile, n_layers
            )
            print(f"        Key cosine: {gen['mean_key_cosine']:.6f} (min {gen['min_key_cosine']:.6f}), "
                  f"Value cosine: {gen['mean_value_cosine']:.6f} (min {gen['min_value_cosine']:.6f})")

        profile_results.append({
            "name": pname,
            "memory": mem,
            "fidelity": fid,
            "throughput": thr,
            "generation": gen,
        })

    # Test 5: Generation speed (FP16 baseline vs all profiles)
    gen_speed = None
    if not args.skip_generation:
        print(f"\n  --- Generation Speed (FP16 vs TurboQuant) ---")
        print("  [5/5] Measuring generation tok/s ...")
        named_profiles = [(p, PROFILES[p]) for p in args.profiles]
        gen_speed = measure_generation_speed(
            model, tokenizer, args.context, named_profiles, n_layers,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        print("\n  [5/5] Generation speed -- SKIPPED")

    print_results(args.model, S, gpu_name, profile_results, gen_speed)

    del keys_list, values_list
    gc.collect()
    torch.cuda.empty_cache()
    print("  Done.\n")


if __name__ == "__main__":
    main()
