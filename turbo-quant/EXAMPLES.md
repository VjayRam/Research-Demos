# turboquant cookbook

Practical, copy-pasteable recipes for using `turboquant`. For the algorithms'
theory and the package's design history, see `docs/blogs/blog-turboquant.md`
and the specs under `docs/superpowers/specs/`. This document is about *how to
call the package*, not why it works.

## Install

From the workspace root:

```bash
uv sync                        # base package (native backend only)
uv sync --extra kernel         # + Triton kernel backend (CUDA-only, see below)
```

`turboquant` itself has no hard dependency on Triton. Only constructing a
class with `backend="kernel"` ever imports it.

## The three algorithms, at a glance

| Class | What it does | `bits` floor | `d` constraint | Extra method |
|---|---|---|---|---|
| `TurboQuantMSE` | Rotate, per-coordinate Lloyd-Max quantize, unrotate | `bits >= 1` | any | — |
| `TurboQuantProd` | `(bits-1)`-bit MSE stage + 1-bit QJL sign residual | `bits >= 2` | any | `.inner_product(y, compressed)` — unbiased `<x, y>` estimate straight from compressed `x` |
| `PolarQuant` | Recursive Cartesian→polar decomposition, one Lloyd-Max codebook per level | `bits >= 1` | must be a power of 2, `d >= 2` | — |

All three share the same constructor shape and the same `quantize`/`dequantize`
round-trip contract.

## Recipe 1: Quantize and reconstruct a batch of vectors (MSE)

```python
import torch
from turboquant import TurboQuantMSE

d, bits = 64, 4
x = torch.randn(1024, d)  # any shape (..., d)

q = TurboQuantMSE(d, bits, seed=0)          # device auto-detected (cuda if available, else cpu)
indices, norm = q.quantize(x)
x_hat = q.dequantize(indices, norm)

error = (x - x_hat).norm(dim=-1) / x.norm(dim=-1)
print(f"mean relative reconstruction error: {error.mean().item():.4f}")
```

`quantize()` returns `(indices, norm)` — `indices` has the same shape as `x`
(one centroid index per coordinate), `norm` has shape `x.shape[:-1]`. Both are
required by `dequantize()`; store them together if you're persisting a cache.

## Recipe 2: Compressed inner products without ever reconstructing `x` (Prod)

This is Algorithm 2's actual payoff — estimating `<x, y>` directly from the
compressed representation, which is what makes it useful for attention-style
dot products against a compressed KV cache.

```python
import torch
from turboquant import TurboQuantProd

d, bits = 64, 4          # bits >= 2 required — the last bit funds the QJL residual
q = TurboQuantProd(d, bits, seed=0)

x = torch.randn(512, d)  # e.g. a batch of key vectors, compressed once
y = torch.randn(512, d)  # e.g. a query vector, kept in full precision

compressed = q.quantize(x)
estimated_ip = q.inner_product(y, compressed)   # unbiased estimate of (x * y).sum(-1)
exact_ip = (x * y).sum(dim=-1)

print(f"mean abs error vs exact: {(estimated_ip - exact_ip).abs().mean().item():.4f}")
```

`compressed` is a dict: `{"indices", "norm", "qjl_signs", "residual_norm"}`.
Treat it as opaque — pass it straight to `.dequantize()` or `.inner_product()`,
don't reach into its fields (the kernel backend's dict has the identical shape
and keys, so code written against `compressed` never needs to know which
backend produced it).

## Recipe 3: Polar quantization (power-of-2 dimensions only)

```python
import torch
from turboquant import PolarQuant

d, bits = 128, 3   # d MUST be a power of 2 -- ValueError otherwise
q = PolarQuant(d, bits, seed=0)

x = torch.randn(256, d)
compressed = q.quantize(x)          # {"angle_indices": [...per level...], "final_radius": ...}
x_hat = q.dequantize(compressed)
```

`PolarQuant` has no `inner_product()` — it's a reconstruction-quality
algorithm (recursive angle decomposition), not built for the unbiased-estimator
trick `TurboQuantProd` provides.

## Recipe 4: Picking a device

```python
q_cpu = TurboQuantMSE(64, 4, device="cpu")
q_gpu = TurboQuantMSE(64, 4, device="cuda")
q_auto = TurboQuantMSE(64, 4)   # device=None -> "cuda" if torch.cuda.is_available() else "cpu"
```

`device` only ever picks CPU vs. GPU. It's orthogonal to `backend` below —
`backend` picks *which GPU implementation* runs, and is meaningless on CPU.

## Recipe 5: The kernel backend — when and how to use it

Every class also accepts `backend: str = "native"`. Setting `backend="kernel"`
switches to a hand-fused Triton implementation that's faster than the native
(chained-torch-ops) path for the sizes this package targets (`d` in 64/128,
≤16 centroids). It requires `device="cuda"` and the `kernel` extra installed.

```python
q = TurboQuantMSE(64, 4, device="cuda", backend="kernel")
indices, norm = q.quantize(x)          # identical call signature to native
x_hat = q.dequantize(indices, norm)    # identical output, just computed faster
```

**Calling code never needs to branch on `backend`.** `.quantize()`/
`.dequantize()`/`.inner_product()` return byte-identical (or tight-tolerance
float) results regardless of which backend is active — that's a tested
guarantee, not a convention to remember.

### What happens when the kernel backend can't run

`backend` is validated *eagerly*, at construction time — you find out
immediately, not on the first `.quantize()` call:

```python
TurboQuantMSE(64, 4, backend="bogus")
# -> ValueError: backend must be 'native' or 'kernel', got 'bogus'

TurboQuantMSE(64, 4, device="cpu", backend="kernel")
# -> RuntimeError: kernel backend requires device='cuda', got 'cpu'

# backend="kernel" on a machine without the `kernel` extra installed:
# -> RuntimeError pointing at `pip install turboquant[kernel]` (or the
#    workspace equivalent: `uv sync --extra kernel`)
```

Both of the above are hard errors — there is no silent fallback for either.
There's a **third**, different case: if the requested `(d, bits)` combination
would need more shared memory than the actual GPU has available (this
depends on the specific device, not just on `d`), you get a *warning*, and
that instance automatically downgrades itself to `backend="native"` rather
than crashing:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    q = TurboQuantProd(128, 4, device="cuda", backend="kernel")
    # on a GPU with a small enough shared-memory budget for this d:
    #   RuntimeWarning: "...shared memory is too small... Falling back to backend='native'..."
    #   q.backend == "native"   <- check this if you need to know which one actually ran
```

If you want to know which backend an instance actually ended up using (in
case of a shared-memory downgrade), just read `q.backend` after construction
— it always reflects the truth, updated before `__init__` returns.

### A known, deliberate performance exception

`TurboQuantProd`'s kernel backend is slower than native specifically at
`d=128` on GPUs with a tight shared-memory budget (measured ~1.2-2.7x slower
on an RTX 4070 Laptop) — correctness is unaffected, only latency. `d=64`, and
every configuration of `TurboQuantMSE` and `PolarQuant`, meet or beat native.
See `docs/superpowers/specs/2026-08-28-turboquant-kernel-backend-design.md`'s
"Known Limitations" section for the full detail. If you're deploying on a
GPU with a larger shared-memory budget (e.g. a data-center card), re-run
`examples/run_perf_benchmark.py` on your actual target hardware before
assuming this applies to you — it's a property of the specific GPU, not the
algorithm.

## Recipe 6: Reproducibility

Every class takes a `seed` — it fixes both the random rotation matrix
(`TurboQuantMSE`/`TurboQuantProd`/`PolarQuant` all rotate) and, for
`TurboQuantProd`, the QJL projection matrix (seeded at `seed + 1` internally,
so it's never identical to the rotation). Same `(d, bits, seed, device)` always
produces the same quantizer:

```python
q1 = TurboQuantMSE(64, 4, seed=42)
q2 = TurboQuantMSE(64, 4, seed=42)
torch.equal(q1.rotation, q2.rotation)  # True
```

Rotation and QJL matrices are cached per `(d, seed, device)` process-wide
(see `rotation.py`/`qjl.py`), so constructing many quantizers with the same
`(d, seed, device)` doesn't regenerate the matrix each time.

## Recipe 7: Plugging into a HuggingFace KV cache

`examples/kv_cache_hook.py` has a working `DynamicCache` subclass that
round-trips every key/value tensor through a quantizer on every `update()`
call — useful for measuring how a given `(algorithm, bits)` choice affects
real generation quality, not a production compressed-cache implementation
(it still stores full-precision reconstructed tensors, just quantized and
immediately dequantized in between):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantMSE
from examples.kv_cache_hook import QuantizingCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

head_dim = model.config.hidden_size // model.config.num_attention_heads
key_q = TurboQuantMSE(head_dim, bits=4, device=model.device)
value_q = TurboQuantMSE(head_dim, bits=4, device=model.device)

cache = QuantizingCache(key_q, value_q)
inputs = tokenizer("The quick brown fox", return_tensors="pt").to(model.device)
output = model.generate(**inputs, past_key_values=cache, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

Swap `key_q`/`value_q` for `TurboQuantProd` or `PolarQuant` instances the same
way — `QuantizingCache._round_trip` handles both the `(indices, norm)` tuple
return (MSE) and the dict return (Prod/Polar) transparently.

## Recipe 8: Running the bundled benchmarks

Three scripts under `examples/`, each answering a different question:

```bash
# "How much does quantizing the KV cache hurt real generation quality?"
uv run python examples/run_benchmark.py --algorithm mse prod --bits 1 2 3 4

# "Does empirical distortion match the solved Lloyd-Max theoretical bound?"
# (also runs the perplexity sweep above, plus a distortion-vs-theory check)
uv run python examples/run_experiments.py --algorithms mse prod polar

# "Is the kernel backend actually faster than native, on this GPU?"
uv run python examples/run_perf_benchmark.py --devices cuda --dims 64 128
```

All three accept `--smoke-test` for a fast, tiny-config sanity check that
doesn't require downloading a real model or a full sweep — useful for
verifying an environment is set up correctly before a real run. Results are
written as CSVs under `examples/results/` (gitignored).

## Common pitfalls

- **`PolarQuant(100, 4)` raises `ValueError`** — `d` must be a power of 2.
  Round your head dimension up/down, or use `TurboQuantMSE`/`TurboQuantProd`
  instead, which have no such constraint.
- **`TurboQuantProd(64, 1)` raises `ValueError`** — `bits >= 2` is required
  since one bit is reserved for the QJL residual sign; there's no valid
  1-bit `Prod` configuration. Use `TurboQuantMSE` if you need 1-bit.
- **Constructing with `backend="kernel"` on CPU raises immediately, not on
  first use** — check for this at startup if you're building quantizers from
  user-supplied config, rather than deep inside a request path.
- **A `RuntimeWarning` about shared memory doesn't mean your code is broken**
  — it means that specific `(d, bits)` on that specific GPU silently now runs
  native instead. Check `.backend` on the instance if you need to know for
  certain, or you're auditing why latency looks like native's.
