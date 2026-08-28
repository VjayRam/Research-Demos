"""Correctness parity between the native and kernel backends.

Skipped entirely on machines without CUDA or without triton installed --
the kernel backend is CUDA-only by design (see the kernel backend spec).
"""

import pytest
import torch

from turboquant import TurboQuantMSE

CUDA_AND_TRITON_AVAILABLE = torch.cuda.is_available()
if CUDA_AND_TRITON_AVAILABLE:
    try:
        import triton  # noqa: F401
    except ImportError:
        CUDA_AND_TRITON_AVAILABLE = False

requires_kernel_backend = pytest.mark.skipif(
    not CUDA_AND_TRITON_AVAILABLE, reason="kernel backend requires CUDA and triton"
)


@requires_kernel_backend
def test_mse_kernel_quantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = TurboQuantMSE(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantMSE(d, bits, seed=1, device="cuda", backend="kernel")

    native_indices, native_norm = native.quantize(x)
    kernel_indices, kernel_norm = kernel.quantize(x)

    assert torch.equal(native_indices, kernel_indices)
    assert torch.allclose(native_norm, kernel_norm, atol=1e-5)


@requires_kernel_backend
def test_mse_kernel_dequantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = TurboQuantMSE(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantMSE(d, bits, seed=1, device="cuda", backend="kernel")

    indices, norm = native.quantize(x)
    native_x_hat = native.dequantize(indices, norm)
    kernel_x_hat = kernel.dequantize(indices, norm)

    assert torch.allclose(native_x_hat, kernel_x_hat, atol=1e-5)


def test_kernel_backend_rejects_cpu():
    with pytest.raises(RuntimeError, match="cuda"):
        TurboQuantMSE(64, 4, device="cpu", backend="kernel")


def test_invalid_backend_raises_value_error():
    with pytest.raises(ValueError, match="backend"):
        TurboQuantMSE(64, 4, device="cpu", backend="bogus")


from turboquant import TurboQuantProd


@requires_kernel_backend
def test_prod_kernel_quantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = TurboQuantProd(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantProd(d, bits, seed=1, device="cuda", backend="kernel")

    native_out = native.quantize(x)
    kernel_out = kernel.quantize(x)

    assert torch.equal(native_out["indices"], kernel_out["indices"])
    assert torch.allclose(native_out["norm"], kernel_out["norm"], atol=1e-5)
    assert torch.equal(native_out["qjl_signs"], kernel_out["qjl_signs"])
    assert torch.allclose(native_out["residual_norm"], kernel_out["residual_norm"], atol=1e-5)


@requires_kernel_backend
def test_prod_kernel_dequantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = TurboQuantProd(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantProd(d, bits, seed=1, device="cuda", backend="kernel")

    compressed = native.quantize(x)
    native_x_hat = native.dequantize(compressed)
    kernel_x_hat = kernel.dequantize(compressed)

    assert torch.allclose(native_x_hat, kernel_x_hat, atol=1e-5)


@requires_kernel_backend
def test_prod_kernel_inner_product_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")
    y = torch.randn(32, d, device="cuda")

    native = TurboQuantProd(d, bits, seed=1, device="cuda", backend="native")
    kernel = TurboQuantProd(d, bits, seed=1, device="cuda", backend="kernel")

    compressed = native.quantize(x)
    native_ip = native.inner_product(y, compressed)
    kernel_ip = kernel.inner_product(y, compressed)

    assert torch.allclose(native_ip, kernel_ip, atol=1e-4)


def test_prod_kernel_backend_rejects_cpu():
    with pytest.raises(RuntimeError, match="cuda"):
        TurboQuantProd(64, 4, device="cpu", backend="kernel")


from turboquant import PolarQuant


@requires_kernel_backend
def test_polar_kernel_quantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = PolarQuant(d, bits, seed=1, device="cuda", backend="native")
    kernel = PolarQuant(d, bits, seed=1, device="cuda", backend="kernel")

    native_out = native.quantize(x)
    kernel_out = kernel.quantize(x)

    assert len(native_out["angle_indices"]) == len(kernel_out["angle_indices"])
    for native_level, kernel_level in zip(native_out["angle_indices"], kernel_out["angle_indices"]):
        assert torch.equal(native_level, kernel_level)
    assert torch.allclose(native_out["final_radius"], kernel_out["final_radius"], atol=1e-5)


@requires_kernel_backend
def test_polar_kernel_dequantize_matches_native():
    d, bits = 64, 4
    torch.manual_seed(0)
    x = torch.randn(32, d, device="cuda")

    native = PolarQuant(d, bits, seed=1, device="cuda", backend="native")
    kernel = PolarQuant(d, bits, seed=1, device="cuda", backend="kernel")

    compressed = native.quantize(x)
    native_x_hat = native.dequantize(compressed)
    kernel_x_hat = kernel.dequantize(compressed)

    assert torch.allclose(native_x_hat, kernel_x_hat, atol=1e-4)


def test_polar_kernel_backend_rejects_cpu():
    with pytest.raises(RuntimeError, match="cuda"):
        PolarQuant(64, 4, device="cpu", backend="kernel")
