"""Device (CPU/CUDA) support tests for rotation/qjl matrices and the top-level
quantizer classes."""

import pytest
import torch

from turboquant.cartesian import TurboQuantMSE, TurboQuantProd
from turboquant.polar import PolarQuant
from turboquant.qjl import generate_qjl_matrix
from turboquant.rotation import generate_rotation_matrix

CUDA_AVAILABLE = torch.cuda.is_available()


def test_generate_rotation_matrix_explicit_cpu():
    q = generate_rotation_matrix(d=8, seed=0, device="cpu")
    assert q.device.type == "cpu"


def test_generate_qjl_matrix_explicit_cpu():
    s = generate_qjl_matrix(d=8, seed=0, device="cpu")
    assert s.device.type == "cpu"


def test_generate_rotation_matrix_auto_detect_does_not_crash():
    q = generate_rotation_matrix(d=8, seed=0, device=None)
    expected = "cuda" if CUDA_AVAILABLE else "cpu"
    assert q.device.type == expected


def test_generate_qjl_matrix_auto_detect_does_not_crash():
    s = generate_qjl_matrix(d=8, seed=0, device=None)
    expected = "cuda" if CUDA_AVAILABLE else "cpu"
    assert s.device.type == expected


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_generate_rotation_matrix_cuda():
    q = generate_rotation_matrix(d=8, seed=0, device="cuda")
    assert q.device.type == "cuda"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_generate_qjl_matrix_cuda():
    s = generate_qjl_matrix(d=8, seed=0, device="cuda")
    assert s.device.type == "cuda"


def test_turboquant_mse_end_to_end_cpu():
    q = TurboQuantMSE(d=16, bits=3, seed=0, device="cpu")
    x = torch.randn(4, 16, device="cpu")
    indices, norm = q.quantize(x)
    x_hat = q.dequantize(indices, norm)
    assert x_hat.shape == x.shape
    assert x_hat.device.type == "cpu"
    assert torch.isfinite(x_hat).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_turboquant_mse_end_to_end_cuda():
    q = TurboQuantMSE(d=16, bits=3, seed=0, device="cuda")
    x = torch.randn(4, 16, device="cuda")
    indices, norm = q.quantize(x)
    x_hat = q.dequantize(indices, norm)
    assert x_hat.shape == x.shape
    assert x_hat.device.type == "cuda"
    assert torch.isfinite(x_hat).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_turboquant_prod_end_to_end_cuda():
    q = TurboQuantProd(d=16, bits=3, seed=0, device="cuda")
    x = torch.randn(4, 16, device="cuda")
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert x_hat.shape == x.shape
    assert x_hat.device.type == "cuda"
    assert torch.isfinite(x_hat).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_polarquant_end_to_end_cuda():
    q = PolarQuant(d=16, bits=2, seed=0, device="cuda")
    x = torch.randn(4, 16, device="cuda")
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert x_hat.shape == x.shape
    assert x_hat.device.type == "cuda"
    assert torch.isfinite(x_hat).all()


def test_auto_detect_device_does_not_crash():
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    q = TurboQuantMSE(d=8, bits=2, seed=0)
    x = torch.randn(4, 8, device=device)
    indices, norm = q.quantize(x)
    x_hat = q.dequantize(indices, norm)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()
