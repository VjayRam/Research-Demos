import builtins
import sys

import pytest

from turboquant.kernel._require import require_kernel_backend


def test_require_kernel_backend_raises_on_cpu():
    with pytest.raises(RuntimeError, match="cuda"):
        require_kernel_backend("cpu")


def test_require_kernel_backend_raises_when_triton_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("no triton here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "triton", raising=False)

    with pytest.raises(RuntimeError, match="triton"):
        require_kernel_backend("cuda")


def test_require_kernel_backend_passes_with_cuda_and_triton(monkeypatch):
    import types

    fake_triton = types.ModuleType("triton")
    monkeypatch.setitem(sys.modules, "triton", fake_triton)

    require_kernel_backend("cuda")  # must not raise
