import pytest

from rdkv.kernel._require import require_kernel_backend


def test_raises_on_non_cuda_device():
    with pytest.raises(RuntimeError, match="requires device='cuda'"):
        require_kernel_backend("cpu")


def test_raises_with_install_hint_when_triton_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "triton", None)  # force ImportError on `import triton`
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("no triton")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="requires the 'triton' package"):
        require_kernel_backend("cuda")
