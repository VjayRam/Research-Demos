"""Guard for the kernel backend's environment requirements."""


def require_kernel_backend(device: str) -> None:
    """Raise RuntimeError if the kernel backend cannot run on this device.

    The kernel backend is CUDA-only (Triton targets NVIDIA GPUs) and requires
    the optional `triton` dependency. Both failures are explicit errors, not
    silent fallbacks to the native backend.
    """
    if device != "cuda":
        raise RuntimeError(
            f"kernel backend requires device='cuda', got {device!r}. "
            "The kernel backend does not support CPU."
        )
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "kernel backend requires the 'triton' package, which is not "
            "installed. Install it via the 'kernel' extra: "
            "`pip install rdkv[kernel]`."
        ) from exc
