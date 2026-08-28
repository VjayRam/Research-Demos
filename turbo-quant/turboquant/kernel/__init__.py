"""GPU-only, Triton-based kernel backend for turboquant algorithms.

This subpackage is imported lazily by the core algorithm classes only when
``backend="kernel"`` is requested. It has no import-time side effects that
would require ``triton`` to be installed to use the default native backend.
"""
