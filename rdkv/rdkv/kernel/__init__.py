"""GPU-only, Triton-based fused decode kernel for the RDKV TriZone packed
cache (spec Sec 8/9, Algorithm 1 Stage 4).

This subpackage is imported lazily by rdkv.decode only when
backend="kernel" is requested. It has no import-time side effects that
would require triton to be installed to use the default native backend.
"""
