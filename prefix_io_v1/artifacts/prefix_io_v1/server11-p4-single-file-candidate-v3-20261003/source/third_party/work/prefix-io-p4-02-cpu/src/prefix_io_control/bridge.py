"""Inactive execution boundary for CPU tests; not installed in the reactor."""
from collections.abc import Callable
from .config import ConfigError

def native_boundary(mode: str, native: Callable, *, observe: Callable | None = None,
                    record_error: Callable | None = None):
    if mode == "off":
        return native()  # No snapshot, queue, event, or ownership allocation.
    if mode != "shadow":
        raise ConfigError("live policy activation is blocked by P1/P2/P3 gates")
    if observe is not None:
        try:
            observe()
        except Exception as exc:
            if record_error is not None:
                try:
                    record_error(type(exc).__name__)
                except Exception:
                    pass  # Optional observer logging must not alter native execution.
    return native()
