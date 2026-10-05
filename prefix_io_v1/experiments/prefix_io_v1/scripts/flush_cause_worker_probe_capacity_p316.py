"""Experiment-only installation; the native uni executor co-locates scheduler/worker."""
import os
from run_p3_native_pilot_capacity_p316 import worker_probe as original_probe

def worker_probe(worker, action, limits=None):
    if action == "install":
        from prefix_io_control.native_flush_probe import NativeFlushProbe
        result = original_probe(worker, action, limits)
        probe = NativeFlushProbe("flush-" + str(os.getpid()))
        probe.install()
        worker._prefix_flush_diagnostic = probe
        result["flush_diagnostic"] = probe.export()
        return result
    try:
        result = original_probe(worker, action, limits)
        probe = worker._prefix_flush_diagnostic
        result["flush_diagnostic"] = probe.export()
        return result
    finally:
        if action == "finish":
            worker._prefix_flush_diagnostic.uninstall()
