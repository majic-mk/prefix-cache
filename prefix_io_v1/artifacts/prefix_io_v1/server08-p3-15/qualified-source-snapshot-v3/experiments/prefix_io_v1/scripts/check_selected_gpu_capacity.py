#!/usr/bin/env python3
"""Read-only NVML capacity gate for the one approved GPU, via budget runner."""
import json
import os
import pynvml

selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if selected != "GPU-8b500efe-1a50-0e8e-b21e-716807eebedf":
    raise RuntimeError("exact approved GPU UUID required")
pynvml.nvmlInit()
try:
    handle = pynvml.nvmlDeviceGetHandleByUUID(selected)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    compute = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    result = {"uuid": pynvml.nvmlDeviceGetUUID(handle),
              "free_bytes": info.free, "used_bytes": info.used,
              "total_bytes": info.total,
              "compute_process_count": len(compute),
              "min_required_free_bytes": 24 * 1024**3,
              "cuda_context_created": False,
              "scope": "NVML selected GPU capacity only"}
    result["status"] = "PASSED" if info.free >= result["min_required_free_bytes"] and not compute else "BLOCKED_OCCUPIED"
    print(json.dumps(result, indent=2))
finally:
    pynvml.nvmlShutdown()
raise SystemExit(0 if result["status"] == "PASSED" else 2)
