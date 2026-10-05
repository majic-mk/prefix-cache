"""Read-only deployment gate. No GPU launch or model download is implemented."""
import argparse
import json
from pathlib import Path
from .config import read_yaml, validate_controller, validate_permissions

def inspect(controller, permissions, capability):
    c = validate_controller(controller)
    p = validate_permissions(permissions)
    runtime = capability.get("runtime", {})
    blockers = []
    if not p["allow_gpu_runs"]:
        blockers.append("GPU permission is false")
    if p["max_gpu_hours"] is None:
        blockers.append("GPU budget is null")
    if not p["approved_gpu_ids"]:
        blockers.append("no approved GPU IDs")
    for key in ("approved_experiment_root", "approved_dependency_root"):
        if p[key] is None:
            blockers.append(key + " is null")
    for key in ("VLLM_AVAILABLE", "PLAN_API_AVAILABLE", "TORCH_COPY_AVAILABLE"):
        if runtime.get(key) is not True:
            blockers.append("real runtime capability missing: " + key)
    if capability.get("io_uring", {}).get("available") is not True:
        blockers.append("io_uring unavailable")
    if capability.get("real_handler_verified") is not True:
        blockers.append("real handler has not been verified")
    if c["mode"] not in ("off", "shadow"):
        blockers.append("research policy is not wired; phase gates remain closed")
    return {"status": "BLOCKED" if blockers else "PRECHECK_ONLY",
            "blockers": blockers, "mode": c["mode"], "gpu_executed": False,
            "gpu_verified": False, "does_not_authorize_execution": True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--permissions", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(read_yaml(args.controller), read_yaml(args.permissions),
                     json.loads(args.capabilities.read_text()))
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "BLOCKED" else 0
if __name__ == "__main__":
    raise SystemExit(main())
