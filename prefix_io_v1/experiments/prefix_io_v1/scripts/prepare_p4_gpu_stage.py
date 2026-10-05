"""CPU-only P4 preparation. This file never launches or reserves GPU work."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

P4_ARTIFACT = "artifacts/prefix_io_v1/server08-p4-01-cpu"
AUTH = P4_ARTIFACT + "/CPU_ONLY_AUTHORIZATION.json"
LEDGER = "experiments/prefix_io_v1/gpu-budget-ledger.json"
PRODUCTION = "artifacts/prefix_io_v1/server08-p3-16/conditional-interference-table.json"

def file_ref(root: Path, relative: str) -> dict:
    p = root / relative
    raw = p.read_bytes()
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

def inspect_cpu_authorization(root: Path) -> dict:
    a = json.loads((root / AUTH).read_text())
    required = {"allow_project_local_edits": True, "allow_cpu_tests": True,
                "allow_gpu_initialization": False, "allow_gpu_runs": False,
                "allow_model_downloads": False, "allow_driver_or_system_changes": False}
    if a.get("schema_version") != 1 or type(a.get("schema_version")) is not int:
        raise ValueError("invalid CPU authorization schema")
    if a.get("status") != "USER_AUTHORIZED_P4_CPU_ONLY":
        raise ValueError("this preparation command requires the recorded CPU-only scope")
    if any(type(a.get(k)) is not bool or a[k] is not v for k, v in required.items()):
        raise ValueError("CPU authorization fields must be explicit and unchanged")
    base = a.get("base_permissions", {})
    if file_ref(root, base["path"]) != base:
        raise ValueError("base permissions changed after CPU authorization")
    return a

def plan(root: Path) -> dict:
    inspect_cpu_authorization(root)
    return {
        "schema_version": 1,
        "status": "P4_CPU_PREPARATION_GPU_BLOCKED",
        "new_gpu_runs": 0, "gpu_initialized": False,
        "gpu_availability_probed": False, "gpu_unavailable": "user_reported",
        "budget_reserved_seconds": 0,
        "authorization": file_ref(root, AUTH), "ledger": file_ref(root, LEDGER),
        "conditional_P3_table": file_ref(root, PRODUCTION),
        "P3_qualified_table_is_P4_production_table": False,
        "slo": None, "positive_effect_established": False,
        "full_P4_production_implementation_complete": False,
        "remaining_production_work": [
            "D qualified native prediction producer and uncertainty source",
            "I/J independent semantic paired GPU verifier and qualified CostTable loader",
            "I/J production quota/batch activation and native liveness qualification"
        ],
        "factorial": {
            "ordinary_budget_base": "same frozen fixed stage caps for C00/C01/C10/C11",
            "C00": "fixed caps only; dependency off; interference off",
            "C01": "same caps; dependency off; interference on",
            "C10": "same caps; dependency on; interference off",
            "C11": "same caps; dependency on; interference on",
            "independent_main_reference": "U (strongest P3 baseline)",
            "I_J_without_qualified_production_table": "U original order, no new performance quota",
            "must_freeze": ["model/source bytes", "KV/staging actual allocation", "exact admission",
                            "preload/fusion/pipeline", "initial cache", "stop/drain", "SLO", "ordinary caps"],
            "factorial_difference_is_not_U_effect": True
        },
        "next_gpu_gates": [
            {"id": "G0", "required": "new explicit authorization, available approved GPU, storage/budget preflight",
             "status": "BLOCKED_AUTHORIZATION_AND_AVAILABILITY"},
            {"id": "G1", "required": "P4 source-locked original CUDA/AIO primitive, off/shadow native safety qualification",
             "status": "BLOCKED_GPU_NATIVE_ABI_QUALIFICATION"},
            {"id": "G2", "required": "real owner generations/refs and full parent closure; no premature completion; drain under zero ordinary credit",
             "status": "BLOCKED_GPU_RELEASE_AND_LIVENESS_EVIDENCE"},
            {"id": "G3", "required": "held-out production-state delta or joint-total interference table; drift and uncertainty",
             "status": "BLOCKED_PRODUCTION_COST_QUALIFICATION"},
            {"id": "G4", "required": "all native modes, preserved exact KV and full model outputs, actual bounded live shadow",
             "status": "BLOCKED_GPU_MODEL_SHADOW_QUALIFICATION"},
            {"id": "G5", "required": "P4 common-versus-research causal validation with independent U reference; P5 authorization separately",
             "status": "BLOCKED_GPU_EFFECT_EVALUATION"}
        ],
        "prepared_command_is_authorization": False,
        "no_actions": ["GPU imports/initialization", "GPU status probe", "GPU budget reservation",
                       "downloads", "driver/system changes", "cache deletion/merge", "P5-P7 execution"]
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--check-launch", action="store_true",
                    help="report denial before any GPU import, process or budget action")
    args = ap.parse_args(argv)
    root = args.project.resolve()
    result = plan(root)
    if args.check_launch:
        result["status"] = "BLOCKED_CPU_ONLY_NO_GPU_LAUNCH"
    if args.output:
        dest = args.output.resolve()
        dest.relative_to(root / P4_ARTIFACT)
        if dest.exists():
            raise ValueError("do not overwrite a preparation receipt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
        ("status", "new_gpu_runs", "gpu_initialized", "budget_reserved_seconds")}))
    return 78 if args.check_launch else 0

if __name__ == "__main__":
    raise SystemExit(main())
