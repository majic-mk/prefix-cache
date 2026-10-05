"""Read-only CPU receipt for the bounded V1 feasibility audit.

Reads frozen files and the existing GPU ledger. It never imports runtime
packages, executes a GPU command, alters the ledger or launches a model.
Only the explicitly named new audit receipt is written, using exclusive create.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CANDIDATE = "artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003"
NATIVE = "artifacts/prefix_io_v1/server11-native-cost-v6-20261003"
RUNTIME = "artifacts/prefix_io_v1/server11-p4-single-file-runtime-v4-20261003"
AUDIT = "artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003"
EXPECTED = {
 CANDIDATE + "/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py":
 "96bfd88dcee7f9c7996518ffe762b87be5ad81ef11d598329d41145e2b01fd87",
 CANDIDATE + "/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_bridge.py":
 "6bec5dd57e25cfc2c0e62540742a781d656ba528cb6c8057cd2402197db93e1a",
 CANDIDATE + "/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_policy.py":
 "bce19c96c7651f69ebb91a515ea3e9e3d6855636b4bd4aa84e60fb2fbc10d673",
 NATIVE + "/gpu-source-lock-native-cost.json":
 "53fe06db2572b6d07b5dfdaac31ca46aed62fe0710bbb63e006a4d7d089d8e22",
 RUNTIME + "/COMMON_SOURCE_LOCK.json":
 "76474f66c3b632850eb1a8ece3650fb44321d95614dd4299bd352a4a1d5b25e5",
 RUNTIME + "/run_p4_single_file_experiment.py":
 "cca77444fa821938c3506c41616f49a454991dd35aa25448abecac59ad90f8b4",
 RUNTIME + "/verify_p4_single_file.py":
 "31d23f3a19a9437ae056be82dc4a2b26a6ea9c587b6f45ceb65a4eacfa553056",
 "experiments/prefix_io_v1/configs/permissions.server11.native-cost.yaml":
 "4a42c67a3e6f51fd28dae1ad27fdd3424b7d21194d4fcae6cc19140cd738b0c1",
 "artifacts/prefix_io_v1/server11-p4-single-file-review-v4-20261003/V4_FINAL_EVIDENCE.json":
 "92f67dccc221e2ed85c71a29ae8e72f14da2c9d3b2bd97bea9b35eb849308c7e",
}
EXPECTED_LEDGER_SECONDS = 22380.561257688794

def reference(path, root):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.parent != (root / AUDIT).resolve(strict=True):
        raise RuntimeError("receipt output must remain inside new audit directory")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only execution must explicitly hide CUDA")
    refs = []
    for name, expected in EXPECTED.items():
        ref = reference(root / name, root)
        if ref["sha256"] != expected:
            raise RuntimeError("frozen reference drift: " + name)
        refs.append(ref)
    ledger_path = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    ledger_raw = ledger_path.read_bytes()
    ledger = json.loads(ledger_raw)
    if abs(ledger["gpu_wall_seconds"] - EXPECTED_LEDGER_SECONDS) > 1e-6:
        raise RuntimeError("GPU ledger changed during CPU-only audit")
    if ledger["active_reservation"] is not None:
        raise RuntimeError("active GPU reservation must be empty")
    git = subprocess.run(["git", "status", "--short", "--untracked-files=no"],
                         cwd=root, capture_output=True, text=True, check=True)
    if git.stdout.strip():
        raise RuntimeError("tracked server workspace changed")
    for package in ("torch", "numpy", "py_kvcache", "vllm"):
        if package in sys.modules:
            raise RuntimeError("unexpected runtime import: " + package)
    cg = Path("/sys/fs/cgroup")
    result = {
        "status": "PASS_READ_ONLY_CPU_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script": reference(Path(__file__).resolve(), root),
        "reference_count": len(refs),
        "reference_checks": refs,
        "verification_scope": "nine named files only; prior complete source locks not rehashed",
        "gpu_operations": 0,
        "new_gpu_seconds": 0,
        "gpu_ledger": {
            "path": str(ledger_path.relative_to(root)),
            "bytes": len(ledger_raw),
            "sha256": hashlib.sha256(ledger_raw).hexdigest(),
            "gpu_wall_seconds": ledger["gpu_wall_seconds"],
            "remaining_original_8h_seconds": 28800 - ledger["gpu_wall_seconds"],
            "active_reservation": ledger["active_reservation"],
        },
        "server": {
            "cpu_max": (cg / "cpu.max").read_text().strip(),
            "memory_max": (cg / "memory.max").read_text().strip(),
            "gpu_device_nodes": sorted(str(x) for x in Path("/dev").glob("nvidia*")),
            "disk_free_bytes": shutil.disk_usage(root).free,
            "tracked_git_status": git.stdout,
        },
        "source_modified": False,
        "old_data_deleted": False,
        "new_performance_claim": False,
    }
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"status": result["status"],
                      "reference_count": len(refs), "gpu_operations": 0,
                      "gpu_wall_seconds": ledger["gpu_wall_seconds"]}))

if __name__ == "__main__":
    main()
