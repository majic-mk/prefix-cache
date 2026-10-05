"""CPU file-byte protection only; never imports the model or issues GPU work."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

PACKER_REL = "artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/pack_repeat_delivery.py"
PACKER_SHA = "c29f26ecba75658369cfd5d2a41ad65c2119ab9291bb2f425352c6fed3869b63"
LEDGER_SHA = "60368b94c494d15227eb528273e7b4a1a20119c6f371767ffde5cc96aa8f52f9"
AUDIT_REL = "artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("BEFORE", "AFTER"), required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    audit = root / AUDIT_REL
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("actual CPU-only invocation requires CUDA_VISIBLE_DEVICES empty")
    path = root / PACKER_REL
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != PACKER_SHA:
        raise RuntimeError("existing reviewed byte verifier changed")
    spec = importlib.util.spec_from_file_location("_reviewed_fixed3_byte_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ledger, ledger_raw, ledger_ref = module.checked_ledger(root)
    if ledger_ref["sha256"] != LEDGER_SHA:
        raise RuntimeError("original idle ledger drifted before CPU audit")
    if (audit / "CPU_AUDIT_START_LEDGER_SNAPSHOT.json").read_bytes() != ledger_raw:
        raise RuntimeError("CPU audit start snapshot differs from original ledger")
    started = time.monotonic()
    lock, lock_ref, source_map = module.verify_source_closure(root)
    archive_ref = module.file_ref(
        root,
        "artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/REPEAT_REPEAT_DELIVERY_fixed3-final.tar.gz",
    )
    if archive_ref["sha256"] != "5acd907b13d0a3999a7fddef7eb885910dfad1ffbe2b85aeed4dd8111139b5d5":
        raise RuntimeError("completed fixed-three evidence archive drifted")
    state = subprocess.run(
        ["git", "diff", "--name-only"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    if state:
        raise RuntimeError("tracked server workspace changes require separate audit: " + repr(state))
    proof = {
        "schema": "server12_fixed_evidence_cpu_source_protection_v1",
        "status": "PASS_COMPLETE_SOURCE_BYTES_AND_UNCHANGED_IDLE_LEDGER",
        "phase": args.phase,
        "elapsed_cpu_file_verification_seconds": time.monotonic() - started,
        "source_lock_ref": lock_ref,
        "complete_source_rows_verified": len(source_map),
        "complete_source_total_bytes_verified": sum(row["bytes"] for row in source_map.values()),
        "source_map": [source_map[k] for k in sorted(source_map)],
        "existing_verifier_ref": module.file_ref(root, PACKER_REL),
        "actual_completed_GPU_archive_ref": archive_ref,
        "original_ledger_ref": ledger_ref,
        "original_GPU_wall_seconds": ledger["gpu_wall_seconds"],
        "original_GPU_remaining_seconds": 28800 - ledger["gpu_wall_seconds"],
        "tracked_server_changes": state,
        "GPU_operations_this_action": 0,
        "model_or_shared_library_loaded": False,
        "data_deletions": 0,
        "new_qualification_issued": False,
        "normal_qualification_passed": False,
        "P4_strategy_effect_verified": False,
    }
    if args.phase == "AFTER":
        before = json.loads((audit / "SOURCE_PROTECTION_BEFORE.json").read_bytes())
        for key in ("source_lock_ref", "source_map", "actual_completed_GPU_archive_ref", "original_ledger_ref"):
            if proof[key] != before[key]:
                raise RuntimeError("CPU audit changed protected bytes: " + key)
    if module.checked_ledger(root)[1] != ledger_raw:
        raise RuntimeError("original GPU ledger changed during CPU-only byte audit")
    result_path = audit / ("SOURCE_PROTECTION_" + args.phase + ".json")
    result_ref = module.put_bytes(root, result_path.relative_to(root).as_posix(), module.document(proof))
    print(json.dumps({k: v for k, v in proof.items() if k != "source_map"} | {"proof_ref": result_ref}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

