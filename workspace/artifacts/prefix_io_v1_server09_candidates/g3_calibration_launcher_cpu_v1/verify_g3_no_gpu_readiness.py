"""Read-only CPU readiness, explicitly deferring the absent GPU driver.

The original strict GPU launch/execution gates are never patched or called.
This verifier returns no SDK pin and grants no driver/GPU qualification.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--source-lock", required=True)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    here = Path(__file__).resolve().parent
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    raw = (here / "g3_calibration_plan.py").read_bytes()
    assert len(raw) == 24475 and hashlib.sha256(raw).hexdigest() == "196700d96733ec07c5a4c08c6104ab31a2a36548cf005fbb8e11b6a769f6f0c0"
    plan = load(here / "g3_calibration_plan.py", "_cpu_g3_readiness_plan")
    lock = root / a.source_lock
    raw = lock.read_bytes()
    lock_ref = dict(path=a.source_lock, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    refs = plan.verify_source_lock(root, lock_ref)
    common_ref = refs["artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py"]
    common = load(plan.checked_ref(root, common_ref), "_cpu_g3_readiness_original_common")
    sdk = common.load_ref(root, refs[common.SDK_MODULE])
    inventory_path = plan.checked_ref(root, refs[common.SDK_INVENTORY])
    proof_path = plan.checked_ref(root, refs[common.SDK_PROOF])
    inventory = sdk.read_json(inventory_path)
    proof = sdk.read_json(proof_path)
    # Reuse original full-tree verification without a synthetic driver pin.
    verified_files = 0
    verified_bytes = 0
    for name in ("bin", "include", "nvvm"):
        entry = inventory["directories"][name]
        files = []
        for row in entry["files"]:
            assert row["path"] == str(Path(entry["root"]) / row["relative"])
            files.append(dict(path=row["relative"], bytes=row["bytes"], sha256=row["sha256"]))
        files.sort(key=lambda row: row["path"])
        assert entry["file_count"] == len(files)
        tree = dict(path=entry["root"], bytes=entry["total_bytes"],
                    sha256=sdk.tree_manifest_digest(files), files=files)
        sdk.verify_tree(tree)
        verified_files += len(files)
        verified_bytes += tree["bytes"]
    for row in [inventory["cudart"], *proof["output_refs"]]:
        sdk.digest_file(Path(row["path"]), row["bytes"], row["sha256"])
    expected_driver = inventory["driver"]
    driver = Path(expected_driver["path"])
    assert driver.is_file() and not driver.is_symlink()
    devices = {name: Path(name).exists() for name in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm")}
    if driver.stat().st_size == 0 and not any(devices.values()):
        driver_status = "DEFERRED_ZERO_LENGTH_DRIVER_WITH_NO_GPU_DEVICE_NODES"
        strict_driver_verified = False
    else:
        sdk.digest_file(driver, expected_driver["bytes"], expected_driver["sha256"])
        driver_status = "ACTUAL_DRIVER_BYTES_MATCH_ONLY_DEVICE_RUNTIME_UNVERIFIED"
        strict_driver_verified = True
    common.verify_ninja(root, refs)
    common.load_optional_probe(root, refs)
    ledger_path = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    st = os.statvfs(root)
    snapshot = dict(origin="actual_statvfs", captured_unix=time.time(), primary_root=str(root),
                    free_bytes=st.f_bavail * st.f_frsize)
    budget = plan.validate_budget_and_storage(ledger, snapshot)
    assert not any(n.split(".")[0] in ("torch", "vllm", "py_kvcache", "flashinfer") for n in sys.modules)
    assert ledger_path.read_bytes() == ledger_bytes
    print(json.dumps(dict(status="CPU_ENTRY_READY_REQUIRES_GPU_POWER_AND_NEW_SCOPE",
        actual_full_source_bytes_verified=True, source_count=len(refs), source_lock=lock_ref,
        verified_toolkit_files=verified_files, verified_toolkit_bytes=verified_bytes,
        cuda13_headers_compilers_runtime_and_original_CPU_outputs_verified=True,
        driver_status=driver_status, actual_driver_bytes=driver.stat().st_size,
        expected_driver=expected_driver, strict_driver_verified=strict_driver_verified,
        device_nodes=devices, actual_GPU_UUID_verified=False, actual_free_VRAM_verified=False,
        original_strict_GPU_asset_gate_unchanged=True, no_runtime_gate_bypass=True,
        budget_and_storage=budget, ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
        actual_GPU_operations=0, framework_imports=0, GPU_jobs_launched=0,
        new_authorization_created=False, production_qualified=False, cost_qualified=False,
        effect_verified=False, performance_claim=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
