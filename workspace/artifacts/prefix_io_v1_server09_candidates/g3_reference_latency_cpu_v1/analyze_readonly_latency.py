"""CPU-only sums of frozen manifests and static call-path accounting."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parent
BASE = CANDIDATES / "g3_calibration_launcher_cpu_v1"
ARTIFACTS = CANDIDATES.parent
GIB = 1024 ** 3
inputs = {}

def document(path):
    raw = path.read_bytes()
    inputs[str(path.resolve())] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw)

def source(path):
    raw = path.read_bytes()
    inputs[str(path.resolve())] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

locks = {}
for path in (BASE / "gpu-source-lock-reference-v1.json",
             BASE / "gpu-source-lock-metrics-v3-final.json",
             CANDIDATES / "g2_normal_worker_site_cache_v4_final/gpu-source-lock-candidate.json"):
    data = document(path)
    rows = data["files"]
    model = [row for row in rows if row["path"].startswith("models/")]
    total, model_total = sum(row["bytes"] for row in rows), sum(row["bytes"] for row in model)
    assert len({row["path"] for row in rows}) == len(rows)
    locks[path.name + "@" + path.parent.name] = dict(
        file_count=len(rows), sum_ref_bytes=total, logical_payload_GiB=total / GIB,
        model_count=len(model), model_bytes=model_total, model_GiB=model_total / GIB,
        model_percent=100 * model_total / total,
        top_13=sorted(rows, key=lambda row: row["bytes"], reverse=True)[:13])

inventory_path = ARTIFACTS / "prefix_io_v1_server09_g2_20261001/cuda13-cpu/CUDA13_SOURCE_INVENTORY.json"
proof_path = inventory_path.with_name("CPU_COMPILE_LINK_RESULT.json")
inventory, proof = document(inventory_path), document(proof_path)
tree = inventory["directories"]
tree_total = sum(row["total_bytes"] for row in tree.values())
assert all(row["file_count"] == len(row["files"]) and
           row["total_bytes"] == sum(f["bytes"] for f in row["files"]) for row in tree.values())
sdk_verify_asset_hash_bytes = (tree_total + inventory["cudart"]["bytes"] + inventory["driver"]["bytes"]
    + inputs[str(inventory_path.resolve())]["bytes"] + inputs[str(proof_path.resolve())]["bytes"]
    + sum(row["bytes"] for row in proof["output_refs"]))

measured = document(BASE / "REFERENCE_SERVER_FULL_PREFLIGHT_01.json")
for path in (BASE / "run_g3_reference_pilot.py", BASE / "g3_reference_plan.py",
             BASE / "g3_reference_runtime.py", BASE / "g3_calibration_runtime_metrics_v2.py",
             BASE / "g3_calibration_plan_metrics_v2.py", BASE / "run_g3_calibration_pilot_v3.py",
             CANDIDATES / "g2_normal_worker_site_cache_v4_final/run_g2_normal_model_lifecycle.py",
             CANDIDATES / "g2_cuda13_toolchain/cuda13_sdk_overlay.py",
             ARTIFACTS / "prefix_io_v1_server09_g1_20261001/current-source/run_gpu_stage.py"):
    source(path)
total = next(v["sum_ref_bytes"] for k, v in locks.items() if k.startswith("gpu-source-lock-reference-v1"))

result = dict(
    schema_version=1, status="PASS_CPU_MANIFEST_AND_STATIC_CALL_ACCOUNTING_ONLY",
    captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    method="Read actual local mirrored JSON/source bytes only. Sums are logical source payload read/hash volumes implied by code; no remote files, GPU timing or block-device physical I/O was measured.",
    input_refs=inputs, frozen_locks=locks,
    reference_success_call_path=[
        dict(stage="parent launch gates/full_sources", full_lock_passes=1, sdk_verify_assets_passes=1, inside_GPU_guard=False),
        dict(stage="guard child execute gates/full_sources", full_lock_passes=1, sdk_verify_assets_passes=1, inside_GPU_guard=True),
        dict(stage="delegated runtime execute preflight_runtime", full_lock_passes=0, sdk_verify_assets_passes=1, inside_GPU_guard=True),
        dict(stage="configure/prepare_process_sdk/load_sdk_assets", full_lock_passes=0, sdk_verify_assets_passes=1, inside_GPU_guard=True),
        dict(stage="prepare_overlay pre/post verify_assets", full_lock_passes=0, sdk_verify_assets_passes=2, inside_GPU_guard=True),
        dict(stage="parent successful post full_sources", full_lock_passes=1, sdk_verify_assets_passes=1, inside_GPU_guard=False)],
    reference_success_minimum_hash_accounting=dict(
        complete_lock_passes_per_job=3, complete_lock_logical_bytes_per_job=3 * total,
        complete_lock_GiB_per_job=3 * total / GIB,
        complete_lock_bytes_two_jobs=6 * total, complete_lock_GiB_two_jobs=6 * total / GIB,
        SDK_verify_assets_passes_per_job=7,
        additional_SDK_asset_hash_bytes_per_job=7 * sdk_verify_asset_hash_bytes,
        core_source_plus_SDK_hash_bytes_per_job=3 * total + 7 * sdk_verify_asset_hash_bytes,
        core_source_plus_SDK_hash_GiB_per_job=(3 * total + 7 * sdk_verify_asset_hash_bytes) / GIB,
        extra_standalone_full_preflight_adds_full_lock_pass=1,
        exclusions="Small bootstrap/helper/JSON/analyzer/YAML repeated reads, manifest metadata reads, ordinary backend/model loading and filesystem traversal. Baseline manifest validation checks row metadata; it does not recursively hash all G2/G1 payload a second time in reference verify_source_lock."),
    SDK=dict(tree_count=sum(row["file_count"] for row in tree.values()), tree_bytes=tree_total,
        trees={name: dict(count=row["file_count"], bytes=row["total_bytes"]) for name, row in tree.items()},
        external_driver_ref=inventory["driver"], cudart_ref=inventory["cudart"],
        verify_assets_hash_bytes_per_call=sdk_verify_asset_hash_bytes,
        asset_passes_inside_guard_per_job=5,
        SDK_asset_hash_bytes_inside_guard_per_job=5 * sdk_verify_asset_hash_bytes,
        failure_boundary="The external driver is not a project-relative source-lock row. SDK digest_file independently checks it; the observed error string existing file byte drift is the size gate in this helper. The raw report does not identify which specific SDK/proof/driver member failed."),
    actual_measured_preflight=dict(evidence_ref=inputs[str((BASE / "REFERENCE_SERVER_FULL_PREFLIGHT_01.json").resolve())],
        elapsed_seconds=measured["elapsed_seconds"], exit=measured["exit"], stdout=measured["stdout"],
        ledger_unchanged=measured["ledger_before_sha256"] == measured["ledger_after_sha256"],
        GPU_operations=measured["GPU_operations"], framework_imported=measured["framework_imported"],
        attribution="Entire failed preflight elapsed, not isolated model/SHA/SDK measurements. Static ordering establishes full-lock payload scan precedes the SDK size failure."),
    budget_boundary=dict(reference_child_full_source_scan_inside_300_seconds=True,
        reference_child_SDK_scans_inside_300_seconds=True,
        parent_before_and_post_scans_inside_guard=False,
        source="run_gpu_stage.py starts its monotonic clock before Popen; all child CPU source/setup, model initialization, inference and shutdown are charged to original GPU wall-time budget. Parent pre/post scans still consume user wall time.",
        G2_difference="Standalone G2 also verifies its whole lock after original model shutdown inside the child; G3 adapters invoke common helper functions, not common.main/execute_guarded, so that G2 whole-lock post pass is not additionally executed in G3."),
    safe_reduction_recommendations=[
        "Run a small source-bound driver/SDK provenance and stat-size compatibility gate before the existing expensive full_sources; keep original execution and post-source whole-file SHA verification unchanged.",
        "Do not repeatedly run an unchanged standalone complete preflight while the same explicit driver/SDK failure remains. Use exact previously frozen CPU test evidence unless implementation changes require a new test.",
        "Current frozen source must keep parent/child/post validation. Removing any of these needs a separately reviewed source change and evidence; do not bypass by caller booleans or reuse prior-process verification.",
        "For a later separately scoped optimization, consider source-bound phase timing to measure SHA vs SDK vs model setup, or an explicit process-local validation transaction for repeated small helper reads. Do not use mtime/stat-only cache, stale driver provenance, unchecked module cache or prior-run model hash."],
    remaining_uncertainty="No per-phase remote latency or storage bandwidth measured. Repeated logical reads may hit OS page cache, but SHA still processes every byte. No assertion that 96.57 seconds is purely hashing.",
    original_sources_modified=False, source_lock_modified=False,
    RPC_calls=0, GPU_calls=0, production_qualified=False, performance_claim=False)
with (HERE / "READONLY_REFERENCE_LATENCY_DIAGNOSTIC.json").open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(dict(output=str(HERE / "READONLY_REFERENCE_LATENCY_DIAGNOSTIC.json"),
    locked_payload_bytes=total, locked_payload_GiB=total/GIB,
    per_job_full_pass_bytes=3*total, SDK_verify_assets_hash_bytes=sdk_verify_asset_hash_bytes,
    core_source_plus_SDK_GiB=(3*total+7*sdk_verify_asset_hash_bytes)/GIB,
    actual_failed_preflight_seconds=measured["elapsed_seconds"], GPU_calls=0, RPC_calls=0), indent=2))
