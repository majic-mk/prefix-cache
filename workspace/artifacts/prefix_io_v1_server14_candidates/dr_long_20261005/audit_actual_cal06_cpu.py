"""One-off local audit of downloaded CAL06, never an old-P316 qualification."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("_dr_local_analysis", HERE / "analyze_dr_long_results.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
run = HERE / "actual_server14/runs/server14-dr-cal06"
paths = dict(config=HERE / "actual_server14/CONFIG_CAL06.json",
    source_lock=HERE / "actual_server14/DR_SOURCE_LOCK_05.json",
    result=run / "details/result.json", adapter=run / "current-device-adapter-evidence.json",
    guard=run / "result.json", frozen_config=run / "details/frozen-config.json")
docs, files = {}, {}
for key, path in paths.items():
    docs[key], files[key] = M.read_leaf(PROJECT, str(path))
proof, proof_ref = M.read_leaf(PROJECT, str(HERE / "MONOTONIC_TOKEN_CLOCK_SOURCE_01.json"))
try:
    M.validate_run(dict(label="CAL06", arm="CAL"), docs, files,
        expected_gpu_uuid="GPU-cb140ee8-a52e-93ba-38f7-e2fddfbb43b3",
        expected_common_domain_sha256="e75c3adff856be6dd9993dc1990a4709b5ea00971a35557bdf5731ee377fcd75",
        clock_proof=proof)
except ValueError as exc:
    M.require(str(exc) == "frozen original input/arrival/cache/source contract", "unexpected prior metadata/lifecycle failure")
    old_input_rejection = str(exc)
else:
    raise ValueError("single-family CAL06 cannot pass the old three-family input contract")
raw, frozen, config = docs["result"], docs["frozen_config"], docs["config"]
manifest = frozen["manifest"]
M.require(manifest["scope"] == "CAL_DEV_SINGLE_FAMILY" and manifest["profile"] == "all_hit" and
    len(manifest["requests"]) == 10 and all(r["family_index"] == 0 for r in manifest["requests"]) and
    manifest["output_tokens"] == 128 and manifest["cache_resets"] == raw["cache_resets"] == 0 and
    manifest["artificial_io_delay"] is False and manifest["slo"] is None and
    raw["source_preservation"] == dict(checked_files=3048, changed=[]), "independent single-family CAL contract")
fixed = config["engine_controls"]["prefix_io_p4_policy"]["fixed_stage_policy"]
M.validate_original_fixed8(fixed, config["run_id"])
locked = {r["path"]:r for r in docs["source_lock"]["files"]}
clock_verified = (all(locked.get(r["source_ref"]["path"]) == r["source_ref"] for r in proof["source_rows"]) and
    frozen["engine"]["distributed_executor_backend"] == "uni")
metrics, outputs = M.request_metrics(raw, token_clock_verified=clock_verified)
by_id = {r["request_id"]:r for r in raw["rows"]}
for request in manifest["requests"]:
    row = by_id["c2-" + str(request["request_id"])]; family = manifest["families"][0]
    M.require(row["scheduled_arrival_seconds"] == request["scheduled_time"] and row["family"] == family["name"] and
        row["initial_ssd_present"] is family["initial_ssd_present"] and len(family["tokens"]) == 16257,
        "actual scheduled input differs from independent frozen single family")
stage = M.native_stage_delta(raw)
M.require(raw["cohort_read_bytes"] == stage["ssd_read"]["accepted_bytes"] and
    raw["cohort_write_bytes"] == stage["ssd_write"]["accepted_bytes"], "actual CAL byte accounting mismatch")
history = raw["probe"]["current_native_snapshots"][0]["p4"]["eta_history"]
samples = history["recent_measurements"]
M.require(history["minimum_samples"] == 4 and history["successful_measurements"] == len(samples) == 1 and
    history["omissions"] == 0 and history["pending_count"] == 0 and history["failed_or_unknown_discarded"] == 0,
    "actual bounded history differs from one retained closure")
sample = samples[0]
M.require(sample["geometry"]["rows"] == [["h2d", 917504, 917504, 6]] and
    sample["completed_ns"] - sample["first_observed_ns"] == sample["elapsed_ns"], "actual closure geometry/time mismatch")
loader, loader_ref = M.read_leaf(PROJECT, str(HERE / "actual_server14/ACTUAL_CAL06_LOADER_CPU_RESULT_01.json"))
for name in ("guard", "result"):
    ref = loader["calibration_" + name + "_ref"]
    M.require(ref["bytes"] == files[name]["bytes"] and ref["sha256"] == files[name]["sha256"], "loader actual CAL byte binding")
M.require(loader["sample_count"] == 1 and loader["status"] == "VALID_CLOSED_BUT_COVERAGE_MISSING" and
    loader["exact_cell_counts"] == [[sample["geometry"]["rows"], sample["context"], 1]], "actual exact cell coverage mismatch")
report = dict(schema="independent_CAL06_actual_CPU_audit_v1", classification="INDEPENDENT_SINGLE_FAMILY_DEVELOPMENT_CALIBRATION",
    old_three_family_input_contract_rejection=old_input_rejection, old_P316_qualification_claim=False,
    effect_sample=False, GPU_operations_this_audit=0, SSH_operations=0, evidence_files=files,
    clock_proof_ref=proof_ref, loader_result_ref=loader_ref, actual_full_requests=len(outputs), actual_generated_tokens=1280,
    primary_response_seconds=raw["response_seconds"], cohort_seconds_including_drain=raw["cohort_seconds_including_drain"],
    tail_drain_seconds=raw["tail_drain_seconds"], native_stage_deltas=stage, requests=metrics,
    token_clock_verified=clock_verified, actual_closed_lifecycle_prior_checks_passed=True,
    actual_eta_history=history, exact_cell_minimum_samples=4, qualifying_pure_SSD_exact_cells=0,
    calibration_coverage_status=loader["status"], production_prediction_qualified=False)
target = HERE / "ACTUAL_INDEPENDENT_CAL06_CPU_AUDIT_01.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(dict(classification=report["classification"], actual_generated_tokens=1280,
    primary_response_seconds=report["primary_response_seconds"], status=loader["status"], GPU_operations=0)))
