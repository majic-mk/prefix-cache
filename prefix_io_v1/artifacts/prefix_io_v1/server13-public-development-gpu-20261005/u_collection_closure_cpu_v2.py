"""Read-only post-guard closure of actual uncalibrated original U observations.

No fresh-namespace replay, model hashing/import, calibration/table issuance or
GPU call occurs. Original guard, frontend and whole-stream capture verifiers
are loaded from their actual frozen source bytes. One new receipt is written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


SCHEMA = "actual_uncalibrated_original_U_mixed_complete_stream_post_guard_CPU_closure_v2"
VALIDATOR_SHA = "675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8"
PASS_NATIVE = "PASS_CURRENT_GPU_FORMAL_U_COLLECTION_REQUIRES_GUARD_CLOSURE"


def require(ok, reason):
    if not ok:
        raise ValueError("U_COLLECTION_CLOSURE_REJECTED: " + reason)


def safe(root, relative):
    require(type(relative) is str and relative and not relative.startswith("/") and
            ":" not in relative and "\\" not in relative and
            all(part not in ("", ".", "..") for part in relative.split("/")), "safe project-relative path")
    path = root / relative
    require(path.resolve().is_relative_to(root), "path remains within project")
    return path


def row(root, relative):
    path = safe(root, relative)
    require(path.is_file(), "actual regular evidence file")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "evidence changed while hashing")
    return dict(path=relative, bytes=after.st_size, sha256=digest.hexdigest())


def read(path, maximum=32 * 1024 ** 2):
    require(path.stat().st_size <= maximum, "bounded evidence JSON")
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("non-finite JSON number: " + value)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=invalid_constant)


def checked(root, ref):
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"} and
            row(root, ref["path"]) == ref, "actual immutable referenced bytes")
    return safe(root, ref["path"])


def load(root, ref, label):
    path = checked(root, ref)
    require(path.suffix == ".py" and ref["bytes"] <= 4 * 1024 ** 2, "bounded pure source")
    name = "_u_closure_" + label + "_" + str(time.monotonic_ns())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    raw = path.read_bytes()
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    checked(root, ref)
    return module


def validate_no_authority(native, descriptor):
    require(all(descriptor.get(name) is False for name in
                ("table_issued", "cost_qualified", "ordinary_I_authorized", "formal_goodput_allowed")),
            "descriptor cannot issue calibration/table/I/effect authority")
    require(all(native.get(name) is False for name in ("private_table_issued", "cost_qualified",
        "current_GPU_cost_table_qualified", "ordinary_I_authorized", "formal_effect_qualified",
        "formal_goodput_allowed", "strategy_improvement_proved", "production_release_qualified",
        "effect_budget_reuse_allowed")), "actual U result cannot promote collection to I/effect")
    require(native.get("current_GPU_U_functional_cost_observation_only") is True,
            "actual current-device U collection scope")


def validate_token_events(events, frontend):
    rows = frontend["rows"]
    expected = {item["request_id"]: item for item in rows}
    seen = {key: [] for key in expected}
    completed = set()
    for event in events:
        require(type(event) is dict and event.get("request_id") in expected, "only submitted actual request events")
        rid = event["request_id"]
        reference = expected[rid]
        if event.get("state") == "COMPLETED":
            require(set(event) == {"request_id", "state", "finish_ns", "total_tokens"} and rid not in completed and
                    event["finish_ns"] == reference["finished_ns"] and event["total_tokens"] == 128 and
                    len(seen[rid]) == 128, "single complete actual request event after all outputs")
            completed.add(rid)
            continue
        ordinal = len(seen[rid])
        require(set(event) == {"request_id", "native_request_id", "token_ordinal", "token_id", "return_ns", "original_step"} and
                rid not in completed and ordinal < 128 and type(event["token_ordinal"]) is int and
                event["token_ordinal"] == ordinal and event["native_request_id"] == reference["native_request_id"] and
                type(event["token_id"]) is int and type(event["return_ns"]) is int and
                event["token_id"] == reference["output_token_ids"][ordinal] and
                event["return_ns"] == reference["token_return_ns"][ordinal] and
                type(event["original_step"]) is int and event["original_step"] > 0,
                "every immutable actual token event matches complete original frontend")
        seen[rid].append(event)
    require(completed == set(expected) and all(len(value) == 128 for value in seen.values()),
            "whole five-request token stream; no dropped or selected events")
    return dict(completed_requests=len(completed), complete_token_events=sum(map(len, seen.values())))


def close(project, config_relative, output_relative):
    root = Path(project).resolve()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only CLI requires CUDA_VISIBLE_DEVICES empty")
    require(not any(name == "torch" or name == "vllm" or name.startswith(("torch.", "vllm.")) for name in sys.modules),
            "no ambient model/framework modules")
    original_sources = [row(root, config_relative)]
    config = read(safe(root, config_relative))
    require((config.get("phase"), config.get("mode"), config.get("arm")) == ("development", "off", "U") and
            "collection_ref" in config and "activation_ref" not in config, "only the actual uncalibrated U/off development config")
    require(Path(config["runner_ref"]["path"]).name == "strong_trace_runner_v7.py" and
            Path(config["runtime_ref"]["path"]).name == "native_runtime_v7.py",
            "only new V7 original U complete mixed stream consumer")
    runner = load(root, config["runner_ref"], "runner")
    refs = runner.source_rows(root, config["source_lock_ref"], full=False)
    runtime = runner.load(root, config["runtime_ref"], "_u_closure_runtime_" + str(time.monotonic_ns()))
    pair = read(checked(root, config["pair_config_ref"]))["configurations"]
    collection = runner.u_collection_gate(root, config, refs, pair)
    descriptor = collection["descriptor"]
    manifest = read(checked(root, config["workload_ref"]))
    require(manifest.get("partition_counts", {}).get("development") == 5 and
            config["formal_trace_binding_ref"] == descriptor["formal_trace_binding_ref"], "unchanged prospectively frozen five-row input")
    records = [record for record in manifest["records"] if record["split"] == "development"]
    require(len(records) == 5 and all(record["min_tokens"] == record["max_tokens"] == 128 for record in records),
            "all five predeclared development requests and all128 tokens")
    gates = dict(config=config, refs=refs, pair=pair, formal_workload_binding=dict(records=records,
        partition="development", original_manifest_workload_sha256=manifest["workload_sha256"]))
    details = config["output_relative"]
    guard_relative = str(Path(details).parent / "result.json").replace("\\", "/")
    paths = [guard_relative, details + "/strong-native-workload-result.json", details + "/actual-request-outputs.json",
             details + "/actual-original-full-step-capture.json", details + "/actual-host-control-observation.json",
             details + "/actual-native-stage-journal.json", details + "/actual-token-return-events.jsonl"]
    actual_refs = [row(root, path) for path in paths]
    guard, native, frontend, capture, host, journal = [read(checked(root, ref)) for ref in actual_refs[:6]]
    validator_relative = str(Path(config["runtime_ref"]["path"]).parent.parent / "activation/native_conditional_cost.py").replace("\\", "/")
    require(refs[validator_relative]["sha256"] == VALIDATOR_SHA, "unchanged actual guard CPU validator source")
    validator = runner.load(root, refs[validator_relative], "_u_closure_guard_" + str(time.monotonic_ns()))
    validator.validate_guard(guard, gpu_uuid=config["gpu_uuid"], job_id=config["run_id"],
                             wrapper_path=config["runner_ref"]["path"])
    require(config_relative in guard["command"] and str(root) in guard["command"],
            "original guard launched this actual config in this project")
    require(native.get("status") == PASS_NATIVE and native.get("error") is None and
            native.get("run_id") == config["run_id"] and native.get("gpu_uuid") == config["gpu_uuid"] and
            native.get("guard_reservation_id") == guard["reservation_id"] and native.get("actual_gpu_runs") == 1 and
            native.get("actual_run_config_ref") == original_sources[0], "same completed real guarded native run/config/UUID")
    for key in ("runner_ref", "runtime_ref", "source_lock_ref", "workload_ref", "pair_config_ref", "collection_ref", "formal_trace_binding_ref"):
        require(native.get(key) == config[key], "native result matches frozen config: " + key)
        checked(root, config[key])
        original_sources.append(config[key])
    validate_no_authority(native, descriptor)
    policy = native.get("strategy_runtime")
    require(type(policy) is dict and policy.get("bridge_is_none") is True and policy.get("bridge_mode") is None and
            policy.get("native_U_preserved") is True and all(policy.get(key) is False for key in
                ("real_interference_table_installed", "optional_I_controller_constructed", "actual_I_strategy_activated",
                 "real_private_cost_table_reissued_for_identity")), "actual original U path with no I controller/private table")
    require(frontend == native.get("frontend") and capture == native.get("full_original_step_capture") and
            journal == native.get("actual_native_stage_journal") and native.get("actual_request_outputs_ref") == actual_refs[2] and
            native.get("actual_original_full_step_capture_ref") == actual_refs[3], "complete immutable raw native evidence matches embedded actual result")
    frontend_closure = runner.validate_formal_frontend_records(gates, frontend)
    require(frontend_closure == native.get("formal_frontend_identity_closure"), "same original exact frontend identity closure")
    require(native.get("complete_stream_observation_valid") is True and native.get("mixed_frames_preserved") is True and
            native.get("full_step_evidence_qualified") is False and
            native.get("exact_finite_cost_cells_qualified") is False,
            "new U preserves every mixed frame without issuing finite-cell cost qualification")
    observation_config = native.get("complete_observation_config")
    require(type(observation_config) is dict and observation_config.get("source_ref") == descriptor["collector_ref"] and
            observation_config.get("common_U_I_observation") is False and
            observation_config.get("uncalibrated_U_full_stream_observation_only") is True,
            "new mixed stream is only U collection; no unchanged common U/I observation qualification")
    frames = capture.get("frames")
    require(type(frames) is list and frames, "all original native stream frames present")
    ordinals = [frame.get("native_step_ordinal") for frame in frames]
    require(all(type(value) is int and value >= 0 for value in ordinals) and
            ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals))),
            "all contiguous original native steps; no filtered mixed/empty frames")
    capture_closure = runner.validate_u_collection_capture(root, gates, capture, frontend,
        source_ref=descriptor["collector_ref"], expected_ordinals=ordinals)
    require(type(capture_closure) is dict and capture_closure == native.get("formal_complete_native_observation"),
            "same immutable complete mixed CUDA/native/frontend observation replay")
    require(capture_closure.get("schema") == "complete_heterogeneous_original_U_stream_observation_v1" and
            capture_closure.get("full_native_steps") == len(frames) and capture_closure.get("complete128_requests") == 5 and
            capture_closure.get("collector_source_ref") == descriptor["collector_ref"] and
            capture_closure.get("mixed_frames_preserved") is True and
            capture_closure.get("complete_stream_observation_valid") is True and
            all(capture_closure.get(name) is False for name in
                ("full_step_evidence_qualified", "exact_finite_cost_cells_qualified", "table_issued", "ordinary_I_authorized")),
            "complete heterogeneous U stream receipt preserves its denominator without issuing an exact-cell table")
    require(native.get("original_engine_shutdown_returned") is True and native.get("native_tail_drained") is True and
            runtime.verify_original_tail(native.get("post_original_shutdown")), "original engine shutdown and actual native/AIO tail drained")
    for name in ("before_workload_native_drain", "after_workload_native_drain", "final_before_shutdown_drain"):
        value = native.get(name)
        require(type(value) is dict, "actual original drain evidence: " + name)
        runtime.validate_drained_snapshot(value["owner_snapshot"])
    require(native.get("host_boundary_wrappers_restored") is True and native.get("optional_probe_restored") is True and
            native.get("immutable_runner_config_workload_unchanged") is True,
            "actual optional host/capability observer wrappers restored and original runtime inputs unchanged")
    require(journal.get("run_id") == config["run_id"] and journal.get("valid") is True and journal.get("lost") == 0,
            "actual bounded native journal retained without lost observations")
    host_relative = str(Path(descriptor["reserve_join_source_ref"]["path"]).with_name("host_control_observer.py")).replace("\\", "/")
    host_module = runner.load(root, refs[host_relative], "_u_closure_host_" + str(time.monotonic_ns()))
    host_closure = host_module.checked_intervals(host, require_production_clock=True)
    require(host.get("run_id") == config["run_id"] and
            host_closure["ordinals"] == [frame["native_step_ordinal"] for frame in capture["frames"]],
            "actual complete host control observation aligned to every native CUDA frame")
    normalized_host = runner.project_ref(root, native["host_control_observation_ref"])
    require(normalized_host == actual_refs[4], "same raw host observation bytes")
    token_path = checked(root, actual_refs[-1])
    require(token_path.stat().st_size <= 10 * 1024 ** 2, "bounded actual token event stream")
    events = [json.loads(line) for line in token_path.read_text(encoding="utf-8").splitlines()]
    token_closure = validate_token_events(events, frontend)
    require(token_closure == dict(completed_requests=5, complete_token_events=640), "exact complete development denominator")
    witnesses = capture["event_witnesses"]
    gpu_ns = [item["gpu_elapsed_ns"] for item in witnesses]
    itl_ns = [value for item in frontend["rows"] for value in item["itl_ns"]]
    for ref in original_sources + actual_refs:
        checked(root, ref)
    require(not any(name == "torch" or name == "vllm" or name.startswith(("torch.", "vllm.")) for name in sys.modules),
            "closure imported no framework/model")
    receipt = dict(schema=SCHEMA, status="PASS_ACTUAL_U_FUNCTION_AND_COMPLETE_MIXED_STREAM_OBSERVATION_ONLY",
        run_id=config["run_id"], gpu_uuid=config["gpu_uuid"], partition="development", completed_guard_ref=actual_refs[0],
        actual_native_result_ref=actual_refs[1], actual_run_config_ref=original_sources[0], actual_raw_source_refs=actual_refs[2:],
        closure_source_ref=row(root, Path(__file__).resolve().relative_to(root).as_posix()),
        original_guard_natural_session_drained=True, original_engine_shutdown_returned=True,
        actual_native_tail_drained=True, original_U_preserved=True, optional_observers_restored=True,
        complete_frontend_identity=frontend_closure, whole_native_capture_replay=capture_closure,
        complete_native_stream_steps=len(frames),
        complete_token_stream=token_closure, complete_host_control_steps=len(host_closure["ordinals"]),
        immutable_raw_evidence_revalidated=True, completed_actual_GPU_jobs=1, CPU_closure_GPU_operations=0,
        descriptive_observation=dict(full_native_steps=len(witnesses),
            CUDA_complete_original_step_duration_ms=dict(min=min(gpu_ns)/1e6, median=statistics.median(gpu_ns)/1e6, max=max(gpu_ns)/1e6),
            original_token_return_interval_ms=dict(min=min(itl_ns)/1e6, median=statistics.median(itl_ns)/1e6, max=max(itl_ns)/1e6),
            guard_elapsed_seconds=guard.get("elapsed_seconds")),
        function_verified=True, current_GPU_full_stream_cost_observation_verified=True,
        complete_stream_observation_valid=True, mixed_frames_preserved=True,
        full_step_evidence_qualified=False, exact_finite_cost_cells_qualified=False,
        mixed_frames_assigned_to_single_exact_cell=False,
        common_U_I_observation_qualified=False, uncalibrated_U_full_stream_observation_only=True,
        cost_qualified=False, private_table_issued=False, table_issued=False, ordinary_I_authorized=False,
        independent_service_SLO_supplied=False, formal_goodput_allowed=False, strategy_improvement_proved=False,
        production_release_qualified=False, source_lock_whole_model_rehash_performed=False,
        inherited_compile_proof_is_current_driver_qualification=False)
    output = safe(root, output_relative)
    require(output.parent.is_dir() and not output.exists(), "append-only receipt in existing directory")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def functional_diagnostic(project, config_relative, output_relative):
    """Explain a failed guard job with complete genuine outputs, without passing it."""
    root = Path(project).resolve()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only diagnostic requires CUDA_VISIBLE_DEVICES empty")
    config_ref = row(root, config_relative)
    config = read(checked(root, config_ref))
    require((config.get("phase"), config.get("mode"), config.get("arm")) == ("development", "off", "U") and
            "collection_ref" in config and "activation_ref" not in config, "only uncalibrated actual U diagnostic")
    runner = load(root, config["runner_ref"], "diagnostic_runner")
    runtime = runner.load(root, config["runtime_ref"], "_u_diagnostic_runtime_" + str(time.monotonic_ns()))
    descriptor = read(checked(root, config["collection_ref"]))
    manifest = read(checked(root, config["workload_ref"]))
    records = [record for record in manifest["records"] if record["split"] == "development"]
    require(len(records) == 5 and manifest["partition_counts"]["development"] == 5,
            "whole original development partition")
    gates = dict(config=config, formal_workload_binding=dict(records=records, partition="development",
        original_manifest_workload_sha256=manifest["workload_sha256"]))
    details = config["output_relative"]
    paths = [str(Path(details).parent / "result.json").replace("\\", "/"),
        details + "/strong-native-workload-result.json", details + "/actual-request-outputs.json",
        details + "/actual-original-full-step-capture.json", details + "/actual-token-return-events.jsonl"]
    actual_refs = [row(root, value) for value in paths]
    guard, native, frontend, capture = [read(checked(root, ref)) for ref in actual_refs[:4]]
    require(guard.get("label") == native.get("run_id") == config["run_id"] and
            guard.get("gpu_uuid") == native.get("gpu_uuid") == config["gpu_uuid"] and
            guard.get("gpu_job_attempted") is True and guard.get("exit") == 1 and guard.get("child_exit") == 1 and
            guard.get("timed_out") is False and guard.get("interrupted_signal") is None and guard.get("error") is None and
            guard.get("session_drained") is True and guard.get("session_members_before_cleanup") == [] and
            guard.get("session_members_after_cleanup") == [] and
            guard.get("reservation_id") == native.get("guard_reservation_id"),
            "actual failed job with natural session drain; cannot label original guard PASS")
    require(native.get("actual_run_config_ref") == config_ref and native.get("status") == "FAILED_STRONG_NATIVE_WORKLOAD" and
            type(native.get("error")) is dict and capture.get("valid") is False,
            "actual failed whole-stream capture lifecycle, not a successful cost closure")
    for name in ("runner_ref", "runtime_ref", "source_lock_ref", "workload_ref", "pair_config_ref", "collection_ref", "formal_trace_binding_ref"):
        require(native.get(name) == config[name], "actual diagnostic immutable binding: " + name)
        checked(root, config[name])
    validate_no_authority(native, descriptor)
    require(frontend == native.get("frontend") and capture == native.get("full_original_step_capture") and
            native.get("actual_request_outputs_ref") == actual_refs[2] and
            native.get("actual_original_full_step_capture_ref") == actual_refs[3], "actual complete raw outputs/capture bytes")
    frontend_closure = runner.validate_formal_frontend_records(gates, frontend)
    require(frontend_closure == native.get("formal_frontend_identity_closure") and
            native.get("original_engine_shutdown_returned") is True and native.get("native_tail_drained") is True and
            runtime.verify_original_tail(native.get("post_original_shutdown")), "genuine completed outputs and original successful shutdown tail")
    for name in ("before_workload_native_drain", "final_before_shutdown_drain"):
        runtime.validate_drained_snapshot(native[name]["owner_snapshot"])
    require(native.get("optional_probe_restored") is True and native.get("host_boundary_wrappers_restored") is True and
            native.get("immutable_runner_config_workload_unchanged") is True, "optional observers restored after failed job")
    policy = native["strategy_runtime"]
    require(policy.get("bridge_is_none") is True and policy.get("native_U_preserved") is True and
            policy.get("optional_I_controller_constructed") is False, "actual original U model path preserved")
    events = [json.loads(value) for value in checked(root, actual_refs[-1]).read_text(encoding="utf-8").splitlines()]
    token_closure = validate_token_events(events, frontend)
    require(token_closure == dict(completed_requests=5, complete_token_events=640), "complete actual output denominator")
    for ref in actual_refs + [config_ref]:
        checked(root, ref)
    require(not any(name == "torch" or name == "vllm" or name.startswith(("torch.", "vllm.")) for name in sys.modules),
            "diagnostic imported no framework/model")
    receipt = dict(schema="actual_failed_U_capture_complete_outputs_CPU_diagnostic_v1",
        status="PARTIAL_ACTUAL_OUTPUTS_AND_SHUTDOWN_CONFIRMED_CAPTURE_AND_JOB_FAILED",
        run_id=config["run_id"], gpu_uuid=config["gpu_uuid"], actual_run_config_ref=config_ref,
        actual_raw_source_refs=actual_refs, complete_frontend_identity=frontend_closure,
        complete_token_stream=token_closure, original_U_preserved=True, actual_outputs_complete=True,
        original_engine_shutdown_returned=True, actual_native_tail_drained=True, natural_OS_session_drained=True,
        original_guard_passed=False, complete_GPU_job_passed=False, current_GPU_full_step_cost_observation_verified=False,
        actual_capture_valid=False, actual_failure=native["error"], actual_capture_failures=capture.get("failures"),
        CPU_diagnostic_GPU_operations=0, cost_qualified=False, table_issued=False, private_table_issued=False,
        ordinary_I_authorized=False, formal_goodput_allowed=False, strategy_improvement_proved=False,
        production_release_qualified=False)
    output = safe(root, output_relative)
    require(output.parent.is_dir() and not output.exists(), "new append-only honest diagnostic receipt")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--functional-diagnostic", action="store_true",
                        help="Only diagnose complete actual outputs/tail of a failed capture job; never close guard or costs")
    args = parser.parse_args(argv)
    result = (functional_diagnostic(args.project, args.config, args.output) if args.functional_diagnostic
              else close(args.project, args.config, args.output))
    print(json.dumps(dict(status=result["status"], run_id=result["run_id"],
        completed_requests=result["complete_token_stream"]["completed_requests"],
        complete_output_tokens=result["complete_token_stream"]["complete_token_events"],
        native_steps=result.get("complete_native_stream_steps"),
        cost_qualified=False, table_issued=False, ordinary_I_authorized=False), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
