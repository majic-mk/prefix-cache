"""Close actual original guarded A/B windows; descriptive raw costs only.

No framework, GPU query, guard launch, fitting, table or strategy activation.
A failed guard is rejected even when its raw capture is separately inspectable.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

P = "artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004"
RUNNER = dict(path=P+"/raw_unit_cost/natural_unit_raw_runner_v2.py", bytes=28974,
    sha256="2f4aae0a1c1650bfd0f83357ddb0d3fbd2c590391c0a864255e572da272c0052")
CLOSER = dict(path=P+"/formal_runtime_bridge/close_formal_peer_cpu.py", bytes=15177,
    sha256="cf16bc3494757c44367ace71858c4f2d9ffabd4d74cbdecff03369bbeb44d25a")


def require(ok, reason):
    if not ok:
        raise ValueError("RAW_AB_CPU_CLOSURE_REJECTED: "+reason)


def cpu_only():
    require(not any(k == "torch" or k.startswith(("torch.", "vllm", "py_kvcache"))
                    for k in sys.modules), "pure CPU command must not import a framework")


def bootstrap(root):
    path = root / RUNNER["path"]
    raw = path.read_bytes()
    require(len(raw) == RUNNER["bytes"] and hashlib.sha256(raw).hexdigest() == RUNNER["sha256"],
            "exact frozen V2 runner source")
    spec = importlib.util.spec_from_file_location("_close_raw_runner_"+str(time.monotonic_ns()), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    return module, module.driver(root)


def document(root, relative, R, used):
    row = R.ref(root, relative)
    used[row["path"]] = row
    return R.read(R.check_ref(root, row))


def arm(root, relative, guard_relative, *, M, R, ledger, used):
    config = document(root, relative, R, used)
    require(type(config) is dict and set(config) == M.CONFIG_FIELDS and
            config.get("schema") == "current_gpu_natural_calibration_raw_unit_config_v1" and
            config.get("runner_ref") == RUNNER and type(config.get("window_index")) is int and
            config["window_index"] in (0, 1), "actual same frozen V2 raw-only configuration")
    for key in M.CONFIG_FIELDS - {"schema", "job_id", "gpu_uuid", "seconds_limit", "window_index",
                                  "output_relative", "storage_template_relative"}:
        R.check_ref(root, config[key]); used[config[key]["path"]] = config[key]
    refs = R.source_rows(root, config["source_lock_ref"], full=False)
    require(refs.get(RUNNER["path"]) == RUNNER and refs.get(CLOSER["path"]) == CLOSER,
            "closed runner and original CPU session/ledger helpers")
    M.verify_sources(root, config, refs, R.read(R.check_ref(root, config["source_proof_ref"])), R=R)
    plan = R.read(R.check_ref(root, config["plan_ref"]))
    record = M.selected_natural_record(R.read(R.check_ref(root, M.NATURAL)))
    require(plan.get("schema") == "current_gpu_natural_calibration_raw_unit_plan_v1" and
            plan.get("job_id") == plan.get("journal_run_id") == config["job_id"] and
            plan.get("gpu_uuid") == config["gpu_uuid"] and
            plan.get("source_lock_ref") == config["source_lock_ref"] and plan.get("wrapper_source_ref") == RUNNER and
            plan.get("natural_source_record") == record and plan.get("natural_manifest_ref") == M.NATURAL and
            plan.get("cells") == [M.cell_descriptor(record)] and
            all(plan.get(k) is False for k in M.FALSE_FIELDS) and plan.get("natural_trace") is False and
            plan.get("independent_family_holdout_qualified") is False,
            "unchanged185/176/offset16/context200/917504 calibration-only controlled geometry")
    for key, row in plan.items():
        if key.endswith("_ref") and type(row) is dict:
            R.check_ref(root, row); used[row["path"]] = row
    pair = R.read(R.check_ref(root, plan["runtime_pair_ref"]))["configurations"]
    require(R.common_domain_sha(pair) == plan["common_runtime_domain_sha256"], "actual same native/model domain")
    guard = document(root, guard_relative, R, used)
    validation = R.load(root, plan["validation_source_ref"], "_raw_AB_validate_"+str(time.monotonic_ns()))
    validation.validate_guard(guard, gpu_uuid=config["gpu_uuid"], job_id=config["job_id"], wrapper_path=RUNNER["path"])
    matches = [event for event in ledger["events"] if event.get("reservation_id") == guard["reservation_id"]]
    require(len(matches) == 1 and matches[0] == guard, "unique actual completed guard in unchanged budget ledger")
    summary = document(root, config["output_relative"]+"/natural-unit-raw-window-result.json", R, used)
    condition = ("A", "B")[config["window_index"]]
    require(summary.get("schema") == "actual_current_gpu_natural_unit_raw_window_v1" and
            summary.get("status") == "PASS_ACTUAL_NATURAL_UNIT_RAW_WINDOW_REQUIRES_PARENT_AB_COMPARISON" and
            summary.get("job_id") == config["job_id"] and summary.get("gpu_uuid") == config["gpu_uuid"] and
            summary.get("condition") == condition and summary.get("plan_ref") == config["plan_ref"] and
            summary.get("runner_ref") == RUNNER and summary.get("estimator_run") is False and
            summary.get("h2d_claim") is False and summary.get("source_whole_model_rehash_performed") is False and
            all(summary.get(k) is False for k in M.FALSE_FIELDS), "actual finished raw-only summary")
    base = config["output_relative"]+"/windows/"+("%02d" % config["window_index"])+"/details"
    report = document(root, base+"/native-cost-runtime-result.json", R, used)
    require(report == summary["original_raw_result"] and
            report.get("status") == "PASS_NATIVE_FULL_STEP_RAW_CAPTURE_REQUIRES_VALIDATION" and
            report.get("origin") == "native_gpu_recording" and report.get("label") == config["job_id"] and
            report.get("gpu_uuid") == config["gpu_uuid"] and report.get("source_lock") == config["source_lock_ref"] and
            report.get("site_binding_ref") == config["plan_ref"] and report.get("window_index") == config["window_index"] and
            report.get("original_engine_shutdown_returned") is True and report.get("optional_probe_restored") is True and
            report.get("fresh_original_process") is True and report.get("cache_reset_count") == 0 and len(report["windows"]) == 1,
            "unchanged actual full original finally report and shutdown")
    active = report["guard"]
    require(active.get("id") == guard["reservation_id"] and active.get("label") == guard["label"] == config["job_id"] and
            active.get("gpu_uuid") == guard["gpu_uuid"] == config["gpu_uuid"] and
            active.get("session_id") == active.get("process_group") == report.get("subprocess_sid") == guard["session_id"] and
            active.get("permissions") == guard.get("permissions") == config["permissions_ref"] and
            active.get("command") == guard.get("command") ==
            [".venv/bin/python", "-B", RUNNER["path"], "--project", str(root), "--config", relative, "--execute"],
            "actual ended guard joins this original child/config/source")
    window = report["windows"][0]; rid = window["request_id"]
    require(document(root, base+"/"+rid+"-window.json", R, used) == window and
            document(root, base+"/"+rid+"-capture.json", R, used) == window["capture"] and
            document(root, base+"/"+rid+"-frontend.json", R, used) == window["frontend"] and
            document(root, base+"/native-journal.json", R, used) == report["native_journal"],
            "all actual separate raw bytes equal the original report")
    joined = dict(window)
    for key in ("native_journal", "native_post_shutdown", "native_tail_assertions", "subprocess_pid", "private_storage"):
        joined[key] = report[key]
    joined["child_receipt_ref"] = used[base+"/native-cost-runtime-result.json"]
    gates = dict(config=config, refs=refs, plan=plan, pair=pair)
    S, old, compat, parameterization = M.configure_window(root, gates, active, relative, R=R)
    require(old._strict_completed_window(joined, config["window_index"]) is True,
            "original complete-window CUDA/output/native IO/shutdown validation")
    require(parameterization == summary["source_parameterization"] and parameterization.get("six_fresh_processes") is False and
            parameterization.get("fit_pairs") == parameterization.get("independent_validation_pairs") == 0,
            "same actual single-window raw parameterization, no fit/holdout alias")
    output = window["frontend"]["output"]
    frames = validation.validate_capture(window["capture"], run_id=rid, request_id=output["native_request_id"],
        output_ids=output["output_token_ids"], prompt_tokens=185, measured_offset=16, warmup_offsets=[1], cached_tokens=176)
    drain = validation.original_post_shutdown_drain(report["native_post_shutdown"], report["native_tail_assertions"],
        run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_ref"]["sha256"])
    io = validation.validate_io(report["native_journal"], drain, capture=window["capture"], frames=frames,
        run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_ref"]["sha256"],
        arm="baseline" if condition == "A" else "action", measured_offset=16, physical_bytes=917504, operations=1,
        independent_payload=window["independent_payload"])
    measured = [frame for frame in frames if frame["step_offset"] == 16]
    warmup = [frame for frame in frames if frame["step_offset"] == 1]
    require(len(measured) == len(warmup) == 1 and
            [f for f in window["capture"]["frames"] if f["native_step_ordinal"] == measured[0]["native_step_ordinal"]][0]
            ["prepared"]["context_length"] == 200 and output["num_cached_tokens"] == 176 and
            io["accepted_operations"] == (1 if condition == "B" else 0) and
            io["accepted_physical_bytes"] == (917504 if condition == "B" else 0), "actual selected context and SSD-only unit")
    durations = [frame["gpu_elapsed_ns"] for frame in frames]
    description = dict(condition=condition, job_id=config["job_id"], gpu_uuid=config["gpu_uuid"], request_id=rid,
        complete_output_tokens=128, complete_CUDA_steps=len(frames), measured_offset=16, measured_context_length=200,
        measured_native_step_ordinal=measured[0]["native_step_ordinal"], measured_gpu_elapsed_ns=measured[0]["gpu_elapsed_ns"],
        warmup_offset1_gpu_elapsed_ns=warmup[0]["gpu_elapsed_ns"],
        full128_gpu_elapsed_ns=dict(min=min(durations), median=statistics.median(durations), max=max(durations), sum=sum(durations)),
        native_io=io, original_engine_shutdown_returned=True, actual_native_tail_drained=True,
        session_id=guard["session_id"], private_storage=report["private_storage"], guard_elapsed_seconds=guard["elapsed_seconds"],
        current_whole_model_rehash_performed=False)
    return config, plan, output["output_token_ids"], description, guard


def close(root, *, a_config, b_config, a_guard, b_guard, output_relative):
    cpu_only(); M, R = bootstrap(root)
    helper = R.load(root, CLOSER, "_raw_AB_CPU_guard_helper_"+str(time.monotonic_ns()))
    used = {RUNNER["path"]: RUNNER, CLOSER["path"]: CLOSER}
    with helper.original_budget_read_lock(root, driver=R) as ledger_path:
        ledger_raw = ledger_path.read_bytes(); ledger = json.loads(ledger_raw)
        require(type(ledger) is dict and ledger.get("active_reservation") is None and type(ledger.get("events")) is list and
                type(ledger.get("gpu_wall_seconds")) in (int, float) and math.isfinite(ledger["gpu_wall_seconds"]) and
                0 <= ledger["gpu_wall_seconds"] <= R.MAX_GPU_SECONDS, "actual idle original authorized budget")
        a = arm(root, a_config, a_guard, M=M, R=R, ledger=ledger, used=used)
        b = arm(root, b_config, b_guard, M=M, R=R, ledger=ledger, used=used)
        require((a[0]["window_index"], b[0]["window_index"]) == (0, 1) and a[0]["job_id"] != b[0]["job_id"] and
                a[0]["gpu_uuid"] == b[0]["gpu_uuid"] and a[0]["source_lock_ref"] == b[0]["source_lock_ref"] and
                a[1]["cells"] == b[1]["cells"] and a[1]["common_runtime_domain_sha256"] == b[1]["common_runtime_domain_sha256"] and
                a[1]["natural_manifest_ref"] == b[1]["natural_manifest_ref"] and a[2] == b[2] and
                a[3]["private_storage"] != b[3]["private_storage"] and a[4]["session_id"] != b[4]["session_id"] and
                a[4]["started_unix"]+a[4]["elapsed_seconds"] <= b[4]["started_unix"],
                "actual sequential AB fresh processes/namespaces and identical complete128 outputs")
        sessions = [a[4]["session_id"], b[4]["session_id"]]
        require(all(helper.session_members(sid) == [] for sid in sessions), "actual original OS sessions empty")
        target = R.safe(root, output_relative)
        require(output_relative.endswith(".json") and target.parent.is_dir() and not target.exists(), "fresh append-only output")
        tool_ref = R.ref(root, Path(__file__).resolve().relative_to(root).as_posix())
        measured_a, measured_b = a[3]["measured_gpu_elapsed_ns"], b[3]["measured_gpu_elapsed_ns"]
        result = dict(schema="actual_current_GPU_natural_calibration_raw_AB_CPU_closure_v1",
            status="PASS_ACTUAL_RAW_AB_FUNCTION_NATIVE_IO_AND_DESCRIPTIVE_COST_ONLY", GPU_operations_this_CPU_command=0,
            real_GPU_job_count=2, complete_output_equality=True, actual_original_guards_passed=True,
            original_native_tail_revalidated=True, source_and_input_evidence_refs=[used[path] for path in sorted(used)],
            closure_tool_ref=tool_ref, ledger_snapshot_sha256=hashlib.sha256(ledger_raw).hexdigest(),
            actual_gpu_wall_seconds=ledger["gpu_wall_seconds"], arms=[a[3], b[3]],
            single_pair_selected_step_difference_ns=measured_b-measured_a, single_pair_selected_step_ratio_B_over_A=measured_b/measured_a,
            analysis_scope="one controlled calibration request, one sequential AB pair, SSD-only preload; descriptive raw measurements",
            repetitions=1, independent_family_holdout_qualified=False, service_SLO=None, statistical_significance_claim=False,
            strategy_improvement_proved=False, estimator_run=False, table_issued=False, cost_qualified=False,
            ordinary_I_authorized=False, formal_goodput_allowed=False, H2D_or_copy_measured=False,
            resource_release_credit=False, current_whole_model_rehash_performed=False, failed_earlier_guards_reclassified=False)
        for row in used.values():
            R.check_ref(root, row)
        cpu_only()
        require(all(helper.session_members(sid) == [] for sid in sessions) and ledger_path.read_bytes() == ledger_raw,
                "sessions/ledger unchanged after CPU verification")
        R.new_json(target, result)
        return R.ref(root, output_relative)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    for name in ("a-config", "b-config", "a-guard", "b-guard", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    try:
        ref = close(args.project.resolve(strict=True), a_config=args.a_config, b_config=args.b_config,
                    a_guard=args.a_guard, b_guard=args.b_guard, output_relative=args.output)
        print(json.dumps(dict(status="PASS_ACTUAL_RAW_AB_CPU_CLOSURE", output_ref=ref, table_issued=False,
                              ordinary_I_authorized=False, GPU_operations_this_CPU_command=0), sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status="BLOCKED_RAW_AB_CPU_CLOSURE", reason=str(exc), failed_guards_reclassified=False,
                              table_issued=False, GPU_operations_this_CPU_command=0)))
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
