"""Pure read-only analysis of one actual bounded U/I pair, not qualification.

Times follow the frozen original drive: scheduled arrival -> original step
return. Native stage counts/host overlap are descriptive; they do not invent
per-request SSD restore identities, CUDA absolute clocks, or resource credit.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

DRIVE = "artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_trace_runner_v7.py"
DRIVE_SHA = "0d916dedd52c2aaec3822adff6560987d443425619f96d57a027a60be10fe9dd"
STAGES = ("ssd_read", "ssd_write", "h2d", "d2h")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def integer(value, reason, minimum=0):
    require(type(value) is int and value >= minimum, reason)
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(root, name):
    path = Path(name)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "actual file outside project/missing")
    raw = path.read_bytes()
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate actual JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite actual JSON")))
    return value, dict(path=path.relative_to(root).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def check_ref(root, row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "actual strict reference")
    value, actual = read(root, row["path"])
    require(actual == row, "actual referenced bytes changed")
    return value


def summary(values):
    require(values and all(type(v) is int and v >= 0 for v in values), "actual nonempty scalar measurements")
    ordered = sorted(values)
    return dict(n=len(values), min_ns=ordered[0], median_ns=statistics.median(ordered),
        mean_ns=sum(ordered) / len(ordered), max_ns=ordered[-1],
        descriptive_p95_ns=ordered[max(0, math.ceil(.95 * len(ordered)) - 1)])


def frontend(result, inputs):
    front = result["frontend"]
    records, rows = inputs["records"], front["rows"]
    require(front["status"] == "PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS" and
        front["actual_request_stream_completed"] is True and front["planned_requests"] == 12 and
        front["successful_requests"] == 12 and front["failure_or_unsubmitted_requests"] == 0 and
        front["measurement_clock"] == "monotonic_ns immediately after original engine.step" and
        front["kernel_cost_measurement"] is False and len(records) == len(rows) == 12,
        "original complete twelve-request frontend fields")
    arrivals, queue, ttft, latencies, gaps, details, outputs = [], [], [], [], [], [], []
    for record, row in zip(records, rows):
        require(row["request_id"] == str(record["request_id"]) and row["state"] == "COMPLETED" and
            row["error"] is None and type(row["native_request_id"]) is str and row["native_request_id"] and
            row["actual_prompt_token_ids"] == record["prompt_token_ids"] and
            row["prompt_sha256"] == record["prompt_sha256"] and record["max_tokens"] == 128,
            "actual request ID/order/prompt identity/output contract")
        tokens, times, itl = row["output_token_ids"], row["token_return_ns"], row["itl_ns"]
        require(type(tokens) is list and len(tokens) == 128 and all(type(t) is int and t >= 0 for t in tokens) and
            type(times) is list and len(times) == 128 and type(itl) is list and len(itl) == 127,
            "all actual token IDs and return timestamps required")
        scheduled = integer(row["scheduled_ns"], "actual scheduled_ns", 1)
        submitted = integer(row["submission_ns"], "actual submission_ns", 1)
        finished = integer(row["finished_ns"], "actual finished_ns", 1)
        require(scheduled <= submitted <= times[0] <= times[-1] <= finished and
            all(type(t) is int and t > 0 for t in times) and
            itl == [right - left for left, right in zip(times, times[1:])] and all(type(v) is int and v >= 0 for v in itl) and
            row["ttft_ns"] == times[0] - scheduled and row["latency_ns"] == finished - scheduled,
            "original arrival/submit/token/finish clock arithmetic")
        integer(row["num_cached_tokens"], "actual cached-token count")
        integer(record["scheduled_ns"], "frozen arrival offset")
        arrivals.append(scheduled - record["scheduled_ns"])
        queue.append(submitted - scheduled); ttft.append(row["ttft_ns"])
        latencies.append(row["latency_ns"]); gaps.extend(itl); outputs.append(tokens)
        details.append(dict(request_id=row["request_id"], frontend_queue_ns=queue[-1], ttft_ns=ttft[-1],
            latency_ns=latencies[-1], num_cached_tokens=row["num_cached_tokens"], ITL=summary(itl),
            output_tokens=128, output_ids_sha256=hashlib.sha256(canonical(tokens)).hexdigest()))
    require(len(set(arrivals)) == 1, "one original frozen arrival clock origin")
    start = integer(result["stream_call_start_ns"], "actual stream call start", 1)
    finish = max(row["finished_ns"] for row in rows)
    drained = integer(result["after_stream_drain_finished_ns"], "actual original drain finish", 1)
    require(start <= arrivals[0] <= min(row["scheduled_ns"] for row in rows) <= finish <= drained,
        "actual stream/arrival/complete/drain clock order")
    return dict(complete_requests=12, output_tokens=1536, frontend_queue=summary(queue), TTFT=summary(ttft),
        ITL=summary(gaps), request_latency=summary(latencies), request_metrics=details,
        cohort_makespan_ns=finish - min(row["scheduled_ns"] for row in rows),
        stream_call_to_last_return_ns=finish - start, drain_after_last_return_ns=drained - finish,
        stream_plus_drain_ns=drained - start, original_engine_steps=front["original_engine_steps"]), outputs


def capture_valid(result):
    capture = result["capture"]
    require(capture["origin"] == "native_gpu_recording" and capture["valid"] is True and
        capture["failures"] == [] and capture["pending_event_pairs"] == 0 and capture["open_event_pair"] is False,
        "invalid/incomplete real native capture")
    frames, witnesses = capture["frames"], capture["event_witnesses"]
    require(frames and len(frames) == len(witnesses) <= 4096, "complete CUDA frame/event bijection")
    ordinals = [frame["native_step_ordinal"] for frame in frames]
    require(ordinals == list(range(ordinals[0], ordinals[0] + len(frames))), "complete native ordinal sequence")
    produced = {row["native_request_id"]: [] for row in result["frontend"]["rows"]}
    for frame, witness in zip(frames, witnesses):
        require(witness["native_step_ordinal"] == frame["native_step_ordinal"] and
            frame["prepared"]["native_step_ordinal"] == frame["native_step_ordinal"] and
            witness["event_elapsed_source"] == "torch.cuda.Event.elapsed_time" and
            witness["cross_clock_absolute_mapping"] is False, "actual paired native/CUDA identities")
        integer(witness["gpu_elapsed_ns"], "actual CUDA event duration", 1)
        require(0 < witness["start_record_before_ns"] <= witness["start_record_after_ns"] <= witness["end_record_before_ns"] <=
            witness["end_record_after_ns"] <= witness["end_completed_query_ns"] and
            witness["start_completed_query_ns"] >= witness["start_record_before_ns"], "real CUDA record/query host brackets")
        for request, tokens in frame["outputs"]:
            require(request in produced and type(tokens) is list and len(tokens) == 1 and
                all(type(t) is int and t >= 0 for t in tokens), "actual captured native sample outputs")
            produced[request].extend(tokens)
    require(all(produced[row["native_request_id"]] == row["output_token_ids"] for row in result["frontend"]["rows"]),
        "native sample capture/front-end complete output alignment")
    return dict(valid=True, frames=len(frames), CUDA_pairs=len(witnesses), captured_output_tokens=1536,
        GPU_event_durations=summary([w["gpu_elapsed_ns"] for w in witnesses]))


def native_activity(result):
    journal = result["journal"]
    require(journal["valid"] is True and journal["lost"] == 0 and journal["run_id"] == result["run_id"] and
        journal["source_sha256"] == result["native_source_ref"]["sha256"], "complete same-source native journal")
    events = journal["events"]
    accepted, completed, totals = {}, {}, {s: dict(accepted_ops=0, accepted_bytes=0, completed_ops=0,
        completed_bytes=0, outstanding_ops=0, accepted_during_frontend=0, completed_during_frontend=0) for s in STAGES}
    previous = -1
    start, finish = result["stream_call_start_ns"], max(r["finished_ns"] for r in result["frontend"]["rows"])
    for event in events:
        require(event["stage"] in totals and event["kind"] in ("accepted", "completed") and
            event["sequence"] > previous, "original journal stage/kind/sequence")
        previous = event["sequence"]
        for key in ("sequence", "operation_sequence", "at_ns", "physical_bytes"):
            integer(event[key], "actual native journal " + key, 1)
        identity = (event["stage"], event["operation_sequence"])
        stage = totals[event["stage"]]
        if event["kind"] == "accepted":
            require(identity not in accepted, "duplicate native acceptance")
            accepted[identity] = event
            stage["accepted_ops"] += 1; stage["accepted_bytes"] += event["physical_bytes"]
            stage["accepted_during_frontend"] += start <= event["at_ns"] <= finish
        else:
            require(identity in accepted and identity not in completed and
                event["physical_bytes"] == accepted[identity]["physical_bytes"] and
                event["at_ns"] >= accepted[identity]["at_ns"] and event["result"] == event["physical_bytes"],
                "original actual native completion join")
            completed[identity] = event
            stage["completed_ops"] += 1; stage["completed_bytes"] += event["physical_bytes"]
            stage["completed_during_frontend"] += start <= event["at_ns"] <= finish
    for stage in totals.values():
        stage["outstanding_ops"] = stage["accepted_ops"] - stage["completed_ops"]
    require(set(accepted) == set(completed), "native operations not completely drained")
    require(result["synthetic_IO_injected"] is False and result["controlled_unit_cost_only"] is False,
        "method run cannot inject controlled calibration I/O")
    decode_intervals = [(f["start_ns"], f["end_ns"]) for f in result["capture"]["frames"] if f["prepared"]["step_kind"] == "decode"]
    ssd_overlap = sum(any(a["at_ns"] <= right and completed[key]["at_ns"] >= left for left, right in decode_intervals)
        for key, a in accepted.items() if a["stage"] == "ssd_read")
    counters = result["actual_native_dispatch_returns"]
    for name in ("calls", "original_returns", "native_deferrals", "other_dispatches"):
        integer(counters[name], "actual native dispatch counter " + name)
    require(counters["calls"] == counters["original_returns"] + counters["native_deferrals"] + counters["other_dispatches"] and
        sum(counters["stages"].values()) == counters["calls"], "actual original dispatch-return counter closure")
    return dict(stages=totals, actual_native_dispatch_returns=counters, SSD_ops_with_decode_host_interval_overlap=ssd_overlap,
        natural_SSD_activity_observed=totals["ssd_read"]["accepted_ops"] > 0,
        natural_SSD_decode_host_overlap_observed=ssd_overlap > 0,
        per_request_SSD_restore_identity_proved=False, overlap_is_device_absolute_mapping=False,
        physical_resource_release_credit=False)


def arm(root, result_path, guard_path, inputs, mode):
    result, result_ref = read(root, result_path)
    guard, guard_ref = read(root, guard_path)
    require(result["schema"] == "bounded_original_model_result_v1" and result["origin"] == "native_gpu_recording" and
        result["mode"] == mode and result["status"] == "COMPLETE_REQUIRES_ORIGINAL_GUARD_CLOSURE_AND_PERFORMANCE_ANALYSIS" and
        result["error"] is None and result["production_qualified"] is False, "actual complete bounded method result")
    require(guard["label"] == result["run_id"] and guard["gpu_uuid"] == result["gpu_uuid"] and
        type(guard["exit"]) is int and guard["exit"] == 0 and type(guard["child_exit"]) is int and guard["child_exit"] == 0 and
        guard["timed_out"] is False and guard["interrupted_signal"] is None and guard["error"] is None and
        guard["gpu_job_attempted"] is True and guard["session_drained"] is True and
        guard["session_members_before_cleanup"] == guard["session_members_after_cleanup"] == [] and
        guard["session_id"] == result["subprocess_sid"] == result["guard_start"]["session_id"] and
        guard["reservation_id"] == result["guard_start"]["id"], "original natural-exit completed guard/session join")
    require(type(guard["elapsed_seconds"]) in (int, float) and math.isfinite(guard["elapsed_seconds"]) and guard["elapsed_seconds"] > 0,
        "actual full guard elapsed duration")
    config = check_ref(root, result["config_ref"])
    require(config["run_id"] == result["run_id"] and config["mode"] == mode and
        check_ref(root, config["inputs_ref"]) == inputs and config["gpu_uuid"] == result["gpu_uuid"],
        "actual child config/frozen inputs/device join")
    require(result["original_engine_shutdown_returned"] is True and result["native_tail_drained"] is True and
        result["native_tail_assertions"] == dict(handler_shutdown=True, worker_alive=False, aio_worker_alive=False, reactor_closed=True) and
        result["native_return_counter_restored"] is True and result["optional_probe_restored"] is True and
        result["native_factory_restored"] is True, "original shutdown/tail/counter/factory closure")
    if mode == "I":
        require(result["I_startup_restored"] is True and result["I_owner_install"]["owner_thread_install"] is True and
            result["I_owner_install"]["drained_before_attach"] is True and result["I_owner_install"]["production_qualified"] is False,
            "actual development I owner lifecycle")
    metrics, outputs = frontend(result, inputs)
    capture = capture_valid(result)
    activity = native_activity(result)
    decisions = None
    if mode == "I":
        evidence = result["I_decisions"]
        require(evidence["gpu_development"] is True and evidence["production_qualified"] is False and
            evidence["production_batch_activation"] is False and evidence["overflow"] is False and evidence["unknown_coverage"] is False,
            "actual finite development decision record without lost coverage")
        decisions = dict(condition_counters=evidence["condition_counters"], eligible_windows=len(evidence["actual_eligible_preview_windows"]),
            eligible_calls=sum(w["actual_eligible_calls"] for w in evidence["actual_eligible_preview_windows"]),
            preview_windows=evidence["preview_windows"], actual_native_deferrals=activity["actual_native_dispatch_returns"]["native_deferrals"],
            proposals_are_not_actual_native_deferrals=True)
    return result, outputs, dict(result_ref=result_ref, completed_guard_ref=guard_ref, metrics=metrics,
        full_guard_seconds=guard["elapsed_seconds"], capture=capture, native_activity=activity, I_decisions=decisions)


def engine_domain(engine):
    result = deepcopy(engine)
    extra = result["kv_transfer_config"]["kv_connector_extra_config"]
    extra.pop("shared_storage_path"); extra.pop("prefix_io_p4_policy")
    extra["prefix_io_parent_admission"].pop("run_id")
    extra.pop("prefix_io_observation_run_id", None)
    return result


def compare(root, inputs_path, u_path, i_path, u_guard, i_guard):
    report = dict(schema="bounded_U_I_single_pair_descriptive_comparison_v1", valid_comparison=False,
        statistical_efficacy_proved=False, SLO_qualification=False, generalization_proved=False,
        production_qualified=False, analysis_GPU_operations=0, invalid_reasons=[])
    try:
        drive = root / DRIVE
        raw = drive.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == DRIVE_SHA, "original frontend drive source drift/missing")
        inputs, input_ref = read(root, inputs_path)
        require(inputs["schema"] == "bounded_u_i_development_request_inputs_v1" and len(inputs["records"]) == 12 and
            inputs["frontend_max_concurrency"] == 3 and inputs["engine_max_num_seqs"] == 1, "frozen twelve-request bounded input")
        u, u_ids, um = arm(root, u_path, u_guard, inputs, "U")
        i, i_ids, im = arm(root, i_path, i_guard, inputs, "I")
        require(u["gpu_uuid"] == i["gpu_uuid"] and u["common_runtime_domain_sha256"] == i["common_runtime_domain_sha256"] and
            u["collector_ref"] == i["collector_ref"] and u["native_source_ref"] == i["native_source_ref"] and
            u["model_manifest_ref"] == i["model_manifest_ref"] and u["kv_layout_ref"] == i["kv_layout_ref"] and
            engine_domain(u["engine_config"]) == engine_domain(i["engine_config"]), "same actual hardware/model/collector/common resource domain")
        require(u_ids == i_ids, "U/I all twelve actual 128-token outputs differ")
        changes = {}
        for name in ("cohort_makespan_ns", "stream_call_to_last_return_ns", "drain_after_last_return_ns", "stream_plus_drain_ns"):
            left, right = um["metrics"][name], im["metrics"][name]
            changes[name] = dict(U=left, I=right, I_minus_U=right-left,
                I_relative_change_percent=(right-left)*100/left if left else None)
        for name in ("frontend_queue", "TTFT", "ITL", "request_latency"):
            left, right = um["metrics"][name]["mean_ns"], im["metrics"][name]["mean_ns"]
            changes[name+"_mean_ns"] = dict(U=left, I=right, I_minus_U=right-left,
                I_relative_change_percent=(right-left)*100/left if left else None)
        left, right = um["full_guard_seconds"], im["full_guard_seconds"]
        changes["full_guard_seconds"] = dict(U=left, I=right, I_minus_U=right-left, I_relative_change_percent=(right-left)*100/left)
        acted = im["native_activity"]["actual_native_dispatch_returns"]["native_deferrals"] > 0
        report.update(valid_comparison=True, frozen_inputs_ref=input_ref, original_drive_source_sha256=DRIVE_SHA,
            complete_equal_output_requests=12, equal_output_tokens=1536, U=um, I=im, changes=changes,
            actual_I_native_deferral_observed=acted, primary="cohort_makespan_ns",
            observed_primary_change_percent=changes["cohort_makespan_ns"]["I_relative_change_percent"],
            interpretation=("One actual exploratory U/I pair; descriptive difference only." if acted else
                "One actual exploratory pair with zero native I deferrals; no active-method benefit established."))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        report["invalid_reasons"].append(type(exc).__name__ + ": " + str(exc))
        report["interpretation"] = "Incomplete/invalid recorded comparison; no effective improvement conclusion."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    for name in ("inputs", "u-result", "i-result", "u-guard", "i-guard", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    root = args.project.resolve(strict=True)
    report = compare(root, args.inputs, args.u_result, args.i_result, args.u_guard, args.i_guard)
    path = Path(args.output)
    if not path.is_absolute():
        path = root / path
    require(path.parent.resolve(strict=True).is_relative_to(root), "existing output parent inside project required")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(valid_comparison=report["valid_comparison"], invalid_reasons=report["invalid_reasons"]), ensure_ascii=False))
    return 0 if report["valid_comparison"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
