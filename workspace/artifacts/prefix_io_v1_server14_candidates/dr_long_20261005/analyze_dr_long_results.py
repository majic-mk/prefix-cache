"""CPU-only analysis of the frozen P316 CAL, U1, DR1, DR2, U2 runs.

Only U1/DR1 and DR2/U2 are paired. F8+D_R is a combined arm; a timing
difference cannot isolate D_R. No model import, experiment fit or SLO claim.
Run index: list of {label, arm, config, result, adapter, guard, frozen_config,
source_lock}; file paths are relative to --project, or explicit absolute paths.
Missing result/frozen leaves remain failed rows, including startup failures.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics

Q = 917504
STAGES = ("ssd_read", "ssd_write", "h2d", "d2h")
SERVER_PROJECT = "/root/autodl-tmp/prefix-io-v1-handoff/project"
CLOCK_REFS = (
    dict(path="third_party/work/vllm-author-p4-02-cpu/vllm/v1/engine/__init__.py", bytes=8835,
         sha256="9e3c3d3c73c57981da92753289531dada9295703e58e7de1400a8182c598890e"),
    dict(path="third_party/work/vllm-author-p4-02-cpu/vllm/v1/metrics/stats.py", bytes=18538,
         sha256="08eb6bb969ad47e4207ba5568beaec35bdb09192b6db046f046dab4892ac1bf3"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, label, minimum=0):
    require(type(value) in (float, int) and math.isfinite(value) and value >= minimum, label)
    return value


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label)
    return value


def read_leaf(project, value):
    if value is None:
        return None, None
    path = Path(value)
    path = path if path.is_absolute() else Path(project) / path
    if not path.is_file():
        return None, dict(path=str(path), missing=True)
    require(not path.is_symlink() and path.stat().st_size <= 32 * 1024**2, "bounded regular result leaf")
    raw = path.read_bytes()
    def unique(rows):
        out = {}
        for k, v in rows:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    document = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: require(False, "nonfinite JSON"))
    return document, dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def request_metrics(report, *, token_clock_verified=False):
    """All latency origins use scheduled arrival. Never use wall arrival_time."""
    rows = report["rows"]
    require(type(rows) is list and len(rows) == 10 and {r["request_id"] for r in rows} ==
        {"c2-" + str(i) for i in range(10)}, "complete original ten-request cohort")
    start = number(report["cohort_start_monotonic"], "original monotonic cohort start", 1)
    output, metrics = {}, []
    for row in sorted(rows, key=lambda r: int(r["request_id"][3:])):
        ids, stamps = row["output_tokens"], row["engine_token_timestamps"]
        require(row.get("per_token_complete") is True and row.get("ambiguous_events") == [] and
            row.get("timestamp_scope") == "engine core output event" and len(ids) == len(stamps) == 128 and
            all(type(t) is int and 0 <= t < 152064 for t in ids) and row.get("prompt_tokens") == 16257 and
            row.get("metrics", {}).get("is_corrupted") is False, "actual full128 uncorrupted original output")
        scheduled = number(row["scheduled_arrival_seconds"], "original scheduled arrival")
        sent = number(row["actual_send_seconds"], "original actual send")
        completed = number(row["response_end_seconds"], "original completion")
        require(scheduled <= sent <= completed and stamps == sorted(stamps) and
            all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in stamps), "actual ordered original request times")
        itl = row["itl_seconds"]
        require(type(itl) is list and len(itl) == 127 and all(math.isclose(v, b - a, rel_tol=0, abs_tol=1e-9)
            for v, a, b in zip(itl, stamps, stamps[1:])), "original per-token interval evidence")
        p95 = number(row["itl_p95_seconds"], "frozen per-request ITL p95")
        ttft = None
        if token_clock_verified:
            ttft = stamps[0] - start - scheduled
            require(ttft >= 0 and stamps[-1] - start <= completed + 1e-6,
                "verified in-process monotonic token events outside original request interval")
        output[row["request_id"]] = ids
        metrics.append(dict(request_id=row["request_id"], family=row["family"],
            scheduled_arrival_seconds=scheduled, actual_send_seconds=sent,
            client_queue_seconds=sent - scheduled, ttft_from_scheduled_arrival_seconds=ttft,
            completion_from_scheduled_arrival_seconds=completed - scheduled,
            in_request_itl_p95_seconds=p95, nominal_num_cached_tokens=row["num_cached_tokens"]))
    return metrics, output


def native_stage_delta(report):
    snapshots = []
    for key in ("cohort_probe_start", "probe", "final_probe"):
        rows = report[key].get("current_native_snapshots")
        require(type(rows) is list and len(rows) == 1, "actual sole native owner snapshot: " + key)
        snapshots.append(rows[0])
    before, after, final = snapshots
    for snapshot in (after, final):
        require(snapshot.get("owner_capture") is True and type(snapshot.get("native")) is dict and
            snapshot["native"] and all(type(x) is int and x == 0 for x in snapshot["native"].values()),
            "actual native tail work remains")
        aio = snapshot["aio"]
        require(aio.get("fatal") is None and all(type(aio.get(k)) is int and aio[k] == 0
            for k in ("outstanding", "pending", "ready", "unreaped")) and
            aio.get("accepted") == aio.get("completed") == aio.get("reaped"), "actual original AIO drain")
    deltas = {}
    for snapshot in snapshots:
        accounting = snapshot.get("stage_accounting")
        require(type(accounting) is dict and accounting.get("valid") is True and accounting.get("bound") is True and
            accounting.get("error") is None and set(accounting["stages"]) == set(STAGES), "actual complete original stage accounting")
    for stage in STAGES:
        first, last = before["stage_accounting"]["stages"][stage], after["stage_accounting"]["stages"][stage]
        for snapshot in (after, final):
            row = snapshot["stage_accounting"]["stages"][stage]
            require(all(type(row[k]) is int and row[k] == 0 for k in ("inflight_ops", "inflight_bytes", "failed_ops")),
                "actual original physical stage not drained")
        ops = integer(last["accepted_ops"], "native accepted ops") - integer(first["accepted_ops"], "initial accepted ops")
        nbytes = integer(last["accepted_bytes"], "native accepted bytes") - integer(first["accepted_bytes"], "initial accepted bytes")
        require(ops >= 0 and nbytes >= 0, "native stage cumulative counters regressed")
        deltas[stage] = dict(accepted_ops=ops, accepted_bytes=nbytes)
    return deltas


def mechanism_evidence(journal, kernel):
    """Join changed ready permutations to successful original syscall records.

    Queue-read return is not syscall acceptance. Only an inverted work pair
    with matching user_data/fd in the kernel successful prefix proves a change.
    Bounded journal deduplication can leave further proposed changes unproved.
    """
    require(journal.get("schema") == "bounded_original_restore_order_journal_v1" and journal.get("valid") is True and
        journal.get("fault") is None and journal.get("overflow") is False and journal.get("new_work_queues") == 0 and
        journal.get("held_native_owners") is False and kernel.get("valid") is True and
        kernel.get("fault") is None and kernel.get("overflow") is False, "complete actual order/submission journals")
    kernel_rows = kernel["records"]
    require(kernel.get("successful_operation_count") == len(kernel_rows) and
        [r["sequence"] for r in kernel_rows] == list(range(1, len(kernel_rows) + 1)), "complete original successful syscall sequence")
    by_user = {}
    for row in kernel_rows:
        uid = integer(row["user_data"], "actual successful syscall user_data")
        require(uid not in by_user and row.get("boundary") == "original_linux_io_submit_successful_prefix" and
            row.get("kind") in ("read", "write"), "unique actual accepted kernel identity")
        integer(row["at_ns"], "actual successful syscall clock", 1)
        integer(row["fd"], "actual successful syscall fd")
        integer(row["nbytes"], "actual successful syscall bytes", 1)
        by_user[uid] = row
    queued = journal["actual_accepted_read_submission_order"]
    require([r["sequence"] for r in queued] == list(range(1, len(queued) + 1)), "complete native ready-read queue sequence")
    linked = []
    for row in queued:
        require(row.get("boundary") == "original_queue_read_returned_then_original_stage_accepted", "original queued-read boundary")
        accepted = by_user.get(row["user_data"])
        if accepted is not None and accepted["kind"] == "read" and accepted["fd"] == row["fd"] and accepted["nbytes"] == Q:
            linked.append((row, accepted))
    confirmed, unconfirmed = [], []
    changes = journal["actual_order_changes"]
    for change in changes:
        before = [tuple(w) for w in change["before_work_ids"]]
        after = [tuple(w) for w in change["after_work_ids"]]
        require(change.get("action") == "selected" and change.get("actual_native_submission_proved_here") is False and
            before != after and len(before) == len(set(before)) == len(after) and set(before) == set(after) and
            len(before) <= 64 and change.get("dependency_parent_ids"), "actual legal closure permutation and dependency required")
        deps = change["dependency_parent_ids"]
        require(all(type(pid) is int and pid > 0 and any(w[0] == pid for w in before) for pid in deps), "actual restore-parent dependency IDs")
        windows = [w for w in journal["common_native_ready_windows"] if w["before_work_ids"] == change["before_work_ids"] and
            w["after_work_ids"] == change["after_work_ids"] and w["at_ns"] >= change["at_ns"]]
        proof = None
        for window in windows:
            facts = {p[0]: p for p in window["parent_facts"]}
            if not all(pid in facts and facts[pid][5] is False and facts[pid][6] is False for pid in deps):
                continue
            available = {}
            for native, accepted in linked:
                wid = tuple(native["work_id"])
                if wid in set(after) and native["at_ns"] >= window["at_ns"] and accepted["at_ns"] >= change["at_ns"]:
                    available.setdefault(wid, (native, accepted))
            for i, left in enumerate(after):
                for right in after[i + 1:]:
                    if (before.index(left) > before.index(right) and left in available and right in available and
                            available[left][0]["sequence"] < available[right][0]["sequence"] and
                            available[left][1]["sequence"] < available[right][1]["sequence"]):
                        proof = dict(change=deepcopy(change), original_ready_window=deepcopy(window),
                            inversion_work_ids=[left, right], native_queue_receipts=[available[left][0], available[right][0]],
                            successful_kernel_receipts=[available[left][1], available[right][1]])
                        break
                if proof is not None:
                    break
            if proof is not None:
                break
        (confirmed if proof is not None else unconfirmed).append(proof if proof is not None else deepcopy(change))
    return dict(classification="EXERCISED_ACTUAL_SUBMISSION_ORDER" if confirmed else "NOT_EXERCISED",
        proposed_changed_permutations=len(changes), kernel_confirmed_order_changes=len(confirmed),
        confirmed_changes=confirmed, unconfirmed_changes=unconfirmed,
        legal_candidate_diagnostics=journal["first_parent_reason_windows"], reason_counts=journal["reason_counts"],
        native_ready_calls=journal["native_ready_calls"], actual_successful_kernel_operations=len(kernel_rows),
        parent_read_queue_receipts=len(queued), kernel_joined_parent_read_queue_receipts=len(linked),
        resource_release_credit=False, production_qualified=False)


def normal_engine(engine):
    value = deepcopy(engine)
    extra = value["kv_transfer_config"]["kv_connector_extra_config"]
    for key in ("shared_storage_path", "prefix_io_observation_run_id", "prefix_io_p4_policy"):
        extra.pop(key, None)
    if type(extra.get("prefix_io_parent_admission")) is dict:
        extra["prefix_io_parent_admission"].pop("run_id", None)
    return value


def validate_original_fixed8(fixed, run_id):
    """Exact preserved P316 fixed8-control, with only the run ID rebound."""
    expected = dict(schema_version=1, run_id=run_id, mode="fixed", epoch_ns=10000000,
        byte_quantum=Q,
        cumulative={s: dict(ops=64 if s == "h2d" else 8, bytes=(64 if s == "h2d" else 8)*Q)
            for s in STAGES},
        inflight={s: dict(ops=64, bytes=(64 if s.startswith("ssd_") else 1152)*Q)
            for s in STAGES},
        shared_ssd_cumulative=dict(ops=16, bytes=16*Q),
        shared_ssd_inflight=dict(ops=64, bytes=64*Q),
        shared_copy_cumulative_bytes=72*Q, shared_copy_inflight_bytes=1152*Q,
        reserve_staging_bytes=0, max_accepted_parents=64, sample_max_age_ns=20000000,
        max_wait_ns=200000000)
    require(fixed == expected, "actual fixed-stage config differs from preserved original P316 fixed8-control")


def validate_frozen_guard_command(command, runner_ref, config_ref):
    """Accept the frozen relative runner or its exact known-server path."""
    for ref in (runner_ref, config_ref):
        path = ref["path"]
        require(type(path) is str and str(PurePosixPath(path)) == path and
            not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts,
            "canonical relative frozen guard source ref")
    prefix = [SERVER_PROJECT + "/.venv/bin/python", "-B"]
    suffix = ["--project", SERVER_PROJECT, "--config", config_ref["path"], "--execute"]
    require(command in (prefix + [runner_ref["path"]] + suffix,
        prefix + [SERVER_PROJECT + "/" + runner_ref["path"]] + suffix),
        "actual guard argv differs from exact known-server frozen source/config")


def validate_run(entry, docs, files, *, expected_gpu_uuid, expected_common_domain_sha256, clock_proof):
    result, adapter, guard, config, frozen, lock = (docs[k] for k in
        ("result", "adapter", "guard", "config", "frozen_config", "source_lock"))
    require(all(type(d) is dict for d in (result, adapter, guard, config, frozen, lock)), "missing actual model/evidence leaf")
    rid, arm = config["run_id"], entry["arm"]
    require(config["arm"] == adapter["arm"] == arm and adapter["run_id"] == result["run_id"] == guard["label"] == rid,
        "actual run/arm join")
    for document in (result, adapter, config, guard):
        require(document.get("gpu_uuid") == expected_gpu_uuid, "actual current GPU mismatch")
    for document in (result, adapter, config):
        require(document.get("common_runtime_domain_sha256") == expected_common_domain_sha256, "actual common resource/source domain mismatch")
    require(result.get("status") == "PASSED_NATIVE_C2_DEVELOPMENT_REPLAY" and result.get("error") is None and
        result.get("engine_shutdown") == "completed" and adapter.get("adapter_error") is None and
        adapter.get("original_engine_shutdown_returned") is True and adapter.get("native_tail_drained") is True and
        adapter.get("optional_probe_restored") is True and adapter.get("kernel_submission_tap_restored") is True,
        "actual original output/startup/shutdown/observer failure")
    require(type(guard.get("exit")) is int and guard["exit"] == 0 and type(guard.get("child_exit")) is int and
        guard["child_exit"] == 0 and guard.get("timed_out") is False and guard.get("error") is None and
        guard.get("interrupted_signal") is None and guard.get("gpu_job_attempted") is True and
        guard.get("session_drained") is True and guard.get("session_members_before_cleanup") == [] and
        guard.get("session_members_after_cleanup") == [], "actual successful original guard and empty OS session")
    active = adapter["guard_start"]
    validate_frozen_guard_command(guard["command"], config["runner_ref"], adapter["config_ref"])
    require(active["id"] == guard["reservation_id"] and active["label"] == rid and active["gpu_uuid"] == expected_gpu_uuid and
        active["session_id"] == active["process_group"] == adapter["subprocess_sid"] == result["subprocess_sid"] == guard["session_id"] and
        active["command"] == guard["command"],
        "actual frozen original child and session join")
    for name, expected in (("config", adapter["config_ref"]), ("source_lock", config["source_lock_ref"])):
        require(files[name]["bytes"] == expected["bytes"] and files[name]["sha256"] == expected["sha256"], "downloaded actual bytes differ from remote ref: " + name)
    locked = {r["path"]: r for r in lock["files"]}
    require(len(locked) == len(lock["files"]) and adapter["adapter_ref"] == config["runner_ref"] == locked.get(config["runner_ref"]["path"]) and
        adapter["runtime_refs"] == result["runtime_refs"] == config["runtime_refs"] and
        all(locked.get(path) == row for path, row in config["runtime_refs"].items()), "actual shared native/adapter source closure")
    require(result["manifest_sha256"] == config["manifest_ref"]["sha256"] and
        result["curves_sha256"] == config["curves_ref"]["sha256"] and
        locked.get(config["manifest_ref"]["path"]) == config["manifest_ref"] and
        locked.get(config["curves_ref"]["path"]) == config["curves_ref"], "actual original input and cost-admission source pins")
    loaded = adapter["loaded_native_source_binding"]
    require(type(loaded) is list and loaded and any(r["module"] == "py_kvcache.reactor" for r in loaded) and
        all(locked.get(r["source_ref"]["path"]) == r["source_ref"] for r in loaded) and
        adapter["original_planner_identity"].get("actual_planner_present") is True and
        adapter["original_planner_identity"].get("same_manager_planner") is True, "actual original worker capture/planner/native source binding")
    tail, assertions = adapter["post_shutdown"], adapter["native_tail_assertions"]
    shutdown_aio = tail["snapshot"]["aio"]
    require(set(tail["actual_aio"]) == {"accepted", "completed", "reaped", "outstanding", "pending", "ready", "unreaped"} and
        all(v == shutdown_aio[k] for k, v in tail["actual_aio"].items()) and
        tail["actual_native"] == tail["snapshot"]["native"], "actual shutdown summary differs from original shutdown snapshot")
    require(assertions.get("handler_shutdown") is True and assertions.get("worker_alive") is False and
        assertions.get("aio_worker_alive") is False and assertions.get("reactor_closed") is True and
        all(type(x) is int and x == 0 for x in tail["actual_native"].values()) and tail["actual_handler_active"] == 0 and
        tail.get("observation_failures") == 0 and shutdown_aio.get("closed") is True and
        shutdown_aio.get("drained") is True and shutdown_aio.get("fatal") is None and
        all(type(tail["actual_aio"].get(k)) is int and tail["actual_aio"][k] == 0 for k in ("outstanding", "pending", "ready", "unreaped")) and
        tail["actual_aio"]["accepted"] == tail["actual_aio"]["completed"] == tail["actual_aio"]["reaped"], "actual both workers ended and native/AIO closure")
    manifest, engine = frozen["manifest"], frozen["engine"]
    profile, pattern = ("all_hit", [0,1,2,0,1,2,0,1,2,0]) if arm == "CAL" else (
        "mixed_readwrite", [0,3,1,0,4,2,3,1,4,2])
    require(manifest["profile"] == result["profile"] == profile and len(manifest["requests"]) == 10 and
        [r["family_index"] for r in manifest["requests"]] == pattern and
        manifest["output_tokens"] == 128 and manifest["cache_resets"] == result["cache_resets"] == 0 and
        manifest["artificial_io_delay"] is False and manifest["slo"] is None and
        result["source_preservation"]["changed"] == [] and result["source_preservation"]["checked_files"] == 3048,
        "frozen original input/arrival/cache/source contract")
    for k, value in dict(dtype="bfloat16", kv_cache_memory_bytes=2147483648, max_num_seqs=2,
                         max_model_len=16400, max_num_batched_tokens=16400).items():
        require(engine.get(k) == value, "original resource or executor parameter changed")
    extra = engine["kv_transfer_config"]["kv_connector_extra_config"]
    require(extra.get("iodepth") == 8 and extra.get("staging_mem") == 1 and extra.get("load_planner") == "on" and
        extra.get("enable_preload") is True and extra.get("preload_share_staging") is True, "original I/O/staging/admission path changed")
    controls = config["engine_controls"]
    require(all(extra.get(k) == v for k, v in controls.items()), "saved actual engine differs from declared controls")
    policy = controls.get("prefix_io_p4_policy", {})
    if arm == "U":
        require(policy.get("mode") == "off" and result["actual_p4_mode"] == result["actual_stage_mode"] == "off" and
            result["actual_ordinary_quota_installed"] is False, "actual U new-policy off path")
    else:
        fixed = policy.get("fixed_stage_policy", {})
        validate_original_fixed8(fixed, rid)
        require(arm in ("CAL", "F8+D_R") and policy.get("mode") == "dependency_only" and fixed.get("mode") == "fixed" and
            result["actual_p4_mode"] == "dependency_only" and result["actual_stage_mode"] == "fixed" and
            result["actual_ordinary_quota_installed"] is True, "actual declared F8 plus restore-only ordering path")
    source_rows = clock_proof.get("source_rows", []) if type(clock_proof) is dict else []
    clock_verified = len(source_rows) == 2 and all(r["source_ref"] == expected and locked.get(expected["path"]) == expected
        for r, expected in zip(source_rows, CLOCK_REFS))
    metrics, outputs = request_metrics(result, token_clock_verified=clock_verified)
    by_id = {r["request_id"]: r for r in result["rows"]}
    for item in manifest["requests"]:
        row = by_id["c2-" + str(item["request_id"])]; family = manifest["families"][item["family_index"]]
        require(row["scheduled_arrival_seconds"] == item["scheduled_time"] and row["family"] == family["name"] and
            row["initial_ssd_present"] is family["initial_ssd_present"] and len(family["tokens"]) == 16257,
            "actual request arrival/prompt family differs from frozen input")
    primary = number(result["response_seconds"], "original batch response_seconds", 1e-12)
    drained = number(result["cohort_seconds_including_drain"], "original cohort including drain", primary)
    tail_seconds = number(result["tail_drain_seconds"], "original tail drain_seconds")
    stage = native_stage_delta(result)
    read_bytes, write_bytes = integer(result["cohort_read_bytes"], "actual cohort read bytes"), integer(result["cohort_write_bytes"], "actual cohort write bytes")
    require(read_bytes > 0 and read_bytes == stage["ssd_read"]["accepted_bytes"] and write_bytes == stage["ssd_write"]["accepted_bytes"],
        "actual normal-model SSD bytes disagree with native counters")
    require(primary >= max(r["response_end_seconds"] for r in result["rows"]) - manifest["requests"][0]["scheduled_time"],
        "original batch primary ends before final complete output")
    mechanism = mechanism_evidence(adapter["restore_order_journal"], adapter["kernel_actual_submission_order"])
    return dict(primary_response_seconds=primary, cohort_seconds_including_drain=drained, tail_drain_seconds=tail_seconds,
        throughput_generated_tokens_per_second=1280/primary, actual_cohort_ssd_read_bytes=read_bytes,
        actual_cohort_ssd_write_bytes=write_bytes, actual_cohort_h2d_accepted_bytes=stage["h2d"]["accepted_bytes"],
        native_stage_deltas=stage, requests=metrics, ttft_monotonic_clock_verified=clock_verified,
        mechanism=mechanism, forecast_disabled=adapter.get("forecast_disabled"),
        calibration_coverage_missing=adapter.get("calibration_coverage_missing")), outputs, dict(
        engine=normal_engine(engine), manifest=manifest, sampling=frozen["sampling"], initial_cache=result["initial_cache"],
        source_registration=config["storage_registration_ref"], curves=config["curves_ref"],
        model_identity=adapter["model_identity"], runtime_refs=config["runtime_refs"])


def analyze_runs(project, entries, *, expected_gpu_uuid, expected_common_domain_sha256, clock_proof=None):
    require(type(entries) is list and entries and len({r["label"] for r in entries}) == len(entries), "unique declared run labels")
    rows, internals = [], {}
    for entry in entries:
        documents, leaves, problems = {}, {}, []
        for key in ("config", "result", "adapter", "guard", "frozen_config", "source_lock"):
            try:
                documents[key], leaves[key] = read_leaf(project, entry.get(key))
            except Exception as exc:
                documents[key], leaves[key] = None, dict(path=entry.get(key), read_error=str(exc))
        guard = documents.get("guard") or {}; adapter = documents.get("adapter") or {}
        row = dict(label=entry["label"], arm=entry["arm"], evidence_files=leaves, valid_performance_sample=False,
            GPU_guard_attempted=guard.get("gpu_job_attempted"), guard_exit=guard.get("exit"),
            guard_elapsed_seconds=guard.get("elapsed_seconds"), OS_session_drained=guard.get("session_drained"),
            adapter_error=adapter.get("adapter_error"), raw_status=(documents.get("result") or {}).get("status"), metrics=None)
        try:
            metrics, output, controls = validate_run(entry, documents, leaves, expected_gpu_uuid=expected_gpu_uuid,
                expected_common_domain_sha256=expected_common_domain_sha256, clock_proof=clock_proof)
            row.update(valid_performance_sample=True, metrics=metrics)
            internals[entry["label"]] = dict(row=row, output=output, controls=controls, guard=guard)
        except Exception as exc:
            problems.append(type(exc).__name__ + ": " + str(exc))
        row["failure_reasons"] = problems
        rows.append(row)
    pairs = []
    for uid, did, order in (("U1", "DR1", "U_THEN_DR"), ("U2", "DR2", "DR_THEN_U")):
        pair = dict(U_label=uid, DR_label=did, preregistered_order=order, valid_performance_pair=False,
            response_improvement_fraction=None, classification="INCOMPLETE_OR_INVALID_PAIR")
        u, d = internals.get(uid), internals.get(did)
        if u is not None and d is not None:
            first, last = (u, d) if order == "U_THEN_DR" else (d, u)
            sequential = (number(first["guard"]["started_unix"], "actual first guard start", 1) +
                number(first["guard"]["elapsed_seconds"], "actual first guard duration") <=
                number(last["guard"]["started_unix"], "actual second guard start", 1))
            if u["row"]["arm"] != "U" or d["row"]["arm"] != "F8+D_R":
                pair["invalid_reason"] = "declared pair arm mismatch"
            elif not sequential:
                pair["invalid_reason"] = "actual sequential guard order differs from preregistered pair"
            elif u["controls"] != d["controls"]:
                pair["invalid_reason"] = "common input/cache/model/resource controls differ"
            elif u["output"] != d["output"]:
                pair["invalid_reason"] = "complete output token mismatch"
                u["row"]["valid_performance_sample"] = d["row"]["valid_performance_sample"] = False
                u["row"]["failure_reasons"].append("paired complete output mismatch")
                d["row"]["failure_reasons"].append("paired complete output mismatch")
            else:
                tu, td = u["row"]["metrics"]["primary_response_seconds"], d["row"]["metrics"]["primary_response_seconds"]
                actual = d["row"]["metrics"]["mechanism"]["kernel_confirmed_order_changes"]
                pair.update(valid_performance_pair=True, response_improvement_fraction=(tu-td)/tu,
                    U_response_seconds=tu, DR_response_seconds=td, complete_outputs_equal=True,
                    actual_DR_kernel_order_changes=actual, classification="NOT_EXERCISED" if actual == 0 else
                        ("COMBINED_ARM_FASTER" if td < tu else "COMBINED_ARM_NO_IMPROVEMENT"),
                    attribution="F8+D_R versus U only; cannot isolate D_R from the fixed-stage contribution")
        pairs.append(pair)
    valid = [p for p in pairs if p["valid_performance_pair"]]
    return dict(schema="P316_U_F8_DR_original_result_analysis_v1", scope="current-device exploratory development comparison",
        primary_metric="original response_seconds: first scheduled arrival to all complete outputs",
        run_rows=rows, preregistered_adjacent_pairs=pairs, valid_pair_count=len(valid),
        median_response_improvement_fraction=statistics.median(p["response_improvement_fraction"] for p in valid) if valid else None,
        CAL_in_effect_estimate=False, all_cross_pairs_computed=False, formal_service_SLO=None,
        statistical_significance_claim=False, isolated_DR_improvement_proved=False,
        production_prediction_qualified=False, GPU_operations_this_analysis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--common-domain-sha256", required=True)
    parser.add_argument("--clock-proof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    entries, index_ref = read_leaf(args.project, str(args.runs))
    proof, proof_ref = read_leaf(args.project, str(args.clock_proof)) if args.clock_proof else (None, None)
    result = analyze_runs(args.project, entries, expected_gpu_uuid=args.gpu_uuid,
        expected_common_domain_sha256=args.common_domain_sha256, clock_proof=proof)
    result.update(run_index_ref=index_ref, monotonic_clock_proof_ref=proof_ref)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(valid_pair_count=result["valid_pair_count"], output=str(args.output), GPU_operations=0)))
    return 0 if result["valid_pair_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
