"""Read archived cost evidence using only the Python standard library.

This is an integrity/field audit, not a receipt issuer, fitter, or GPU runner.
Original server paths remain the evidence identities. Local flat copies are
resolved only through a supplied, byte-checked delivery mapping.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

A = "artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004"
D = "artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004"
CAL = "artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004"
NORMAL = "artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004"
LOCAL = "artifacts/prefix_io_v1_server12_candidates/native_gpu_delivery"
STAGES = ("ssd_read", "ssd_write", "h2d", "d2h")


class AuditRejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditRejected(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ": strict integer required")
    return value


def pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(AuditRejected("nonfinite JSON: " + value)))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reference(path, data):
    return dict(path=path, bytes=len(data), sha256=digest(data))


def canonical_ref(row):
    require(type(row) is dict and all(k in row for k in ("path", "bytes", "sha256")), "reference shape")
    path = row["path"]
    require(type(path) is str and path and "\\" not in path and ":" not in path,
        "reference must be a relative POSIX path")
    pure = PurePosixPath(path)
    require(not pure.is_absolute() and pure.as_posix() == path and
        all(x not in ("", ".", "..") for x in path.split("/")), "reference traversal/normalization")
    integer(row["bytes"], "reference bytes")
    sha = row["sha256"]
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set("0123456789abcdef"), "reference SHA")
    return {k: row[k] for k in ("path", "bytes", "sha256")}


def contained_file(root, candidate):
    root = root.resolve()
    candidate = Path(candidate)
    if not candidate.is_absolute():
        candidate = root / candidate
    # Lexical confinement is checked before resolution; symlinks are never used.
    require(".." not in candidate.parts, "lexical traversal")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise AuditRejected("file outside project") from exc
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), "symlink evidence path")
    require(candidate.resolve().is_relative_to(root) and candidate.is_file(), "evidence file unavailable")
    return candidate


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.rows = {}
        self.locations = {}
        self.reads = {}
        self.manifests = []

    def add_manifest(self, filename):
        path = contained_file(self.root, filename)
        data = path.read_bytes()
        doc = parse_json(data)
        rows = doc.get("files")
        require(type(rows) is list and 1 <= len(rows) <= 1024, "bounded delivery mapping")
        seen = set()
        for raw in rows:
            ref = canonical_ref(raw)
            require(ref["path"] not in seen, "duplicate manifest path")
            seen.add(ref["path"])
            require(ref["path"] not in self.rows or self.rows[ref["path"]] == ref,
                "conflicting manifest evidence")
            self.rows[ref["path"]] = ref
            if "local_path" in raw:
                # Relocation uses only the manifest-provided basename, never a
                # recursive search or an alternate copy with unchecked bytes.
                require(type(raw["local_path"]) is str, "local mapping path type")
                local = Path(raw["local_path"])
                if not local.is_absolute() or not local.is_relative_to(self.root):
                    local = path.parent / "recorded_files" / PureWindowsPath(raw["local_path"]).name
                self.locations[ref["path"]] = local
        self.manifests.append(dict(reference=reference(path.relative_to(self.root).as_posix(), data),
            mapped_files=len(rows), schema=doc.get("schema"), recorded_status=doc.get("status")))
        self.reads[path.relative_to(self.root).as_posix()] = (path, reference(path.relative_to(self.root).as_posix(), data))

    def bytes(self, identity):
        require(identity in self.rows, "reference absent from verified manifest: " + identity)
        ref = self.rows[identity]
        direct = self.root / identity
        location = direct if direct.is_file() or direct.is_symlink() else self.locations.get(identity)
        require(location is not None, "local/server mapping unavailable: " + identity)
        path = contained_file(self.root, location)
        data = path.read_bytes()
        require(reference(identity, data) == ref, "evidence byte/SHA drift: " + identity)
        self.reads[identity] = (path, ref)
        return data

    def json(self, identity):
        return parse_json(self.bytes(identity))

    def from_ref(self, row):
        ref = canonical_ref(row)
        require(self.rows.get(ref["path"]) == ref, "embedded reference differs from manifest")
        return self.json(ref["path"])

    def recheck(self):
        result = []
        for identity, (path, before) in sorted(self.reads.items()):
            contained_file(self.root, path)
            after = reference(identity, path.read_bytes())
            require(after == before, "input changed during audit: " + identity)
            result.append(after)
        return result


def amount(row):
    require(type(row) is dict and set(row) in ({"ops", "bytes"}, {"ops", "nbytes"}), "I/O amount schema")
    return (integer(row["ops"], "I/O operations"), integer(row.get("bytes", row.get("nbytes")), "I/O bytes"))


def audit_capture(capture, window, *, expected_event_sha, offset=16):
    require(capture.get("origin") == "native_gpu_recording" and capture.get("valid") is True
        and capture.get("failures") == [] and capture.get("pending_event_pairs") == 0
        and capture.get("open_event_pair") is False, "actual complete capture")
    require(capture.get("selected_offsets") == [offset] and
        capture.get("gpu_duration_scope") == "execute_model_through_sample_tokens_current_stream"
        and capture.get("host_window_scope") == "original_execute_through_original_sample_host_calls"
        and capture.get("cross_clock_absolute_mapping") is False
        and capture.get("no_added_synchronization") is True, "unchanged selected/full-step clock domain")
    event = capture["event_source"]
    require(event.get("sha256") == expected_event_sha and event.get("class_name") == "Event"
        and event.get("module") == "torch.cuda.streams", "actual event source metadata")
    frames, witnesses = capture["frames"], capture["event_witnesses"]
    require(type(frames) is list and len(frames) == 128 and type(witnesses) is list and len(witnesses) == 128,
        "complete 128 frames and witnesses")
    output = window["frontend"]["output"]
    native_id, ids = output["native_request_id"], output["output_token_ids"]
    require(output["prompt_token_ids"] == window["prompt_token_ids"] and len(ids) == 128,
        "actual complete output/prompt")
    prompt = window["prompt_token_ids"]
    require(type(prompt) is list and len(prompt) == 129 and
        all(type(v) is int for v in prompt + ids), "typed complete prompt/output token IDs")
    steps = window["frontend"]["steps"]
    require(len(steps) == 128, "frontend complete original steps")
    first = integer(frames[0]["native_step_ordinal"], "first ordinal", 1)
    result, previous_end = [], 0
    for i, (frame, witness, step) in enumerate(zip(frames, witnesses, steps)):
        ordinal = first + i
        require(type(frame.get("native_step_ordinal")) is int and frame["native_step_ordinal"] == ordinal
            and type(witness.get("native_step_ordinal")) is int and witness["native_step_ordinal"] == ordinal,
            "complete consecutive frame/witness ordinal")
        begin, end = integer(frame["start_ns"], "frame start", 1), integer(frame["end_ns"], "frame end", 1)
        require(previous_end <= begin < end, "ordered nonoverlapping host frame calls")
        previous_end = end
        prepared = frame["prepared"]
        expected = dict(native_step_ordinal=ordinal, batch=1, active_decode=0 if i == 0 else 1,
            prefill_tokens=1 if i == 0 else 0, context_length=128 + i,
            step_kind="prefill" if i == 0 else "decode", context_basis="pre_computed_tokens")
        require(all(type(prepared.get(k)) is type(v) and prepared.get(k) == v for k, v in expected.items()),
            "actual prepared load/ordinal differs")
        require(prepared["rows"] == [dict(request_id=native_id, pre_context=128+i, prompt_tokens=129, scheduled_tokens=1)]
            and prepared["input_seq_lens_from_cpu_inputs"] == [129 if i == 0 else 129+i]
            and frame["outputs"] == [[native_id, [ids[i]]]], "all original prepared rows/outputs")
        require(frame.get("intended_timing_scope") == "full_decode_step", "full-step intended scope")
        times = {k: integer(witness[k], k, 1) for k in ("start_record_before_ns", "start_record_after_ns",
            "start_completed_query_ns", "end_record_before_ns", "end_record_after_ns", "end_completed_query_ns")}
        require(times["start_record_before_ns"] <= times["start_record_after_ns"] <= times["start_completed_query_ns"]
            and times["start_record_after_ns"] <= begin < end <= times["end_record_before_ns"]
            <= times["end_record_after_ns"] <= times["end_completed_query_ns"], "CUDA witness causal enclosure")
        duration = integer(witness["gpu_elapsed_ns"], "witness elapsed", 1)
        require(witness.get("event_elapsed_source") == "torch.cuda.Event.elapsed_time"
            and witness.get("cross_clock_absolute_mapping") is False
            and duration <= times["end_completed_query_ns"] - times["start_record_before_ns"], "CUDA elapsed domain")
        require(frame.get("gpu_elapsed_ns") is None or
            (type(frame["gpu_elapsed_ns"]) is int and frame["gpu_elapsed_ns"] == duration), "raw elapsed vs witness mismatch")
        require(integer(step["before_ns"], "frontend before", 1) <= begin < end <= integer(step["after_ns"], "frontend after", 1)
            and step["cumulative_output_counts"] == [i+1], "actual frontend/ordinal enclosure")
        result.append(dict(offset=i, native_step_ordinal=ordinal, context_length=128+i,
            step_kind=prepared["step_kind"], gpu_elapsed_ns_from_witness=duration,
            raw_gpu_elapsed_ns=frame.get("gpu_elapsed_ns"), host_call_ns=end-begin))
    require(offset == 16 and result[offset]["context_length"] == 144, "fixed selected offset/context")
    require(result[15]["native_step_ordinal"]+1 == result[offset]["native_step_ordinal"] ==
        result[17]["native_step_ordinal"]-1, "selected boundary neighbors")
    return dict(full_frames=128, full_witnesses=128, full_output_tokens=128,
        prompt_token_ids_sha256=digest(json.dumps(prompt,separators=(",",":")).encode()),
        output_token_ids_sha256=digest(json.dumps(ids,separators=(",",":")).encode()),
        first_native_step_ordinal=first, last_native_step_ordinal=first+127,
        measured_offset=offset, selected=result[offset], warmup_offset_1=result[1],
        selected_neighbor_offsets=[result[15], result[17]], all_saved_steps=result,
        raw_gpu_elapsed_missing_count=sum(f.get("gpu_elapsed_ns") is None for f in frames),
        event_source_ref={k:event[k] for k in ("path", "bytes", "sha256")},
        GPU_host_absolute_clock_mapping_inferred=False,
        any_step_outside_selected_condition_not_used_to_refit=True)


def audit_io(raw, window, audit, *, action):
    capture = window["capture"]
    selected = capture["frames"][16]
    witness = capture["event_witnesses"][16]
    actions = capture["actions"]
    require(type(actions) is list and len(actions) == 1, "one original selected trigger")
    trigger = actions[0]
    require(type(trigger.get("step_offset")) is int and trigger["step_offset"] == 16
        and trigger.get("native_step_ordinal") == selected["native_step_ordinal"]
        and trigger.get("action_enabled") is action, "selected action offset/ordinal/arm")
    journal = raw["native_journal"]
    require(journal.get("valid") is True and journal.get("lost") == 0, "saved journal complete")
    require(journal.get("source_sha256") == raw["native_source_binding"]["actual_source_ref"]["sha256"],
        "same actual original native journal source")
    events = journal["events"]
    accepted, completed = {}, {}
    last = 0
    for sequence, event in enumerate(events, 1):
        require(type(event.get("sequence")) is int and event["sequence"] == sequence
            and event.get("stage") in STAGES and event.get("kind") in ("accepted", "completed"), "original journal sequence/stage")
        stamp = integer(event["at_ns"], "journal timestamp", 1)
        require(stamp >= last, "journal timestamp order")
        last = stamp
        op = integer(event["operation_sequence"], "operation identity", 1)
        integer(event["physical_bytes"], "physical bytes", 1)
        if event["kind"] == "accepted":
            require(op == sequence and op not in accepted, "unique native accepted operation")
            accepted[op] = event
        else:
            require(op in accepted and op not in completed and event["stage"] == accepted[op]["stage"]
                and event["physical_bytes"] == accepted[op]["physical_bytes"]
                and type(event.get("result")) is int and event["result"] == event["physical_bytes"], "completed exact physical bytes")
            completed[op] = event
    require(set(accepted) == set(completed), "complete saved operation lifetimes")
    first = capture["event_witnesses"][0]["start_record_before_ns"]
    end = capture["event_witnesses"][-1]["end_record_after_ns"]
    selected_events = [e for e in accepted.values() if first < e["at_ns"] < end]
    require(len(selected_events) == (1 if action else 0), "whole measured request exact operation count")
    outstanding = [e for op,e in accepted.items() if e["at_ns"] < witness["start_record_before_ns"]
        and completed[op]["at_ns"] >= witness["start_record_before_ns"]]
    require(not outstanding, "selected step existing I/O must be empty")
    require(window["native_before"]["owner_snapshot"]["admission"]["max_accepted_parents"] == 8,
        "original owner cap")
    result = dict(existing_io=[dict(ops=0, bytes=0)]*4,
        whole_request_accepted_operations=len(selected_events), physical_bytes=0,
        selected_step_native_acceptance_verified=False, resource_release_credit=False,
        GPU_completion_nonoverlap_inferred=False)
    if action:
        event = selected_events[0]
        payload = window["independent_payload"]
        require(event["stage"] == "ssd_read" and event["physical_bytes"] == 917504
            and payload.get("ssd_only") is True and payload.get("cache_miss_before_submit") is True
            and type(payload.get("operations")) is int and payload["operations"] == 1
            and type(payload.get("physical_bytes")) is int and payload["physical_bytes"] == 917504,
            "exact original independent SSD payload")
        keys, foreground = payload["preload_key_sha256"], payload["request_prefix_key_sha256"]
        require(len(keys) == 1 and foreground and not set(keys) & set(foreground), "independent preload/foreground keys")
        require(selected["start_ns"] < witness["start_completed_query_ns"] < trigger["trigger_before_ns"]
            <= trigger["trigger_after_ns"] < witness["end_record_before_ns"] and
            trigger["trigger_before_ns"] <= event["at_ns"] < witness["end_record_before_ns"]
            and selected["start_ns"] < event["at_ns"] < selected["end_ns"], "selected witness/trigger/native acceptance enclosure")
        require(trigger["trigger_result"].get("accepted") is True
            and trigger["trigger_result"].get("physical_bytes") == 917504
            and trigger["trigger_result"].get("h2d_requested") is False, "original SSD-only trigger outcome")
        result.update(physical_bytes=917504, selected_step_native_acceptance_verified=True,
            accepted_native_operation=event, completed_native_operation=completed[event["operation_sequence"]])
    else:
        require(trigger.get("trigger_result") is None and window.get("independent_payload") is None, "baseline no action payload")
    return result


def normalized_engine(config):
    result = parse_json(json.dumps(config).encode())
    extra = result["kv_transfer_config"]["kv_connector_extra_config"]
    require(type(extra.get("shared_storage_path")) is str, "actual private engine path")
    del extra["shared_storage_path"]
    return result


def audit_guard(evidence, row, label, gpu_uuid):
    guard = evidence.from_ref(row)
    require(guard.get("label") == label and guard.get("gpu_uuid") == gpu_uuid
        and guard.get("gpu_job_attempted") is True and guard.get("session_drained") is True
        and type(guard.get("exit")) is int and guard["exit"] == 0
        and type(guard.get("child_exit")) is int and guard["child_exit"] == 0
        and guard.get("timed_out") is False and guard.get("error") is None
        and guard.get("session_members_before_cleanup") == []
        and guard.get("session_members_after_cleanup") == [], "saved natural completed guard")
    require(type(guard.get("reservation_id")) is str and guard["reservation_id"]
        and type(guard.get("session_id")) is int and guard["session_id"] > 0, "saved guard reservation/session")
    return dict(guard_ref=row, completed_guard_fields_match=True,
        live_guard_or_ledger_replayed=False, authority_issued=False)


def signature_fields(source):
    tree = ast.parse(source)
    load = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "load_verified_single_file")
    call = next(x for x in ast.walk(load) if isinstance(x, ast.Call)
        and isinstance(x.func, ast.Name) and x.func.id == "ExactSingleFileReceipt")
    value = next(x.value for x in call.keywords if x.arg == "signature")
    require(isinstance(value, ast.Tuple) and len(value.elts) == 9, "actual public signature tuple shape")
    return dict(tuple_expression=ast.unparse(value), tuple_AST_sha256=digest(ast.dump(value, include_attributes=False).encode()),
        keys=["model_sha256", "gpu_uuid", "kv_layout_sha256", "kernel_mode", "active_decode", "batch",
            "prefill_tokens", "context_length", "physical_bytes"],
        prompt_contents_bound_in_signature=False, seed_bound_in_signature=False,
        workload_hash_bound_in_signature=False, selected_ordinal_bound_in_signature=False,
        external_temperature_GPU_clocks_SSD_page_cache_bound_in_signature=False)


def analyze(evidence):
    plan = evidence.json(CAL+"/NATIVE_COST_PLAN.json")
    measurements = evidence.json(CAL+"/NATIVE_PAIRED_MEASUREMENTS.json")
    verification = evidence.json(CAL+"/NATIVE_COST_VERIFICATION.json")
    protocol = evidence.json(D+"/PROTOCOL.json")
    summary = evidence.json(D+"/REPEATABILITY_SUMMARY.json")
    cal_raw = evidence.json("experiments/prefix_io_v1/runs/server12-c5-native-common-cost-gpu01/details/native-cost-runtime-result.json")
    prior = evidence.json(NORMAL+"/QUALIFICATION_off.json")
    public_source = evidence.bytes(NORMAL+"/p4_single_file_receipt.py").decode("utf-8")
    signature = signature_fields(public_source)
    event_sha = plan["cuda_event_source_ref"]["sha256"]
    original_code_audit = []
    for key in ("native_source_ref","collector_source_ref","cuda_event_source_ref"):
        source = canonical_ref(plan[key])
        if source["path"] in evidence.rows:
            require(evidence.rows[source["path"]] == source, "frozen original source reference")
            evidence.bytes(source["path"])
            checked = True
        else:
            # The archived verified plan supplies this explicit source ref.
            # The Event .py payload was excluded from the local tar. A live
            # server may read that exact ROOT-contained text file, without
            # importing Torch; an absent local payload remains unavailable.
            require(key == "cuda_event_source_ref", "required original source absent from delivery")
            candidate = evidence.root/source["path"]
            checked = candidate.is_file()
            if checked:
                candidate = contained_file(evidence.root,candidate)
                require(reference(source["path"],candidate.read_bytes()) == source, "actual Event source byte drift")
                evidence.reads[source["path"]] = (candidate,source)
        original_code_audit.append(dict(source_ref=source,payload_bytes_reverified_here=checked,
            source_metadata_and_all_witnesses_matched=True))
    condition = verification["condition"]
    calibration_guard = audit_guard(evidence,verification["evidence_refs"]["actual_guard"],
        plan["job_id"],condition["gpu_uuid"])
    require(condition["load"] == dict(active_decode=1,batch=1,prefill_tokens=0,context_length=144)
        and condition["existing_io"] == [dict(ops=0,bytes=0)]*4 and condition["physical_bytes"] == 917504
        and condition["operations"] == 1 and condition["stage"] == "ssd_read", "frozen finite cost condition")
    require(len(cal_raw["windows"]) == len(measurements["windows"]) == len(verification["windows"]) == 6,
        "six original calibration/heldout records")
    records, engines, cal_durations = [], [], {}
    for index, window in enumerate(cal_raw["windows"]):
        measure, checked = measurements["windows"][index], verification["windows"][index]
        require(measure["capture"] == window["capture"] and measure["prompt_token_ids"] == window["prompt_token_ids"]
            and measure["seed"] == window["seed"], "serialized vs actual calibration capture/workload")
        child = evidence.from_ref(measure["raw_child_ref"])
        require(len(child["windows"]) == 1 and child["gpu_uuid"] == condition["gpu_uuid"]
            and all(window.get(k) == v for k,v in child["windows"][0].items()), "actual calibration child")
        augmentation = {"native_journal", "native_post_shutdown", "native_tail_assertions", "subprocess_pid", "private_storage", "child_receipt_ref"}
        require(set(window)-set(child["windows"][0]) == augmentation
            and all(window[k] == child[k] for k in augmentation-{"child_receipt_ref"})
            and window["child_receipt_ref"] == measure["raw_child_ref"], "explicit parent calibration augmentation")
        audit = audit_capture(window["capture"], window, expected_event_sha=event_sha)
        io = audit_io(child, window, audit, action=window["condition"] == "B")
        value = audit["selected"]["gpu_elapsed_ns_from_witness"]
        require(checked["selected_gpu_elapsed_ns"] == value and checked["selected_load_context"] == 144,
            "original calibration selected witness value")
        require(measure["gpu_uuid"] == condition["gpu_uuid"] and measure["model_sha256"] == condition["model_sha256"]
            and measure["kv_layout_sha256"] == condition["kv_layout_sha256"], "calibration exact GPU/model/layout")
        directory = str(PurePosixPath(measure["raw_child_ref"]["path"]).parent)
        for suffix, content in (("-capture.json",window["capture"]),("-frontend.json",window["frontend"]),
            ("-window.json",child["windows"][0])):
            require(evidence.json(directory+"/"+window["request_id"]+suffix) == content,
                "inline/standalone calibration capture differs")
        engines.append(normalized_engine(child["engine_config"]))
        cal_durations[(window["pair_index"],window["condition"])] = value
        records.append(dict(record_group="original_calibration_or_holdout", pair_index=window["pair_index"],
            condition=window["condition"], split=window["split"], seed=window["seed"],
            first_prompt_token=window["prompt_token_ids"][0], workload_sha256=window["workload_sha256"],
            raw_child_ref=measure["raw_child_ref"], full_step_audit=audit, native_io_audit=io))
    require(all(config == engines[0] for config in engines), "same six nonstorage original engine configurations")
    frozen_upper = verification["calibration_predicted_upper_ns"]
    require(protocol["cost_upper_ns"] == summary["frozen_cost_upper_ns"] == frozen_upper == 16238752,
        "frozen upper unchanged")
    a_values = [cal_durations[(0,"A")], cal_durations[(1,"A")]]
    budget = max(a_values)
    require(protocol["step_budget_ns"] == summary["frozen_a_only_budget_ns"] == budget == 13171328,
        "A-only maximum budget unchanged")
    baseline = verification["calibration_only_cost"]["baseline_ns"]
    increment = verification["calibration_only_cost"]["incremental_or_joint_ns"]
    uncertainty = verification["calibration_only_cost"]["uncertainty_ns"]
    require(baseline == (sum(a_values)+1)//2, "stored calibration A baseline")
    require(all(type(v) is int for v in (baseline,increment,uncertainty)) and
        frozen_upper == baseline+increment+uncertainty, "unchanged stored estimator arithmetic identity")
    # This is an identity check on already frozen numbers; no new fit is made.
    require(protocol["repetitions"] == 3 and len(summary["records"]) == 3
        and protocol["decision_rule"]["no_refit"] is True
        and summary["calibration_refit"] is False and summary["thresholds_changed"] is False,
        "fixed-three/no-refit protocol")
    diagnostics = []
    prompt = [protocol["prompt_first_token"]]+list(range(protocol["prompt_tail_first"],protocol["prompt_tail_last"]+1))
    for index, checked in enumerate(summary["records"]):
        require(type(checked["diagnostic_index"]) is int and checked["diagnostic_index"] == index, "fixed ordered three indices")
        raw = evidence.from_ref(checked["evidence_refs"]["raw_result"])
        require(raw["diagnostic_index"] == index and raw["protocol_ref"] == summary["protocol_ref"]
            and raw["label"] == protocol["jobs"][index]["label"] and raw["mode"] == "off", "exact new diagnostic identity")
        config = evidence.from_ref(checked["evidence_refs"]["config"])
        require(config["diagnostic_index"] == index and config["protocol_ref"] == summary["protocol_ref"], "actual diagnostic config identity")
        require(len(raw["windows"]) == 1, "one B window per new diagnostic")
        window = raw["windows"][0]
        require(window["prompt_token_ids"] == prompt and window["seed"] == protocol["seed"]
            and window["condition"] == "B", "fixed actual fourth prompt/seed")
        audit = audit_capture(window["capture"], window, expected_event_sha=event_sha)
        io = audit_io(raw, window, audit, action=True)
        for suffix, content in (("-capture.json",window["capture"]),("-frontend.json",window["frontend"]),("-window.json",window)):
            identity = "experiments/prefix_io_v1/runs/"+raw["label"]+"/details/"+window["request_id"]+suffix
            require(evidence.json(identity) == content, "inline/standalone actual window evidence differs")
        analysis = checked["original_analysis_result"]
        guard_audit = audit_guard(evidence,checked["evidence_refs"]["actual_guard"],raw["label"],condition["gpu_uuid"])
        proof_refs = dict(before=checked["evidence_refs"]["source_before"],
            launch=checked["source_launch_ref"],after=checked["evidence_refs"]["source_after"])
        proof_rows = {phase:evidence.from_ref(row) for phase,row in proof_refs.items()}
        require(all(doc["phase"] == phase and doc["failed"] == []
            and doc["source_lock_ref"] == analysis["source_lock_ref"] for phase,doc in proof_rows.items()),
            "saved diagnostic full source proof domain")
        values = [f["gpu_elapsed_ns_from_witness"] for f in audit["all_saved_steps"]]
        require(values == analysis["all_step_gpu_elapsed_ns"] and values[16] == checked["selected_gpu_elapsed_ns"]
            == analysis["selected_gpu_elapsed_ns"] and sum(values) == analysis["sum_full_step_gpu_ns"], "original complete-step/selected aggregation")
        signature_prefix = raw["verified_runtime_identity"]["signature_prefix"]
        require(signature_prefix[:3] == [condition["model_sha256"],condition["gpu_uuid"],condition["kv_layout_sha256"]]
            and raw["gpu_uuid"] == condition["gpu_uuid"] and normalized_engine(raw["engine_config"]) == engines[0],
            "same actual model/GPU/layout/kernel and original nonstorage engine")
        require((values[16] <= frozen_upper) is checked["covered_original_upper"] and
            checked["upper_exceedance_ns"] == values[16]-frozen_upper, "frozen signed threshold decision unchanged")
        diagnostics.append(dict(record_group="fixed_new_three_only", diagnostic_index=index,label=raw["label"],
            seed=window["seed"], first_prompt_token=prompt[0], workload_sha256=window["workload_sha256"],
            selected_gpu_elapsed_ns=values[16], covered_frozen_upper=values[16]<=frozen_upper,
            signed_error_vs_frozen_upper_ns=values[16]-frozen_upper,
            exceeds_A_only_budget=values[16]>budget, full_step_audit=audit, native_io_audit=io,
            saved_guard_audit=guard_audit, saved_source_proof_refs=proof_refs))
    prior_raw = evidence.from_ref(protocol["immutable_prior_off_counterexample"]["raw_result_ref"])
    prior_window = prior_raw["windows"][0]
    prior_audit = audit_capture(prior_window["capture"],prior_window,expected_event_sha=event_sha)
    require(prior_audit["selected"]["gpu_elapsed_ns_from_witness"] == prior["selected_gpu_elapsed_ns"]
        == protocol["immutable_prior_off_counterexample"]["selected_gpu_elapsed_ns"], "old counterexample selected witness")
    prior_io = audit_io(prior_raw,prior_window,prior_audit,action=True)
    prior_guard = audit_guard(evidence,protocol["immutable_prior_off_counterexample"]["guard_ref"],
        prior_raw["label"],condition["gpu_uuid"])
    return dict(frozen_condition=condition, public_signature_source_audit=signature,
        original_code_audit=original_code_audit,
        original_six_records=records, calibration_guard_audit=calibration_guard,fixed_new_three_records=diagnostics,
        old_off_counterexample_separate=dict(selected_gpu_elapsed_ns=prior["selected_gpu_elapsed_ns"],
            first_prompt_token=prior_window["prompt_token_ids"][0], seed=prior_window["seed"],
            full_step_audit=prior_audit,native_io_audit=prior_io,saved_guard_audit=prior_guard,pooled_as_fourth_repeat=False),
        frozen_math=dict(total_step_cost_upper_ns=frozen_upper,total_step_budget_ns=budget,
            calibration_A_values_ns=a_values,baseline_ns=baseline,paired_increment_ns=increment,
            positive_residual_uncertainty_ns=uncertainty,required_increment_and_uncertainty_ns=increment+uncertainty,
            budget_source="maximum_of_two_frozen_calibration_A_steps",
            independently_declared_service_SLO=False,upper_budget_same_domain=True,
            cost_domain="complete GPU execute_model through sample_tokens current-stream Event.elapsed_time ns",
            headroom_ns=budget-baseline,shortfall_ns=frozen_upper-budget,ordinary_cost_admission_fits=False,
            estimator_refit=False,heldout_reused_to_fit=False),
        observed_differences=dict(calibration_prompt_first_tokens=sorted({x["first_prompt_token"] for x in records}),
            calibration_seeds=sorted({x["seed"] for x in records}),new_prompt_first_token=prompt[0],new_seed=protocol["seed"],
            prompt_seed_not_bound_in_cost_condition=True,same_prepared_selected_load=True,same_nonstorage_engine_configuration=True,
            all_saved_outputs_have_same_token_ID_hash=len({x["full_step_audit"]["output_token_ids_sha256"]
                for x in records+diagnostics}) == 1,
            wrapper_revisions_differ=True,causal_explanation_established=False),
        actual_shared_engine_fields={key:engines[0].get(key) for key in ("dtype","kv_cache_dtype","quantization",
            "enforce_eager","compilation_config","attention_config","tensor_parallel_size","async_scheduling",
            "max_num_seqs","block_size","kv_cache_memory_bytes")},
        next_GPU_decision=dict(repeating_same_configuration_changes_frozen_admission=False,
            current_shadow_on_remain_blocked=True,recalibration_guarantees_smaller_cost_or_admission=False,
            changing_budget_to_pass_allowed=False,causal_root_cause_from_current_records_known=False,
            independent_new_prospective_protocol_required_for_any_new_calibration=True),
        limits=["Full saved steps have different context lengths; only offset16/context144 is the frozen cost cell.",
            "Offset0 is prefill; offset1 is the declared excluded warmup; remaining offsets do not form extra fitted samples.",
            "Prompt/seed equality is checked within each planned workload, but not a field of the public cost signature.",
            "Temperature, GPU clocks, SSD/page-cache state and unobserved concurrent load cannot be inferred from witness durations.",
            "No GPU-host absolute clock mapping or completed GPU release credit is inferred from native I/O completion.",
            "Integrity and field checks here do not independently reissue the original native receipt or replay its full guard/ledger qualification.",
            "Repeating the same configuration alone cannot change frozen U>budget. Recalibration has no guaranteed numerical outcome and cannot be used here to refit away these failures."])


def default_manifest(root, server_name, local_name):
    for candidate in (root/A/server_name, root/LOCAL/local_name):
        if candidate.is_file():
            return candidate
    raise AuditRejected("delivery manifest unavailable: "+server_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-manifest", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--prior-manifest", type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root/args.output
    require(output.parent.resolve().is_relative_to(root) and not output.exists() and not output.is_symlink(),
        "append-only output must be new and inside project")
    evidence = Evidence(root)
    own_before = Path(__file__).read_bytes()
    flags = dict(GPU_operations=0,models_imported=0,shared_objects_loaded=0,RPC_calls=0,
        new_receipt_issued=False,native_execution_qualification_issued=False,calibration_refit=False,
        thresholds_changed=False,normal_qualification_passed=False,P4_strategy_effect_verified=False,
        performance_benefit_proved=False,permits_next_mode=None)
    try:
        for filename in (args.local_manifest or default_manifest(root,"REPEAT_REPEAT_DELIVERY_fixed3-final_MANIFEST.json","repeat_fixed3_verified/LOCAL_REPEAT_DELIVERY_BYTE_VERIFICATION.json"),
            args.calibration_manifest or default_manifest(root,"GPU_DELIVERY_MANIFEST.json","LOCAL_GPU_EVIDENCE_BYTE_VERIFICATION.json"),
            args.prior_manifest or default_manifest(root,"NORMAL_NORMAL_DELIVERY_off01-final_MANIFEST.json","normal_off01_verified/LOCAL_NORMAL_DELIVERY_BYTE_VERIFICATION.json")):
            evidence.add_manifest(filename)
        analysis = analyze(evidence)
        inputs_after = evidence.recheck()
        require(Path(__file__).read_bytes() == own_before, "audit source changed")
        require(not any(name.split(".")[0] in {"torch","vllm","py_kvcache"} for name in sys.modules), "forbidden model import")
        result = dict(schema="c5_fixed_cost_evidence_field_audit_cpu_v1",status="PASS_READONLY_CPU_FIELD_AUDIT",
            origin="stdlib_readonly_actual_archived_field_and_byte_audit",analysis=analysis,
            manifests=evidence.manifests,input_refs_before=inputs_after,input_refs_after=inputs_after,
            source_ref=reference(str(Path(__file__).resolve()),own_before),forbidden_imports=[],**flags)
        code = 0
    except (AuditRejected,ValueError,KeyError,TypeError,IndexError,OSError) as exc:
        result = dict(schema="c5_fixed_cost_evidence_field_audit_cpu_v1",status="REJECTED_CPU_FIELD_AUDIT",
            error=str(exc),source_ref=reference(str(Path(__file__).resolve()),own_before),**flags)
        code = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode("utf-8"))
    print(json.dumps(dict(status=result["status"],output=str(output),bytes=output.stat().st_size,
        sha256=digest(output.read_bytes()),GPU_operations=0),ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
