"""Strict, finite native evidence bridge to the frozen original paired math.

This does not install a hook, launch CUDA, issue I/O, publish a production table,
or infer resource-release credit. Native origin text alone grants no authority.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import importlib.util
import sys
from types import SimpleNamespace


ORIGINAL_BYTES = 48136
ORIGINAL_SHA256 = "3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac"
STAGES = ("ssd_read", "ssd_write", "h2d", "d2h")
FORMULA = "ceil_calibration_means_plus_max_positive_residual_v1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + " must be integer")
    return value


def digest(value, label):
    require(type(value) is str and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), label + " must be SHA-256")
    return value


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode()).hexdigest()


def keys(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), label + " exact keys")


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_mapping(cls, value):
        keys(value, ("path", "bytes", "sha256"), "evidence reference")
        integer(value["bytes"], "evidence bytes", 1)
        digest(value["sha256"], "evidence SHA")
        return cls(**value)

    def read(self, project):
        require(type(self.path) is str and "\\" not in self.path and
                ":" not in self.path and "\0" not in self.path, "relative POSIX path")
        pure = PurePosixPath(self.path)
        require(not pure.is_absolute() and
                all(p not in ("", ".", "..") for p in self.path.split("/")), "bounded evidence path")
        project = Path(project).resolve(strict=True)
        path = project
        for part in pure.parts:
            path = path / part
            require(not path.is_symlink(), "symlink evidence forbidden")
        require(path.is_file() and project in path.resolve().parents and
                0 < self.bytes <= 32 * 1024**2 and path.stat().st_size == self.bytes,
                "bounded regular evidence file")
        raw = path.read_bytes()
        require(len(raw) == self.bytes and sha256(raw).hexdigest() == self.sha256,
                "evidence bytes/SHA drift")
        return raw

    def json(self, project):
        return json.loads(self.read(project), object_pairs_hook=_pairs,
            parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def original_estimator(source_path):
    """Compile exact original duration helpers and numerical AST, unchanged.

    Native acquisition/causality is checked separately. This deliberately does
    not re-label the old CPU fixture builder or bypass its production loader.
    """
    path = Path(source_path)
    require(path.is_file() and not path.is_symlink(), "regular original estimator source")
    raw = path.read_bytes()
    require((len(raw), sha256(raw).hexdigest()) == (ORIGINAL_BYTES, ORIGINAL_SHA256),
            "frozen original estimator drift")
    tree = ast.parse(raw, filename=str(path))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    body = functions["_verify_cell"].body
    first = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and
                 isinstance(n.targets[0], ast.Name) and n.targets[0].id == "pair_means")
    last = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and
                isinstance(n.targets[0], ast.Name) and n.targets[0].id == "residual")
    require(last > first, "exact original numerical AST span")
    selected = deepcopy(body[first:last + 1])
    template = ast.parse("def calculate(base, action, baseline_runs, by_split, plan):\n    pass\n").body[0]
    template.body = selected + [ast.parse("return dict(baseline_ns=baseline_ns, "
        "incremental_or_joint_ns=incremental, uncertainty_ns=residual, pair_means=pair_means, "
        "measured_windows=measured, warmup_windows=warmup)").body[0]]
    nodes = [deepcopy(functions[name]) for name in ("_ceil_mean", "_v2_timing", "_duration_ns")]
    namespace = dict(_require=require, _keys=keys, _integer=integer, EvidenceRef=EvidenceRef)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes + [template], type_ignores=[])),
                 str(path) + "::unchanged_numerical_AST", "exec", dont_inherit=True), namespace)
    proof = dict(source_bytes=ORIGINAL_BYTES, source_sha256=ORIGINAL_SHA256,
        formula=FORMULA, numerical_ast_sha256=sha256(ast.dump(ast.Module(body=selected,
            type_ignores=[]), include_attributes=False).encode()).hexdigest(),
        mathematical_expressions_changed=False)
    return namespace["calculate"], proof


def validate_capture(capture, *, run_id, request_id, output_ids, prompt_tokens,
                     measured_offset, warmup_offsets, cached_tokens=128):
    require(capture.get("scope") == "server11_full_step_native_capture_v1" and
            capture.get("run_id") == run_id and capture.get("origin") == "native_gpu_recording" and
            capture.get("valid") is True, "actual full-step capture identity/validity")
    frames, witnesses = capture.get("frames"), capture.get("event_witnesses")
    require(type(frames) is list and type(witnesses) is list and len(frames) == len(witnesses) == 128,
            "all 128 original steps and CUDA witnesses required")
    require(type(output_ids) is list and len(output_ids) == 128, "actual full 128 output tokens")
    for token in output_ids:
        integer(token, "actual output token")
    integer(prompt_tokens, "actual prompt tokens", 1)
    require(type(cached_tokens) is int and 0 <= cached_tokens < prompt_tokens, "frozen cached prompt amount")
    require(capture.get("failures") == [] and capture.get("pending_event_pairs") == 0 and
            capture.get("open_event_pair") is False and capture.get("no_added_synchronization") is True and
            capture.get("cross_clock_absolute_mapping") is False and
            capture.get("selected_offsets") == [measured_offset], "complete query-only collector tail")
    require(type(warmup_offsets) is list and warmup_offsets == sorted(set(warmup_offsets)) and
            warmup_offsets and all(type(x) is int and 0 < x < measured_offset for x in warmup_offsets)
            and type(measured_offset) is int and measured_offset < 128, "frozen decode offsets")
    first = frames[0]["native_step_ordinal"]
    integer(first, "first actual native ordinal")
    result = []
    for offset, (frame, witness) in enumerate(zip(frames, witnesses)):
        ordinal = first + offset
        require(frame["native_step_ordinal"] == witness["native_step_ordinal"] == ordinal,
                "contiguous full original step ordinals")
        begin, end = frame["start_ns"], frame["end_ns"]
        integer(begin, "original host start", 1)
        require(integer(end, "original host end", 1) > begin and
                (offset == 0 or begin >= frames[offset - 1]["end_ns"]), "ordered complete host frames")
        require(frame.get("intended_timing_scope") == "full_decode_step" and
                frame.get("gpu_elapsed_ns") is None and frame.get("existing_io") is None and
                frame.get("new_io") is None, "original ClosedFrame must stay unpromoted")
        prepared = frame["prepared"]
        context = cached_tokens if offset == 0 else prompt_tokens + offset - 1
        expected = dict(native_step_ordinal=ordinal, batch=1, active_decode=0 if offset == 0 else 1,
            prefill_tokens=prompt_tokens - cached_tokens if offset == 0 else 0, context_length=context,
            step_kind="prefill" if offset == 0 else "decode", context_basis="pre_computed_tokens")
        require(all(prepared.get(k) == v and type(prepared.get(k)) is type(v) for k, v in expected.items()),
                "actual complete cold/decode load differs")
        require(prepared["rows"] == [dict(request_id=request_id, pre_context=context,
            prompt_tokens=prompt_tokens, scheduled_tokens=prompt_tokens - cached_tokens if offset == 0 else 1)] and
            prepared["input_seq_lens_from_cpu_inputs"] == [prompt_tokens if offset == 0 else context + 1],
            "copied original prepared cohort/load mismatch")
        require(frame["outputs"] == [[request_id, [output_ids[offset]]]], "all actual sampled output IDs")
        names = ("start_record_before_ns", "start_record_after_ns", "start_completed_query_ns",
                 "end_record_before_ns", "end_record_after_ns", "end_completed_query_ns")
        values = {name: integer(witness[name], name, 1) for name in names}
        require(values["start_record_before_ns"] <= values["start_record_after_ns"] <=
                values["start_completed_query_ns"] and values["start_record_after_ns"] <= begin and
                end <= values["end_record_before_ns"] <= values["end_record_after_ns"] <=
                values["end_completed_query_ns"], "CUDA record/query causal bounds")
        require(witness.get("event_elapsed_source") == "torch.cuda.Event.elapsed_time",
                "actual CUDA elapsed source required")
        duration = integer(witness["gpu_elapsed_ns"], "actual CUDA elapsed ns", 1)
        require(duration <= values["end_completed_query_ns"] - values["start_record_before_ns"],
                "CUDA elapsed exceeds physical host enclosure")
        result.append(dict(step_offset=offset, native_step_ordinal=ordinal, gpu_elapsed_ns=duration,
                           host_start_ns=begin, host_end_ns=end, witness=witness))
    return result


def _amount(value):
    require(type(value) is dict and set(value) in ({"ops", "nbytes"}, {"ops", "bytes"}),
            "original physical amount scalar")
    return (integer(value["ops"], "physical operations"),
            integer(value.get("nbytes", value.get("bytes")), "physical bytes"))


def validate_io(journal, drain, *, capture, frames, run_id, native_source_sha256,
                arm, measured_offset, physical_bytes, operations, independent_payload):
    require(type(journal) is dict and journal.get("valid") is True and
            journal.get("lost", 0) == 0 and type(journal.get("events")) is list and
            type(journal.get("frames")) is list, "complete native owner journal required")
    events, owners = journal["events"], journal["frames"]
    require(len(events) <= 4096 and 2 <= len(owners) <= 4096, "bounded owner coverage")
    require(drain.get("run_id") == run_id and drain.get("native_source_sha256") == native_source_sha256 and
            drain.get("boundary") == "after_original_handler_shutdown", "actual original drain identity")
    for name in ("accounting_valid", "worker_joined", "ring_closed", "ring_drained"):
        require(drain.get(name) is True, "actual drain field: " + name)
    for name in ("outstanding_records", "ring_outstanding", "pending_copies", "active_native",
                 "inflight_parents", "observation_failures"):
        require(type(drain.get(name)) is int and drain[name] == 0, "native owner remains: " + name)
    accepted, completed = {}, {}
    last_time = 0
    for seq, event in enumerate(events, 1):
        require(event.get("sequence") == seq and event.get("kind") in ("accepted", "completed") and
                event.get("stage") in STAGES, "complete ordered owner events")
        now = integer(event["at_ns"], "owner timestamp", 1)
        require(now >= last_time, "owner clock regression")
        last_time = now
        op = integer(event["operation_sequence"], "owner operation", 1)
        integer(event["physical_bytes"], "owner physical bytes", 1)
        if event["kind"] == "accepted":
            require(op == seq and op not in accepted, "unique owner acceptance")
            accepted[op] = event
        else:
            require(op in accepted and op not in completed and
                    event["stage"] == accepted[op]["stage"] and
                    event["physical_bytes"] == accepted[op]["physical_bytes"] and
                    type(event["result"]) is int and event["result"] == event["physical_bytes"],
                    "actual CQE result/bytes must exactly match accepted I/O")
            completed[op] = event
    require(set(accepted) == set(completed), "actual accepted I/O incomplete")
    totals = []
    for stage in STAGES:
        selected = [e for e in accepted.values() if e["stage"] == stage]
        totals.append((len(selected), sum(e["physical_bytes"] for e in selected)))
    require([_amount(x) for x in drain["accepted"]] == totals ==
            [_amount(x) for x in drain["completed"]] and drain["failed_ops"] == [0] * 4 and
            drain["transferred_bytes"] == [x[1] for x in totals], "journal/actual drain totals differ")
    for owner in owners:
        require(owner.get("run_id") == run_id and owner.get("source_sha256") == native_source_sha256 and
                owner.get("clock_domain") == "monotonic_ns" and owner.get("valid") is True and
                owner.get("journal_complete") is True and owner.get("failed_ops") == [0] * 4,
                "original owner frame identity/completeness")
    require(all(a["captured_ns"] <= b["captured_ns"] for a, b in zip(owners, owners[1:])),
            "owner frame clock ordering")
    # Validate the shared job's original accounting snapshots, then project a
    # window using real timestamps. All six windows may share this same journal.
    for owner in owners:
        seq = integer(owner["sequence"], "owner frame sequence")
        require(seq <= len(events), "owner snapshot beyond journal")
        accrued = [e for e in accepted.values() if e["sequence"] <= seq]
        finished = [e for e in completed.values() if e["sequence"] <= seq]
        for i, stage in enumerate(STAGES):
            a = [e for e in accrued if e["stage"] == stage]
            c = [e for e in finished if e["stage"] == stage]
            a_amount = (len(a), sum(e["physical_bytes"] for e in a))
            c_amount = (len(c), sum(e["physical_bytes"] for e in c))
            require(_amount(owner["accepted"][i]) == a_amount and
                    _amount(owner["completed"][i]) == c_amount and
                    _amount(owner["inflight"][i]) == tuple(a_amount[n] - c_amount[n] for n in range(2)),
                    "original frame counters disagree with owner event journal")
    first_start = frames[0]["witness"]["start_record_before_ns"]
    last_end = frames[-1]["witness"]["end_record_after_ns"]
    require(not any(e["at_ns"] in (first_start, last_end) for e in events),
            "native event on window boundary is ambiguous")
    prior = [o for o in owners if o["captured_ns"] <= first_start]
    after = [o for o in owners if o["captured_ns"] >= last_end]
    require(prior and after and all(_amount(x) == (0, 0) for x in prior[-1]["inflight"]) and
            all(_amount(x) == (0, 0) for x in after[0]["inflight"]), "zero initial/final owner enclosure")
    accepted = {op: e for op, e in accepted.items() if first_start < e["at_ns"] < last_end}
    totals = [(sum(e["stage"] == stage for e in accepted.values()),
               sum(e["physical_bytes"] for e in accepted.values() if e["stage"] == stage)) for stage in STAGES]
    actions = capture.get("actions")
    require(type(actions) is list, "explicit actual trigger list")
    if arm == "baseline":
        require(not accepted and len(actions) == 1 and actions[0].get("action_enabled") is False and
                actions[0].get("trigger_result") is None and independent_payload is None,
                "no-I/O baseline must have zero actual native operations")
    else:
        require(arm == "action" and len(actions) == 1 and len(accepted) == operations and
                totals == [(operations, physical_bytes), (0, 0), (0, 0), (0, 0)],
                "one finite actual SSD-only preload required")
        action = actions[0]
        window = frames[measured_offset]
        witness = window["witness"]
        require(action.get("action_enabled") is True and action["step_offset"] == measured_offset and
                action["native_step_ordinal"] == window["native_step_ordinal"] and
                witness["start_completed_query_ns"] < action["trigger_before_ns"] <=
                action["trigger_after_ns"] < witness["end_record_before_ns"], "selected trigger causal enclosure")
        require(all(action["trigger_before_ns"] <= e["at_ns"] < witness["end_record_before_ns"]
                    and witness["start_completed_query_ns"] < e["at_ns"] and
                    window["host_start_ns"] < e["at_ns"] < window["host_end_ns"]
                    for e in accepted.values()),
                "SSD acceptance not certainly inside actual CUDA interval")
        require(type(independent_payload) is dict and independent_payload.get("ssd_only") is True and
                independent_payload.get("cache_miss_before_submit") is True and
                independent_payload.get("physical_bytes") == physical_bytes and
                independent_payload.get("operations") == operations,
                "actual distinct native SSD preload payload required")
        payload_keys = independent_payload.get("preload_key_sha256")
        request_keys = independent_payload.get("request_prefix_key_sha256")
        require(type(payload_keys) is list and type(request_keys) is list and len(payload_keys) == operations and
                len(set(payload_keys)) == operations and request_keys and
                not set(payload_keys) & set(request_keys), "preload must be independent from request prefix")
        for value in payload_keys + request_keys:
            digest(value, "actual native key")
    return dict(stage="ssd_read", accepted_operations=totals[0][0],
                accepted_physical_bytes=totals[0][1], accepted_causal_overlap_verified=arm == "action",
                native_bytes_completed=True, resource_release_credit=False)


def validate_guard(guard, *, gpu_uuid, job_id, wrapper_path):
    require(guard.get("label") == job_id and guard.get("gpu_uuid") == gpu_uuid and
            guard.get("gpu_job_attempted") is True and
            type(guard.get("exit")) is int and guard["exit"] == 0 and
            type(guard.get("child_exit")) is int and guard["child_exit"] == 0 and
            guard.get("timed_out") is False and guard.get("interrupted_signal") is None and
            guard.get("error") is None and guard.get("session_drained") is True and
            guard.get("session_members_before_cleanup") == [] and
            guard.get("session_members_after_cleanup") == [], "actual GPU guard/natural session drain failed")
    command = guard.get("command")
    require(type(command) is list and wrapper_path in command and "--execute" in command,
            "actual guard did not launch frozen native wrapper")
    require(type(guard.get("reservation_id")) is str and guard["reservation_id"] and
            type(guard.get("session_id")) is int and guard["session_id"] > 0,
            "actual budget reservation/OS session missing")
    return True


def original_post_shutdown_drain(post, tail_assertions, *, run_id, native_source_sha256):
    """Project the pinned original successful shutdown snapshot, never infer it.

    Boolean closure fields are read by the new thin caller from the actual
    original objects after the unchanged post_shutdown_snapshot has returned.
    """
    require(type(tail_assertions) is dict and tail_assertions.get("worker_alive") is False and
            tail_assertions.get("aio_worker_alive") is False and
            tail_assertions.get("handler_shutdown") is True and
            tail_assertions.get("reactor_closed") is True, "actual original shutdown tail fields")
    snapshot = post["snapshot"]
    require(snapshot.get("native_shutdown_read") is True and snapshot.get("owner_capture") is False and
            snapshot.get("physical_drain_inferred") is False and snapshot.get("gpu_release_credit") is False,
            "original shutdown snapshot boundary/release flags")
    native = snapshot["native"]
    require(native == post["actual_native"] and set(native) == {"active_parents", "ring_ops", "pending_copies",
            "copy_ready", "ready_load_fds", "ready_preload_fds"} and
            all(type(x) is int and x == 0 for x in native.values()), "actual original native drain")
    require(type(post.get("actual_handler_active")) is int and post["actual_handler_active"] == 0 and
            type(post.get("observation_failures")) is int and post["observation_failures"] == 0,
            "original handler/observer remains")
    aio = snapshot["aio"]
    require(all(aio.get(k) == v and type(aio.get(k)) is type(v) for k, v in post["actual_aio"].items()) and
            aio.get("closed") is True and aio.get("drained") is True and aio.get("fatal") is None,
            "original actual AIO closure")
    require(all(type(aio.get(k)) is int and aio[k] == 0 for k in ("outstanding", "pending", "ready", "unreaped")) and
            integer(aio["accepted"], "actual AIO accepts") == integer(aio["completed"], "actual AIO complete") ==
            integer(aio["reaped"], "actual AIO reap"), "original AIO incomplete")
    parents = snapshot["parent_admission"]
    require(parents == snapshot["admission"] and parents.get("count_valid") is True and
            parents.get("native_drain_unknown") is False and parents.get("fatal_reason") is None and
            parents.get("source_gpu_reuse_inferred") is False and parents.get("staging_release_inferred") is False and
            all(type(parents.get(k)) is int and parents[k] == 0 for k in
                ("accepted_parents", "accepted_count_lower_bound", "waiting_submitters", "failure_stream_syncs")),
            "original parent drain/no sync fallback/no release inference")
    accounting = snapshot["stage_accounting"]
    require(type(accounting) is dict and accounting.get("valid") is True and accounting.get("bound") is True and
            accounting.get("error") is None and accounting.get("resource_release_inferred") is False and
            type(accounting.get("outstanding_records")) is int and accounting["outstanding_records"] == 0 and
            accounting.get("stages") == post["actual_stage_counters"], "actual original stage accounting tail")
    stages = accounting["stages"]
    require(set(stages) == set(STAGES), "original complete four-stage counters")
    accepted, completed, transferred, failed = [], [], [], []
    for stage in STAGES:
        row = stages[stage]
        require(all(type(row[k]) is int and row[k] == 0 for k in ("inflight_ops", "inflight_bytes", "failed_ops")),
                "original physical stage incomplete/failed")
        accepted.append(dict(ops=integer(row["accepted_ops"], "accepted ops"),
                             nbytes=integer(row["accepted_bytes"], "accepted bytes")))
        completed.append(dict(ops=integer(row["completed_ops"], "completed ops"),
                              nbytes=integer(row["completed_requested_bytes"], "completed bytes")))
        transferred.append(integer(row["transferred_bytes"], "transferred bytes"))
        failed.append(row["failed_ops"])
    return dict(run_id=run_id, native_source_sha256=native_source_sha256,
        boundary="after_original_handler_shutdown", accepted=accepted, completed=completed,
        transferred_bytes=transferred, failed_ops=failed, accounting_valid=True, outstanding_records=0,
        worker_joined=True, ring_closed=True, ring_drained=True, ring_outstanding=0, pending_copies=0,
        active_native=0, inflight_parents=0, observation_failures=0)


def analyze_paired(plan, windows, *, journal=None, drain=None, original_source_path):
    """Validate native shapes and reuse original math; no GPU attestation here.

    `verify_native_cell` adds independently expected on-disk execution evidence.
    Six windows may be sequential requests of one normally closed engine/job.
    """
    require(plan.get("scope") == "server11_preregistered_native_conditional_cell_v1" and
            plan.get("qualification_rule") == "zero_observed_holdout_underprediction_no_refit_v1",
            "pre-registered finite conditional qualification required")
    require(plan.get("stage") == "ssd_read" and plan.get("units") == 1 and
            type(plan.get("units")) is int, "finite original SSD geometry")
    physical = integer(plan["transfer_quantum_bytes"], "frozen transfer quantum", 1) * plan["units"]
    operations = integer(plan["operations"], "frozen operations", 1)
    require(operations == 1, "single-file original SSD operation required")
    require(plan.get("cached_prompt_tokens") == 128 and plan.get("prompt_tokens") == 129 and
            plan.get("output_tokens") == 128 and plan.get("measured_offset") == 16 and
            plan.get("warmup_offsets") == [1], "one frozen full-output load and selection only")
    entries = plan["entries"]
    require(type(entries) is list and len(entries) == 3 and type(windows) is list and len(windows) == 6,
            "exactly two calibration pairs and one independent held-out pair")
    require([e["split"] for e in entries] == ["calibration", "calibration", "validation"] and
            [e["arm_order"] for e in entries] == ["AB", "BA", "AB"] and
            len({e["pair_id"] for e in entries}) == 3, "frozen AB/BA calibration and independent AB validation")
    for entry in entries:
        ids = entry["prompt_token_ids"]
        require(type(ids) is list and len(ids) == 129, "full actual frozen prompt")
        for token in ids:
            integer(token, "frozen prompt token")
        integer(entry["seed"], "frozen seed")
        require(entry["prompt_sha256"] == canonical_hash(ids) and
                entry["prefix_family_sha256"] == canonical_hash(ids[:16]) and
                entry["trace_sha256"] == canonical_hash(dict(prompt_token_ids=ids,
                    seed=entry["seed"], output_tokens=128)) and
                entry["workload_sha256"] == canonical_hash(dict(prompt_token_ids=ids,
                    output_tokens=128, temperature=0, ignore_eos=True)),
                "independently recomputed prompt/prefix/workload/trace pins")
    for field in ("trace_sha256", "prefix_family_sha256", "workload_sha256"):
        require(entries[2][field] not in {e[field] for e in entries[:2]}, "heldout " + field + " leakage")
    require(len({(e["trace_sha256"], e["seed"]) for e in entries}) == 3,
            "independent trace/seed pair identity")
    expected_order = [(entry, arm) for entry in entries for arm in
                      (("baseline", "action") if entry["arm_order"] == "AB" else ("action", "baseline"))]
    collector_ref = EvidenceRef.from_mapping(plan["collector_source_ref"])
    timing_contract = dict(timing_scope="full_decode_step", clock_domain="cuda_event_elapsed",
        reference_source_ref=collector_ref, reference_valid=True, fallback_used=False, clock_domain_valid=True)
    base, action, runs, groups, verified, interval_end = {}, {}, {}, {"calibration": [], "validation": []}, [], 0
    outputs = {}
    all_request_ids = set()
    for window, (entry, arm) in zip(windows, expected_order):
        require(window.get("pair_id") == entry["pair_id"] and window.get("arm") == arm and
                window.get("seed") == entry["seed"] and window.get("prompt_token_ids") == entry["prompt_token_ids"],
                "actual window differs from frozen pair/order/prompt/seed")
        require(window.get("gpu_uuid") == plan["gpu_uuid"] and
                window.get("source_lock_sha256") == plan["source_lock_ref"]["sha256"] and
                window.get("model_sha256") == plan["model_sha256"] and
                window.get("kv_layout_sha256") == plan["kv_layout_sha256"], "actual window machine/source/model/layout pins")
        require(type(window.get("request_id")) is str and window["request_id"] and
                window["request_id"] not in all_request_ids, "independent native request identity")
        all_request_ids.add(window["request_id"])
        capture = window["capture"]
        event_source = capture.get("event_source", {})
        event_ref = EvidenceRef.from_mapping(plan["cuda_event_source_ref"])
        require(event_source.get("origin") == "native_gpu_recording" and
                event_source.get("module") == "torch.cuda.streams" and event_source.get("class_name") == "Event" and
                event_source.get("sha256") == event_ref.sha256 and event_source.get("bytes") == event_ref.bytes and
                event_source.get("path") == plan["project_root"].rstrip("/") + "/" + event_ref.path,
                "installed actual CUDA Event source pin")
        frames = validate_capture(capture, run_id=window["run_id"], request_id=window["request_id"],
            output_ids=window["output_token_ids"], prompt_tokens=129, measured_offset=16,
            warmup_offsets=[1], cached_tokens=128)
        require(frames[0]["witness"]["start_record_before_ns"] > interval_end,
                "independent request windows overlap/order differs")
        interval_end = frames[-1]["witness"]["end_record_after_ns"]
        actual_journal = window.get("native_journal", journal)
        actual_drain = window.get("final_drain", drain)
        if "native_post_shutdown" in window:
            actual_drain = original_post_shutdown_drain(window["native_post_shutdown"], window["native_tail_assertions"],
                run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_sha256"])
        io = validate_io(actual_journal, actual_drain, capture=capture, frames=frames,
            run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_sha256"],
            arm=arm, measured_offset=16, physical_bytes=physical, operations=operations,
            independent_payload=window["independent_payload"])
        for offset, phase in ((1, "warmup"), (16, "measured")):
            row = frames[offset]
            timing = dict(timing_contract, reference_source_ref=plan["collector_source_ref"],
                          gpu_elapsed_ns=row["gpu_elapsed_ns"])
            target = base if arm == "baseline" else action
            target[(entry["pair_id"], "step-" + str(offset), phase)] = dict(
                output_tokens=1, timing=timing)
        if arm == "baseline":
            runs[entry["pair_id"]] = dict(pair_id=entry["pair_id"])
            groups[entry["split"]].append(dict(pair_id=entry["pair_id"]))
        pair_outputs = outputs.setdefault(entry["pair_id"], {})
        pair_outputs[arm] = window["output_token_ids"]
        verified.append(dict(run_id=window["run_id"], pair_id=entry["pair_id"], arm=arm,
            full_output_tokens=128, full_frames=128, selected_load_context=144,
            selected_gpu_elapsed_ns=frames[16]["gpu_elapsed_ns"], native_io=io))
    require(all(v["baseline"] == v["action"] for v in outputs.values()),
            "paired complete output token IDs differ")
    calculate, proof = original_estimator(original_source_path)
    math_plan = SimpleNamespace(schema_version=2, timing_contract=tuple(timing_contract.items()))
    full = calculate(base, action, runs, groups, math_plan)
    calibration_ids = {e["pair_id"] for e in entries[:2]}
    # Same original numerical AST, with only calibration evidence available.
    calibration = calculate({k: v for k, v in base.items() if k[0] in calibration_ids},
        {k: v for k, v in action.items() if k[0] in calibration_ids},
        {k: v for k, v in runs.items() if k in calibration_ids},
        {"calibration": groups["calibration"], "validation": []}, math_plan)
    upper = calibration["baseline_ns"] + calibration["incremental_or_joint_ns"] + calibration["uncertainty_ns"]
    heldout = [dict(pair_id=k[0], measured_gpu_ns=v["timing"]["gpu_elapsed_ns"],
        predicted_upper_ns=upper, underprediction_ns=max(0, v["timing"]["gpu_elapsed_ns"] - upper),
        signed_error_ns=v["timing"]["gpu_elapsed_ns"] - upper) for k, v in action.items()
        if k[0] not in calibration_ids and k[2] == "measured"]
    covered = bool(heldout) and all(r["underprediction_ns"] == 0 for r in heldout)
    return dict(scope="server11_native_conditional_cost_candidate_v1", formula=FORMULA,
        condition=dict(gpu_uuid=plan["gpu_uuid"], source_lock_sha256=plan["source_lock_ref"]["sha256"],
            model_sha256=plan["model_sha256"], kv_layout_sha256=plan["kv_layout_sha256"],
            load=dict(active_decode=1, batch=1, prefill_tokens=0, context_length=144),
            stage="ssd_read", existing_io=[dict(ops=0, bytes=0)] * 4, operations=operations,
            physical_bytes=physical, basis="existing_io_plus_delta"),
        calibration_only_cost=calibration, calibration_predicted_upper_ns=upper,
        heldout_errors=heldout, heldout_covered=covered,
        all_pairs_original_estimator_audit_only=full, original_estimator_proof=proof,
        calibration_pairs=2, validation_pairs=1, full_output_tokens=768,
        windows=verified, conditional_cost_cell_qualified=False,
        native_execution_verified=False, production_qualified=False, strategy_effect_verified=False,
        resource_release_credit=False, interference_allowance=None,
        qualification_rule=plan["qualification_rule"],
        qualification_scope="finite_engineering_condition_not_SLO_or_probability_guarantee",
        unknown_conditions="reject", holdout_used_to_refit=False)


ENTRY = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
ENTRY_LABEL = 'server12-c5-native-common-cost-gpu01'
ENTRY_COMMON = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate'
ENTRY_OVERLAY = ENTRY_COMMON + '/source'


def binding_api():
    """Metadata validation only; this module cannot touch CUDA or create a grant."""
    path=Path(__file__).with_name('gpu_entry_binding.py')
    require(path.is_file() and not path.is_symlink(), 'new site binding validator missing')
    name='_c5_native_evidence_site_binding'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try: spec.loader.exec_module(module)
    except BaseException:sys.modules.pop(name,None);raise
    return module


def validate_native_plan(project,plan):
    """Require a real preregistered new job and its complete site source closure."""
    require(type(plan) is dict and plan.get('evidence_origin')=='native_runtime_preregistered' and
            plan.get('cpu_preparation_only') is False and plan.get('job_id')==ENTRY_LABEL,
            'synthetic, preparation and old native plans cannot grant authority')
    root=Path(project).resolve(strict=True)
    require(plan.get('project_root')==root.as_posix(), 'native plan actual project root')
    api=binding_api()
    config,refs,binding=api.verify_configuration(root,root/ENTRY/'NATIVE_COST_CONFIG.json')
    config_ref=EvidenceRef.from_mapping(plan['config_ref'])
    require(config_ref.path==ENTRY+'/NATIVE_COST_CONFIG.json' and
            refs.get(config_ref.path)==plan['config_ref'], 'site-frozen actual native config')
    config_ref.read(root)
    require(plan.get('site_binding_ref')==config['gpu_entry_binding_ref'] and
            plan.get('gpu_uuid')==config['gpu_uuid'] and plan.get('job_id')==config['label'] and
            plan.get('journal_run_id')==config['label'] and
            plan['source_lock_ref']['path']==config['source_lock'] and
            plan.get('wrapper_source_ref')==refs.get(ENTRY+'/run_native_cost_experiment.py') and
            plan.get('common_overlay_relative')==ENTRY_OVERLAY,
            'actual native job/site/UUID/common owner binding')
    owner=plan.get('common_owner_parameters')
    require(type(owner) is dict and set(owner)=={'max_accepted_parents','bridge_is_none'} and
            type(owner.get('max_accepted_parents')) is int and owner['max_accepted_parents']==8 and
            owner.get('bridge_is_none') is True, 'typed unchanged common owner parameters')
    EvidenceRef.from_mapping(plan['site_binding_ref']).read(root)
    lock=EvidenceRef.from_mapping(plan['source_lock_ref']).json(root)
    require(type(lock.get('files')) is list and len(lock['files'])==plan['source_lock_files'] and
            {r['path']:r for r in lock['files']}==refs and len(refs)==len(lock['files']),
            'complete native site source closure')
    for key,relative,sha in (
        ('native_source_ref',ENTRY_OVERLAY+'/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
         'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47'),
        ('collector_source_ref',ENTRY_COMMON+'/native_full_step_collector.py',
         'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf')):
        row=plan[key]
        require(row.get('path')==relative and row.get('sha256')==sha and refs.get(relative)==row,
                'unchanged C5 actual '+key)
        EvidenceRef.from_mapping(row).read(root)
    helper=ENTRY_COMMON+'/single_file_runtime_binding.py'
    require(helper in refs and refs[helper]['sha256']==
            '8b66b3aba1c936b85f4d82fe73c568650b712acf250aae454823cd35da060d60',
            'actual runtime identity helper in source closure')
    return config,refs,binding


def validate_prelaunch_plan(project,plan_ref):
    """Raw data must use the independently frozen prelaunch plan, never a self-ref."""
    api=binding_api()
    intent=api.read(project,ENTRY+'/GPU_LAUNCH_INTENT.json')
    require(intent.get('plan_ref')==plan_ref, 'plan differs from independent prelaunch intent')
    require(EvidenceRef.from_mapping(intent['plan_ref_receipt_ref']).json(project)==plan_ref,
            'prelaunch frozen plan-reference receipt drift')
    return intent


def verify_native_cell(project, *, plan_ref, measurements_ref, guard_ref,
                       expected_plan_ref, expected_guard_ref, original_source_path):
    """Final evidence entry: expected refs come from the guarded parent workflow.

    The parent pins the plan BEFORE launch and the original guard result AFTER
    natural completion. It must never create these inputs from CPU fixtures.
    The standalone analyzer above intentionally cannot qualify a native cell.
    """
    require(plan_ref == expected_plan_ref and guard_ref == expected_guard_ref,
            "independently expected frozen plan/actual GPU guard refs")
    pref, mref, gref = (EvidenceRef.from_mapping(r) for r in (plan_ref, measurements_ref, guard_ref))
    plan=pref.json(project)
    config,refs,binding=validate_native_plan(project,plan)
    validate_prelaunch_plan(project,plan_ref)
    record,guard=mref.json(project),gref.json(project)
    completed=binding_api().verify_completed_guard(project,config,binding,guard_ref=guard_ref)
    require(completed==guard, "actual completed guard matches fixed original result")
    require(record.get("scope") == "server11_native_paired_measurements_v1" and
            record.get("origin") == "native_gpu_recording" and record.get("plan_ref") == plan_ref,
            "actual measurement plan/origin binding")
    validate_guard(guard, gpu_uuid=plan["gpu_uuid"], job_id=plan["job_id"],
                   wrapper_path=plan["wrapper_source_ref"]["path"])
    for key in ("wrapper_source_ref", "collector_source_ref", "source_lock_ref", "cuda_event_source_ref"):
        EvidenceRef.from_mapping(plan[key]).read(project)
    for phase in ("before", "after"):
        ref = EvidenceRef.from_mapping(record["source_verification_refs"][phase])
        receipt = ref.json(project)
        require(receipt.get("phase") == phase and receipt.get("source_lock_ref") == plan["source_lock_ref"] and
                receipt.get("failed") == [] and type(receipt.get("files_verified")) is int and
                receipt["files_verified"] == plan["source_lock_files"], "actual source verification receipt mismatch")
    lock = EvidenceRef.from_mapping(plan["source_lock_ref"]).json(project)
    require(type(lock.get("files")) is list and len(lock["files"]) == plan["source_lock_files"],
            "actual source-lock closure size")
    locked = {r["path"]: r for r in lock["files"]}
    require(all(locked.get(plan[k]["path"]) == plan[k] for k in
                ("collector_source_ref", "wrapper_source_ref", "cuda_event_source_ref")),
            "native collector/wrapper absent from source closure")
    result = analyze_paired(plan, record["windows"], journal=record.get("journal"),
                           drain=record.get("final_drain"), original_source_path=original_source_path)
    for ref in (pref, mref, gref):
        ref.read(project)
    result.update(scope="server11_verified_native_conditional_cost_v1", native_execution_verified=True,
        conditional_cost_cell_qualified=result["heldout_covered"],
        evidence_refs=dict(plan=plan_ref, measurements=measurements_ref, actual_guard=guard_ref,
                           sources=record["source_verification_refs"]))
    return result


def main(argv=None):
    print(json.dumps(dict(status="USE_GUARDED_NATIVE_WORKFLOW", gpu_started=False,
        native_execution_verified=False, valid_native_receipt=None)))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
