"""CPU verification of real off/shadow/on native single-file experiments.

Reuses frozen v6 full-step and original owner-accounting validation. On-mode
deferral has its own causal rules: it need not issue inside the selected step.
Every mode reports the whole request and drain; no single run proves speedup.
"""
import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys


V6 = "artifacts/prefix_io_v1/server11-native-cost-v6-20261003/native_conditional_cost.py"
V6_BYTES = 36869
V6_SHA = "e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291"
SCOPE = "server11_p4_single_file_runtime_qualification_v1"
PROMPT = [28100] + list(range(1001, 1129))
SEED = 2829
BYTES = 917504
RECEIPT_REL = "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py"
REACTOR_REL = "third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name)
    return value


def migration_gate(mode, duration_ns, *, limit_ns):
    require(mode in ("off", "shadow", "on"), "known finite arm")
    duration_ns = integer(duration_ns, "actual selected CUDA duration", 1)
    covered = duration_ns <= integer(limit_ns, "strictly replayed calibration upper", 1)
    return covered, covered if mode in ("off", "shadow") else True


def safe(root, relative):
    require(type(relative) is str and relative and ":" not in relative and "\\" not in relative
            and all(x not in ("", ".", "..") for x in relative.split("/")), "project-relative path required")
    result = root
    for part in relative.split("/"):
        result /= part
        require(not result.is_symlink(), "symlink source/evidence refused")
    require(root in result.resolve().parents, "path outside project")
    return result


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file() and 0 < path.stat().st_size <= 32 * 1024**2, "bounded evidence/source file")
    raw = path.read_bytes()
    return dict(path=relative, bytes=len(raw), sha256=sha256(raw).hexdigest())


def load_module(root, reference, name):
    require(ref(root, reference["path"]) == reference, "source pin changed")
    path = safe(root, reference["path"])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    require(ref(root, reference["path"]) == reference, "source changed while loading")
    return module


def contract(root):
    return load_module(root, dict(path=V6, bytes=V6_BYTES, sha256=V6_SHA), "_runtime_frozen_v6_contract")


def accounting_validator(C, source_path):
    """Reuse the exact frozen accounting prefix, before arm-specific timing.

    This retains all event sequence, CQE bytes, four-stage counter, owner frame,
    and shutdown conservation checks unchanged. The sole new return exposes the
    already validated dictionaries to the different on-mode timing projection.
    """
    raw = Path(source_path).read_bytes()
    require(len(raw) == V6_BYTES and sha256(raw).hexdigest() == V6_SHA, "frozen accounting source")
    function = next(node for node in ast.parse(raw).body
                    if isinstance(node, ast.FunctionDef) and node.name == "validate_io")
    stop = next(index for index, node in enumerate(function.body) if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "first_start")
    require(stop > 10, "unchanged complete accounting prefix")
    function = deepcopy(function)
    function.name = "validate_accounting"
    function.body = function.body[:stop] + ast.parse("return accepted, completed, owners").body
    namespace = dict(C.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source_path) + "::unchanged_accounting_prefix", "exec", dont_inherit=True), namespace)
    return namespace["validate_accounting"]


def validate_native_source_binding(root, locked, raw, config, receipt):
    """Reject a legacy actual owner even when old and C4 files are both locked."""
    native_path = config['overlay_relative'] + '/' + REACTOR_REL
    native = locked[native_path]
    binding = raw['native_source_binding']
    mode = config['mode']
    require(mode in ('off','shadow','on') and binding.get('requested_mode') == mode,
            'actual native owner arm differs')
    require(binding.get('actual_source_ref') == native and binding.get('original_run_source_ref') == native and
            binding.get('original_run_code_verified') is True and
            type(binding.get('max_accepted_parents')) is int and binding['max_accepted_parents'] == 8 and
            binding.get('bridge_is_none') is (mode == 'off') and binding.get('same_requested_bridge') is True and
            binding.get('bridge_policy_mode') == (None if mode == 'off' else 'interference'),
            'actual common C4 owner/code/parent/bridge binding')
    require(native['sha256'] == receipt.calibration_native_source_sha256 and
            native in [item.mapping() for item in receipt.runtime_common_refs] and
            ref(root,native_path) == native, 'actual native owner is not common calibrated C4')
    loaded = binding.get('loaded_native_modules')
    require(type(loaded) is list and 1 <= len(loaded) <= 128 and
            len({row['module'] for row in loaded}) == len(loaded), 'bounded unique actual native modules')
    for row in loaded:
        name, source = row['module'], row['source_ref']
        require(type(name) is str and (name in ('py_kvcache','prefix_io_control') or
                name.startswith(('py_kvcache.','prefix_io_control.'))) and
                source['path'].startswith(config['overlay_relative']+'/') and
                locked.get(source['path']) == source and ref(root,source['path']) == source,
                'actual native module is outside the common C4 closure')
    require(any(row['module'] == 'py_kvcache.reactor' and row['source_ref'] == native for row in loaded),
            'actual C4 reactor missing from loaded modules')
    require(native in raw['verified_runtime_identity']['source_refs'],
            'actual identity helper did not observe the common C4 owner')


def validate_deferral(mode, before, after, tail, *, selected, accepted, trigger, binding_sha):
    if mode == "off":
        require(before is None and after is None and tail is None, "off must construct no bridge")
        return dict(proposed=0, actual_blocked_attempts=0, actual_decisions=[], off_bridge_is_none=True)
    for index, snapshot in enumerate((before, after, tail)):
        require(type(snapshot) is dict and snapshot.get("valid") is True and snapshot.get("fault") is None
                and snapshot.get("mode") == "interference" and snapshot.get("new_work_queues") == 0
                and snapshot.get("held_job_or_resource_owners") is False
                and snapshot.get("interference_production_qualified") is False,
                "actual healthy finite bridge without resource ownership")
        state = snapshot["conditional_single_file"]
        require(state.get("enabled") is True and type(state.get("shadow")) is bool
                and (index == 0 or state["shadow"] is (mode == "shadow"))
                and state.get("generic_production_qualified") is False
                and state.get("evidence_binding_sha256") == binding_sha, "actual finite receipt binding")
    initial = before["conditional_single_file"]
    state = after["conditional_single_file"]
    ending = tail["conditional_single_file"]
    for key in ("previews", "proposed_deferrals", "actual_blocked_attempts"):
        require(integer(initial[key], "initial policy count") == 0, "policy active before measured request")
        integer(state[key], "actual policy count")
        require(ending[key] == state[key], "policy counters changed after measured drain")
    require(initial["actual_decisions"] == [] and state["actual_decisions"] == ending["actual_decisions"],
            "complete bounded decision tail")
    require(state["previews"] >= state["proposed_deferrals"] > 0, "no real qualified defer proposal")
    if mode == "shadow":
        require(state["actual_blocked_attempts"] == 0 and state["actual_decisions"] == [],
                "shadow must preserve all native issue opportunities")
    else:
        decisions = state["actual_decisions"]
        require(mode == "on" and type(decisions) is list and len(decisions) == 1
                and state["actual_blocked_attempts"] > 0, "on must actually defer the one native operation")
        row = decisions[0]
        witness = selected["witness"]
        require(row["native_step_ordinal"] == selected["native_step_ordinal"]
                and row["start_record_before_ns"] == witness["start_record_before_ns"]
                and row["start_completed_query_ns"] == witness["start_completed_query_ns"],
                "deferral bound to actual current original step")
        first = integer(row["first_defer_ns"], "first actual defer", 1)
        last = integer(row["last_defer_ns"], "last actual defer", 1)
        require(trigger["trigger_before_ns"] <= first <= last
                and witness["start_completed_query_ns"] < first
                and last < witness["end_record_before_ns"] and last < accepted["at_ns"],
                "defer outside real causal step or after native acceptance")
        require(last - first <= 100000000, "deferral exceeds frozen native max-wait envelope")
        require(integer(row["count"], "actual bounded deferral count", 1) == state["actual_blocked_attempts"]
                and row.get("resource_release_credit") is False and row.get("production_qualified") is False,
                "actual defer count/release claims")
        require(type(row.get("key")) is list and len(row["key"]) == 2
                and row["key"][0] == selected["native_step_ordinal"] and type(row["key"][1]) is str,
                "one original native work identity")
    return dict(proposed=state["proposed_deferrals"], actual_blocked_attempts=state["actual_blocked_attempts"],
                actual_decisions=state["actual_decisions"], off_bridge_is_none=False)


def _window_io(C, accounting, raw, row, frames, config, native_sha):
    drain = C.original_post_shutdown_drain(raw["native_post_shutdown"], raw["native_tail_assertions"],
        run_id=config["label"], native_source_sha256=native_sha)
    journal = raw["native_journal"]
    accepted, completed, owners = accounting(journal, drain, capture=row["capture"], frames=frames,
        run_id=config["label"], native_source_sha256=native_sha, arm="action", measured_offset=16,
        physical_bytes=BYTES, operations=1, independent_payload=row["independent_payload"])
    timing = raw["request_and_drain_timing"]
    times = [integer(timing[key], "actual request/drain timestamp", 1) for key in
             ("request_started_ns", "request_finished_ns", "drain_started_ns", "drain_finished_ns")]
    require(times == sorted(times) and times[0] < times[-1], "actual request and drain chronology")
    begin = frames[0]["witness"]["start_record_before_ns"]
    end = frames[-1]["witness"]["end_record_after_ns"]
    require(times[0] <= begin < end <= times[1], "all 128 full steps enclosed by original request")
    require(all(event["at_ns"] not in (times[0], times[-1]) for event in journal["events"]),
            "ambiguous operation on request/drain boundary")
    selected = {op: event for op, event in accepted.items() if times[0] < event["at_ns"] < times[-1]}
    require(not any(event["at_ns"] >= times[-1] for event in accepted.values()),
            "new native work after measured request was fully drained")
    require(len(selected) == 1, "exactly one native operation over whole request plus drain")
    op, event = next(iter(selected.items()))
    completion = completed[op]
    require(event["stage"] == "ssd_read" and event["physical_bytes"] == BYTES
            and event["at_ns"] <= completion["at_ns"] < times[-1], "one complete native 917504-byte SSD read")
    prior = [owner for owner in owners if owner["captured_ns"] <= begin]
    after = [owner for owner in owners if owner["captured_ns"] >= end]
    require(prior and after and all(C._amount(x) == (0, 0) for x in prior[-1]["inflight"])
            and all(C._amount(x) == (0, 0) for x in after[-1]["inflight"]), "zero initial/final owner states")
    actions = row["capture"]["actions"]
    require(type(actions) is list and len(actions) == 1, "one actual original preload trigger")
    trigger = actions[0]
    witness = frames[16]["witness"]
    require(trigger.get("action_enabled") is True and trigger["step_offset"] == 16
            and trigger["native_step_ordinal"] == frames[16]["native_step_ordinal"]
            and witness["start_completed_query_ns"] < trigger["trigger_before_ns"]
            <= trigger["trigger_after_ns"] < witness["end_record_before_ns"]
            and trigger["trigger_before_ns"] <= event["at_ns"], "actual selected preload causal trigger")
    payload = row["independent_payload"]
    require(type(payload) is dict and payload.get("ssd_only") is True
            and payload.get("cache_miss_before_submit") is True
            and payload.get("operations") == 1 and payload.get("physical_bytes") == BYTES,
            "actual SSD-only independent input")
    keys = payload["preload_key_sha256"]
    foreground = payload["request_prefix_key_sha256"]
    require(type(keys) is list and len(keys) == 1 and type(foreground) is list and foreground
            and not set(keys) & set(foreground), "independent original cache-key payload")
    for key in keys + foreground:
        C.digest(key, "actual original cache key")
    action = trigger["trigger_result"]
    require(action.get("accepted") is True and action.get("hashes") == keys
            and action.get("physical_bytes") == BYTES and action.get("stage") == "ssd_read"
            and action.get("h2d_requested") is False, "actual original preload result/bytes")
    if config["mode"] in ("off", "shadow"):
        # The unchanged original overlap rule applies only to unblocked controls.
        C.validate_io(journal, drain, capture=row["capture"], frames=frames,
            run_id=config["label"], native_source_sha256=native_sha, arm="action", measured_offset=16,
            physical_bytes=BYTES, operations=1, independent_payload=payload)
    return event, completion, trigger, times, drain


def analyze_runtime(C, accounting, raw, config, receipt, native_sha):
    """Semantics only; CPU fixtures cannot set native_execution_verified."""
    mode = config["mode"]
    require(mode in ("off", "shadow", "on") and raw.get("mode") == mode and raw.get("label") == config["label"]
            and raw.get("status") == "PASS_P4_SINGLE_FILE_RAW_REQUIRES_VALIDATION"
            and raw.get("origin") == "native_gpu_recording" and raw.get("gpu_uuid") == receipt.signature[1]
            and raw.get("source_lock") == config["source_lock"]
            and raw.get("original_engine_shutdown_returned") is True, "actual mode/native execution identity")
    require(raw["model"]["manifest_sha256"] == raw["model"]["plan_sha256"] == receipt.signature[0],
            "actual unchanged model manifest")
    limit_ns = integer(receipt.cost_upper_ns, "strictly replayed v6 calibration-only upper", 1)
    budget_ns = integer(receipt.step_budget_ns, "strictly replayed v6 A-only budget", 1)
    require(raw["policy_receipt_binding_ref"] == receipt.binding_ref.mapping(), "actual receipt binding")
    require(raw.get("fresh_original_process") is True and raw.get("actual_guarded_gpu_job_count") == 1
            and raw.get("cache_reset_count") == 0 and raw.get("baseline_policy_changed") is False
            and raw.get("policy_parameters") == dict(sample_max_age_ns=100000000, max_wait_ns=100000000,
                                                     max_accepted_parents=8), "frozen finite policy/engine settings")
    runtime = raw["verified_runtime_identity"]
    require(runtime.get("signature_prefix") == list(receipt.signature[:4])
            and runtime.get("same_original_runner") is True, "actual source-bound original runtime identity")
    require(type(raw["windows"]) is list and len(raw["windows"]) == 1, "one full original request")
    row = raw["windows"][0]
    require(row["condition"] == "B" and row["prompt_token_ids"] == PROMPT and row["seed"] == SEED,
            "same preregistered single-file foreground workload")
    output = row["frontend"]["output"]
    require(output["request_id"] == row["request_id"] == row["capture"]["run_id"]
            and output["num_cached_tokens"] == 128, "original external/native frontend identity and hot prefix")
    frames = C.validate_capture(row["capture"], run_id=row["request_id"], request_id=output["native_request_id"],
        output_ids=output["output_token_ids"], prompt_tokens=129, cached_tokens=128, measured_offset=16, warmup_offsets=[1])
    warmups = raw["warmups"]
    require(type(warmups) is list and len(warmups) == 1 and warmups[0]["prompt_token_ids"] == PROMPT
            and warmups[0]["seed"] == SEED and warmups[0]["output_token_ids"] == output["output_token_ids"]
            and len(warmups[0]["flush"]["output_token_ids"]) == 1, "full original cache warmup and output equality")
    accepted, completed, trigger, times, drain = _window_io(C, accounting, raw, row, frames, config, native_sha)
    selected = frames[16]
    decisions = validate_deferral(mode, raw["p4_before_measurement"], raw["p4_after_measurement"],
        raw["p4_after_shutdown"], selected=selected, accepted=accepted, trigger=trigger,
        binding_sha=receipt.binding_ref.sha256)
    migration_pass, qualified = migration_gate(mode, selected["gpu_elapsed_ns"], limit_ns=limit_ns)
    witness = selected["witness"]
    return dict(scope=SCOPE, mode=mode, native_execution_verified=False, runtime_condition_qualified=False,
        qualification_passed=False,
        semantic_conditions_passed=qualified, frozen_cost_migration_pass=migration_pass,
        frozen_cost_upper_ns=limit_ns, frozen_a_only_budget_ns=budget_ns,
        selected_gpu_elapsed_ns=selected["gpu_elapsed_ns"], full_output_tokens=128, full_frames=128,
        output_token_ids=output["output_token_ids"], prompt_sha256=C.canonical_hash(PROMPT), seed=SEED,
        all_step_gpu_elapsed_ns=[frame["gpu_elapsed_ns"] for frame in frames],
        sum_full_step_gpu_ns=sum(frame["gpu_elapsed_ns"] for frame in frames),
        whole_request_ns=times[1]-times[0], drain_ns=times[3]-times[2],
        total_request_and_drain_ns=times[3]-times[0],
        native_io=dict(stage="ssd_read", operations=1, physical_bytes=BYTES,
            accepted_at_ns=accepted["at_ns"], completed_at_ns=completed["at_ns"],
            preload_key_sha256=row["independent_payload"]["preload_key_sha256"],
            accepted_minus_selected_host_end_ns=accepted["at_ns"]-selected["host_end_ns"],
            accepted_minus_selected_end_record_ns=accepted["at_ns"]-witness["end_record_before_ns"],
            accepted_minus_selected_end_query_ns=accepted["at_ns"]-witness["end_completed_query_ns"],
            completed_and_original_shutdown_drained=True, resource_release_credit=False,
            gpu_nonoverlap_inferred=False), deferral=decisions,
        permits_next_mode=None, production_qualified=False, strategy_effect_verified=False,
        qualification_scope="finite_native_lifecycle_and_conditional_scheduling_not_performance_proof")


def verify_runtime(root, *, config_ref, result_ref, guard_ref, before_ref, after_ref, _depth=0):
    require(_depth <= 2, "bounded ordered predecessor chain")
    root = Path(root).resolve(strict=True)
    C = contract(root)
    read = lambda item: C.EvidenceRef.from_mapping(item).json(root)
    config, raw, guard = read(config_ref), read(result_ref), read(guard_ref)
    require(config["root"] == str(root) and config["gpu_uuid"] == raw["gpu_uuid"], "actual project/device config")
    require(raw.get("config_ref") == config_ref and raw.get("subprocess_sid") == guard.get("session_id"),
            "actual immutable config and original guard OS session")
    lock_ref = ref(root, config["source_lock"])
    lock = read(lock_ref)
    locked = {item["path"]: item for item in lock["files"]}
    require(len(locked) == len(lock["files"]), "unique common source closure")
    for phase, reference in (("before", before_ref), ("after", after_ref)):
        source = read(reference)
        require(source.get("phase") == phase and source.get("source_lock_ref") == lock_ref
                and source.get("config_ref") == config_ref
                and source.get("failed") == [] and source.get("files_verified") == len(locked),
                "full actual before/after source closure verification")
    module_path = config["overlay_relative"] + "/" + RECEIPT_REL
    native_path = config["overlay_relative"] + "/" + REACTOR_REL
    required = [module_path, native_path, config["collector_relative"], config["runtime_binding_relative"],
                config["binding_relative"], Path(__file__).resolve().relative_to(root).as_posix()]
    for path in required:
        require(path in locked and ref(root, path) == locked[path], "actual runtime source absent/changed: " + path)
    source_binding = raw["collector_native_source_binding"]
    require(source_binding == dict(legacy_lookup_key=REACTOR_REL,
            actual_source_ref=locked[native_path], adapter_native_source_sha256_before=locked[native_path]["sha256"],
            adapter_native_source_sha256_after=locked[native_path]["sha256"],
            same_adapter_all_frames=True, full_frame_count=128), "actual unmodified scalar adapter source binding")
    R = load_module(root, locked[module_path], "_runtime_single_file_receipt")
    receipt = R.load_verified_single_file(root, config["binding_relative"])
    require(locked[config["binding_relative"]] == receipt.binding_ref.mapping(), "binding outside actual new closure")
    require(locked[native_path]["sha256"] == receipt.calibration_native_source_sha256
            and locked[native_path] in [item.mapping() for item in receipt.runtime_common_refs],
            "actual runtime reactor must be the same common C4 calibration source")
    validate_native_source_binding(root,locked,raw,config,receipt)
    for reference in raw["verified_runtime_identity"]["source_refs"]:
        require(locked.get(reference["path"]) == reference and ref(root, reference["path"]) == reference,
                "actual loaded runtime identity source outside common closure")
    wrapper = next((value for value in guard.get("command", []) if type(value) is str
                    and value.endswith("/run_p4_single_file_experiment.py")), None)
    require(wrapper is not None and wrapper in locked and ref(root, wrapper) == locked[wrapper], "frozen actual GPU wrapper")
    C.validate_guard(guard, gpu_uuid=config["gpu_uuid"], job_id=config["label"], wrapper_path=wrapper)
    require("--config" in guard["command"] and guard["command"][guard["command"].index("--config")+1] == config_ref["path"],
            "guard launched the independently pinned per-arm config")
    row = raw["windows"][0]
    event = row["capture"]["event_source"]
    calibration = read(receipt.binding_ref.mapping())["calibration"]
    old_plan = read(calibration["plan_ref"])
    eref = old_plan["cuda_event_source_ref"]
    require(event.get("origin") == "native_gpu_recording" and event.get("module") == "torch.cuda.streams"
            and event.get("class_name") == "Event" and event.get("sha256") == eref["sha256"]
            and event.get("bytes") == eref["bytes"] and event.get("path") == str(safe(root, eref["path"])),
            "actual installed CUDA Event source")
    # Same actual original engine options as calibration, except private paths.
    old_record = read(calibration["measurements_ref"])
    original_child = read(old_record["windows"][0]["raw_child_ref"])
    configs = [deepcopy(value) for value in (raw["engine_config"], original_child["engine_config"])]
    for value in configs:
        value["kv_transfer_config"]["kv_connector_extra_config"].pop("shared_storage_path")
    require(configs[0] == configs[1], "actual engine differs from single-file calibration")
    inputs = read(ref(root, config["input_manifest"]))
    payload = inputs["groups"][0]["files"][0]
    require(payload["bytes"] == BYTES and row["independent_payload"]["preload_key_sha256"] == [payload["block_hash"]],
            "same original single-file input for all modes")
    membership = row["cold_input_membership"]
    require(type(membership) is list and len(membership) == 1 and membership[0]["block_hash"] == payload["block_hash"]
            and all(membership[0].get(key) is False for key in ("shared_cached", "preload_slots", "staging_cached"))
            and all(type(membership[0].get(key)) is int and membership[0][key] == 0
                    for key in ("pending", "inflight", "retained_refs")), "actual cold input owner membership")
    mapping = row["actual_native_file_mapping"]
    require(mapping == raw["actual_native_file_mapping"] and len(mapping) == 24
            and next(item["path"] for item in mapping if item["block_hash"] == payload["block_hash"])
                == str(safe(root, raw["private_storage"] + "/" + payload["path"])), "actual native FileMapper input")
    actual_file = dict(path=raw["private_storage"] + "/" + payload["path"], bytes=payload["bytes"], sha256=payload["sha256"])
    C.EvidenceRef.from_mapping(actual_file).read(root)
    scheduler = row["frontend"]["original_scheduler_hash_observations"]
    require(scheduler and all(item["request_id"] == row["frontend"]["output"]["native_request_id"]
            and item["prompt_token_ids"] == PROMPT and payload["block_hash"] not in item["block_hashes"] for item in scheduler)
            and row["independent_payload"]["request_prefix_key_sha256"] == scheduler[0]["block_hashes"][:8],
            "actual independent foreground original block hashes")
    result = analyze_runtime(C, accounting_validator(C, safe(root, V6)), raw, config, receipt,
                             locked[native_path]["sha256"])
    previous_ref = config.get("previous_qualification_ref")
    if config["mode"] == "off":
        require(previous_ref is None, "off is the first arm")
    else:
        previous = read(previous_ref)
        require(previous.get("scope") == SCOPE and previous.get("permits_next_mode") == config["mode"],
                "wrong independently qualified predecessor")
        evidence = previous["evidence_refs"]
        checked = verify_runtime(root, config_ref=evidence["config"], result_ref=evidence["raw_result"],
            guard_ref=evidence["actual_guard"], before_ref=evidence["source_before"],
            after_ref=evidence["source_after"], _depth=_depth+1)
        require(checked == previous and checked["runtime_condition_qualified"] is True
                and checked["source_lock_ref"] == lock_ref and checked["binding_ref"] == receipt.binding_ref.mapping(),
                "predecessor must replay from actual raw source/guard evidence")
        require(checked["output_token_ids"] == result["output_token_ids"]
                and checked["native_io"]["preload_key_sha256"] == result["native_io"]["preload_key_sha256"],
                "same full outputs and SSD work across modes")
        result["descriptive_change_from_previous"] = dict(previous_mode=checked["mode"],
            selected_gpu_elapsed_ns=result["selected_gpu_elapsed_ns"]-checked["selected_gpu_elapsed_ns"],
            sum_full_step_gpu_ns=result["sum_full_step_gpu_ns"]-checked["sum_full_step_gpu_ns"],
            whole_request_ns=result["whole_request_ns"]-checked["whole_request_ns"],
            total_request_and_drain_ns=result["total_request_and_drain_ns"]-checked["total_request_and_drain_ns"],
            performance_effect_verified=False)
    qualified = result["semantic_conditions_passed"]
    result.update(native_execution_verified=True, runtime_condition_qualified=qualified, qualification_passed=qualified,
        permits_next_mode=({"off": "shadow", "shadow": "on"}.get(config["mode"]) if qualified else None),
        binding_ref=receipt.binding_ref.mapping(), source_lock_ref=lock_ref,
        evidence_refs=dict(config=config_ref, raw_result=result_ref, actual_guard=guard_ref,
                           source_before=before_ref, source_after=after_ref))
    for reference in (config_ref, result_ref, guard_ref, before_ref, after_ref, lock_ref):
        C.EvidenceRef.from_mapping(reference).read(root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "config", "result", "guard", "before", "after", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    result = verify_runtime(root, config_ref=ref(root, args.config), result_ref=ref(root, args.result),
        guard_ref=ref(root, args.guard), before_ref=ref(root, args.before), after_ref=ref(root, args.after))
    output = safe(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(mode=result["mode"], native_execution_verified=True,
        runtime_condition_qualified=result["runtime_condition_qualified"], permits_next_mode=result["permits_next_mode"],
        output_ref=ref(root, args.output)), sort_keys=True))
    return 0 if result["runtime_condition_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
