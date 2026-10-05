"""Pure-stdlib P316 closeout audit v3; read-only evidence, no native/GPU imports.

V3 keeps v2 CPU/GiB/runtime checks and permits the one authenticated historical
AUX metadata reference in the full lock08 superset. It is not runtime source.

CLI: --root ROOT --manifest JSON --output NEW_JSON.
Every FileRef is exactly {path, sha256}; paths are root-relative or absolute
inside ROOT. Duplicate JSON keys and nonfinite JSON constants are rejected. Manifest schema_version=1 binds gpu_uuid, requirements_review,
preregistration, core_runs(6), finite_runs(4), capacity_gates(2),
capacity_runs(0..2), hot_runs(4), observation_runs(4), calibration_runs(3),
conditional_table, cpu_receipts, patch_roundtrip, native_qualifications,
lineage_receipts, delivery_receipts, native_source_hashes.

A model entry binds label, arm(U/F16/P4/F8/P512), policy_mode, profile,
native_observation, wrapper/result/config/analysis/workload_manifest/reference/
control_config FileRefs. A capacity model entry additionally binds capacity_domain_id.
A calibration entry binds label, partition(calibration/validation), required_status,
wrapper/result.
Capacity gate disposition is PASS, FAILED_NUMERIC_GATE or FAILED_COST_GATE;
it binds domain, phases (2 or 6 receipt entries), reference_result and, for a
cost gate, cost_result; PASS also binds permit. Original source/plan bindings
are checked from those producer outputs, not recomputed by importing producers.
A command CPU entry has kind=command, receipt and guard; kind=matrix uses
the existing CPU summary. A patch receipt has status PASS_PATCH_ROUNDTRIP,
check_exit/apply_exit=0, exact_content=true, files with expected/actual SHA.
Other qualification/lineage/delivery entries bind a receipt and nonempty checks
[{field: [typed JSON object keys/list indices], equals: typed expected value}].
lineage_receipts must name all of batch/iodepth/cache_capacity/preload_capacity/
reserve/stage_quota. delivery_receipts must name source_lock/version_lock/
GPU_budget_idle/storage_and_source_preserved/problem_report/reproduction.
These assertions must refer to actual producer evidence; declarations alone
cannot certify native physical behavior or create performance observations.

The script never starts work, changes caches, enables P4, invents missing
values, aggregates token samples as independent runs, or claims a global optimum.
P3_EVIDENCE_COMPLETE_FOR_ROOT_REVIEW is an engineering evidence disposition,
not a formal efficacy result, production-state interference table, or approval.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics

STAGES = ("ssd_read", "h2d", "d2h", "ssd_write")
ARMS = ("U", "F16", "P4", "F8", "P512")
MODES = {"U": "off", "F16": "fixed", "P4": "pressure", "F8": "fixed", "P512": "pressure"}
CORE_ORDER = ["U", "F16", "P4", "P4", "F16", "U"]
FINITE_ORDER = ["F8", "P512", "P512", "F8"]
CAPACITY_IDS = ("cap960-l1", "cap1024-l2")
CELL_IDS = {"none8", "warm_h2d8", "d2h_write8", "cold_ssd_h2d8", "joint8", "warm_h2d1", "d2h_write1"}
CPU_PHASES = ("warmup", "cohort", "tail")
CPU_REGIONS = ("_prefix_stage_decide", "_prefix_stage_settle", "_prefix_stage",
               "_prefix_stage_copy", "_prefix_capacity_snapshot", "_capture_owner_snapshot", "observer_sink")
SHA = re.compile(r"^[0-9a-f]{64}$")


class AuditError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise AuditError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer " + name)
    return value


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), "invalid finite scalar " + name)
    return value


def same(left, right):
    """JSON typed equality: False must never satisfy expected 0."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def close(left, right):
    return math.isclose(number(left, "observed"), number(right, "expected"), rel_tol=1e-10, abs_tol=1e-12)


def lookup(value, fields):
    require(type(fields) is list and fields, "typed field selector required")
    for field in fields:
        if type(value) is dict:
            require(type(field) is str and field in value, "missing object field " + str(field))
        elif type(value) is list:
            require(type(field) is int and 0 <= field < len(value), "invalid array field")
        else:
            raise AuditError("selector crossed a scalar")
        value = value[field]
    return value


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}
        self.inputs = {}
        self.labels = set()
        self.run_paths = set()
        self.successful_gpu_sessions = []

    def path(self, value):
        require(type(value) is str and value, "path required")
        raw = Path(value)
        path = raw if raw.is_absolute() else self.root / raw
        require(path.is_absolute(), "absolute root required")
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise AuditError("input outside project root") from exc
        require(".." not in relative.parts, "parent path traversal")
        cursor = self.root
        for part in relative.parts:
            cursor = cursor / part
            require(not cursor.is_symlink(), "symlink evidence forbidden")
        require(path.is_file(), "missing regular evidence: " + str(path))
        return path

    def read(self, ref):
        require(type(ref) is dict and set(ref) == {"path", "sha256"}, "strict FileRef required")
        require(type(ref["sha256"]) is str and SHA.fullmatch(ref["sha256"]), "SHA-256 required")
        path = self.path(ref["path"])
        key = (str(path), ref["sha256"])
        if key not in self.cache:
            require(path.stat().st_size <= 64 * 1024 * 1024, "evidence too large")
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            require(digest == ref["sha256"], "evidence changed: " + str(path))
            def bad_constant(token):
                raise AuditError("nonfinite JSON constant " + token)
            def unique(items):
                result = {}
                for key, value in items:
                    require(key not in result, "duplicate JSON object key")
                    result[key] = value
                return result
            value = json.loads(data, parse_constant=bad_constant, object_pairs_hook=unique)
            require(type(value) is dict, "object evidence required")
            self.cache[key] = value
            self.inputs[str(path.relative_to(self.root))] = digest
        return self.cache[key]

    def run_identity(self, entry):
        label = entry["label"]
        require(type(label) is str and label and len(label) <= 160 and label not in self.labels,
                "missing/duplicate run label")
        path = str(self.path(entry["wrapper"]["path"]))
        require(path not in self.run_paths, "duplicate wrapper used as independent run")
        self.labels.add(label)
        self.run_paths.add(path)


def gpu_wrapper(evidence, entry, gpu_uuid):
    evidence.run_identity(entry)
    wrapper = evidence.read(entry["wrapper"])
    require(wrapper["label"] == entry["label"], "wrapper label mismatch")
    require(wrapper["gpu_uuid"] == gpu_uuid, "GPU UUID mismatch")
    require(wrapper["gpu_job_attempted"] is True and
            type(wrapper["exit"]) is int and wrapper["exit"] == 0 and
            type(wrapper["child_exit"]) is int and wrapper["child_exit"] == 0,
            "actual GPU child did not succeed")
    require(wrapper["timed_out"] is False and wrapper["interrupted_signal"] is None and
            wrapper["error"] is None and wrapper["session_drained"] is True and
            wrapper["session_members_after_cleanup"] == [], "GPU session not cleanly drained")
    integer(wrapper["session_id"], "session_id", 1)
    number(wrapper["elapsed_seconds"], "wrapper duration", positive=True)
    require(type(wrapper["command"]) is list and wrapper["command"] and
            all(type(x) is str for x in wrapper["command"]), "actual argv required")
    evidence.successful_gpu_sessions.append(dict(label=entry["label"], session_id=wrapper["session_id"]))
    return wrapper


def flag(argv, name):
    positions = [i for i, value in enumerate(argv) if value == name]
    require(len(positions) == 1 and positions[0] + 1 < len(argv), "missing/duplicate argv flag " + name)
    return argv[positions[0] + 1]


def bound_path(evidence, declared, expected):
    left = Path(declared)
    left = left if left.is_absolute() else evidence.root / left
    right = Path(expected)
    right = right if right.is_absolute() else evidence.root / right
    require(left.resolve() == right.resolve(), "producer path binding differs")


def actual_controls(raw, analysis):
    controls = raw["final_probe"]["simple_native_control"]
    require(type(controls) is list and controls and same(controls, analysis["final_control"]),
            "analysis final physical controls differ")
    for control in controls:
        require(control["physical_drained"] is True, "physical drain absent")
        adm = control["admission"]
        require(adm["count_valid"] is True and integer(adm["accepted_parents"], "parents") == 0 and
                integer(adm["peak_accepted_parents"], "parent peak") <= 64 and
                adm["native_drain_unknown"] is False, "parent protection not fully retired")
        accounting = control["stage_accounting"]
        require(accounting["valid"] is True and accounting["error"] is None and integer(accounting["outstanding_records"], "records") == 0 and
                set(accounting["stages"]) == set(STAGES), "invalid stage drain")
        policy = control["controller"]
        for stage, actual in accounting["stages"].items():
            require(integer(actual["inflight_ops"], stage) == 0 and integer(actual["inflight_bytes"], stage) == 0,
                    "physical stage remains inflight")
            integer(actual["accepted_ops"], stage)
            integer(actual["accepted_bytes"], stage)
            require(integer(actual["failed_ops"], "failed physical stage") == 0 and
                    same(actual["completed_ops"], actual["accepted_ops"]) and
                    same(actual["completed_requested_bytes"], actual["accepted_bytes"]),
                    "failed/incomplete actual stage execution")
            if policy is not None:
                accepted = policy["observed_api_accepted"][stage]
                require(same(accepted["ops"], actual["accepted_ops"]) and
                        same(accepted["bytes"], actual["accepted_bytes"]), "accepted counter mismatch")
        if policy is not None:
            require(policy["faulted"] is False and policy["pending_attempt"] is False and
                    integer(policy["uncertain_ops"], "uncertain") == 0 and
                    integer(policy["completion_unknown_ops"], "completion unknown") == 0,
                    "controller accounting uncertain")
    return controls


CPU_COUNTING = "outermost_only; nested inclusive hook costs are never added twice"
CPU_POINT_SCOPE = "bounded pure metadata/control/observer hooks; includes timer overhead"
CPU_EXCLUDES = ["native GPU/IO launch and wait", "model execution CPU",
                "frontend/core whole-process CPU", "unwrapped cache bookkeeping"]
CPU_INTERVAL_SCOPE = ("outermost metadata/control/observer hook CPU including timing instrumentation; "
                      "explicit RPC phase boundaries; no whole-model CPU claim")
CPU_ANALYSIS_SCOPE = ("pure metadata/control/observer hook thread CPU; timer overhead included; "
                      "native/backend/model CPU excluded")
CPU_INTERVAL_PAIRS = (
    ("warmup_and_start_boundary", "native_kv", "cohort_probe_start"),
    ("cohort_and_tail", "cohort_probe_start", "probe"),
    ("finish", "probe", "final_probe"))


def _cpu_point(value):
    """Recorded producer schema; no absent counter becomes zero."""
    keys = {"schema_version", "clock", "valid", "error", "counting", "phase_regions",
            "phase_totals", "publication_calls", "publication_thread_cpu_ns", "scope",
            "excludes", "algorithm_cpu_claim", "full_production_cpu_breakdown"}
    require(type(value) is dict and set(value) == keys, "CPU snapshot exact schema required")
    require(same(value["schema_version"], 1) and value["valid"] is True and value["error"] is None and
            value["clock"] == "thread_time_ns" and value["counting"] == CPU_COUNTING and
            value["scope"] == CPU_POINT_SCOPE and same(value["excludes"], CPU_EXCLUDES) and
            value["algorithm_cpu_claim"] is False and value["full_production_cpu_breakdown"] is False,
            "CPU probe invalid/scope changed")
    require(type(value["phase_regions"]) is dict and set(value["phase_regions"]) == set(CPU_PHASES) and
            type(value["phase_totals"]) is dict and set(value["phase_totals"]) == set(CPU_PHASES),
            "CPU phase coverage missing")
    for phase in CPU_PHASES:
        regions = value["phase_regions"][phase]
        require(type(regions) is dict and set(regions) == set(CPU_REGIONS), "CPU region coverage missing")
        for region in regions.values():
            require(type(region) is dict and set(region) == {"calls", "thread_cpu_ns"}, "CPU scalar shape")
            integer(region["calls"], "CPU calls")
            integer(region["thread_cpu_ns"], "thread CPU")
        require(same(value["phase_totals"][phase],
                     {key: sum(row[key] for row in regions.values()) for key in ("calls", "thread_cpu_ns")}),
                "nested/summed CPU totals differ")
    integer(value["publication_calls"], "publication calls")
    integer(value["publication_thread_cpu_ns"], "publication thread CPU")


def _cpu_interval(before, after):
    """Recompute each region so cancelling edits cannot hide behind equal totals."""
    phases = {}
    for phase in CPU_PHASES:
        phases[phase] = {}
        for region in CPU_REGIONS:
            row = {}
            for key in ("calls", "thread_cpu_ns"):
                a, b = before["phase_regions"][phase][region][key], after["phase_regions"][phase][region][key]
                require(b >= a, "CPU region counter moved backwards")
                row[key] = b - a
            phases[phase][region] = row
    for key in ("publication_calls", "publication_thread_cpu_ns"):
        require(after[key] >= before[key], "CPU publication counter moved backwards")
    totals = {p: {key: sum(row[key] for row in phases[p].values())
                  for key in ("calls", "thread_cpu_ns")} for p in CPU_PHASES}
    return dict(phase_regions=phases, phase_totals=totals, clock="thread_time_ns",
                counting=CPU_COUNTING, algorithm_cpu_claim=False, full_production_cpu_breakdown=False)


def cpu_measurement(raw, analysis):
    require(raw["native_cpu_probe_requested"] is True, "native metadata CPU probe absent")
    points = {k: raw[k]["native_metadata_cpu"] for k in
              ("native_kv", "cohort_probe_start", "probe", "final_probe")}
    for value in points.values():
        _cpu_point(value)
    intervals = raw["native_metadata_cpu_intervals"]
    names = {row[0] for row in CPU_INTERVAL_PAIRS}
    require(type(intervals) is dict and set(intervals) == names | {"scope"} and
            intervals["scope"] == CPU_INTERVAL_SCOPE, "CPU raw interval envelope/scope changed")
    canonical = {}
    for name, start, end in CPU_INTERVAL_PAIRS:
        canonical[name] = _cpu_interval(points[start], points[end])
        require(same(intervals[name], canonical[name]), "CPU interval region/count/scope differs from snapshots")
    cpu = analysis["native_metadata_cpu"]
    measured = sum(canonical["cohort_and_tail"]["phase_totals"][phase]["thread_cpu_ns"]
                   for phase in CPU_PHASES) / 1e9
    keys = {"intervals", "cohort_and_tail_thread_cpu_seconds", "final_cumulative",
            "algorithm_cpu_claim", "full_production_cpu_breakdown", "scope"}
    require(type(cpu) is dict and set(cpu) == keys and same(cpu["intervals"], canonical) and
            close(cpu["cohort_and_tail_thread_cpu_seconds"], measured) and
            same(cpu["final_cumulative"], points["final_probe"]) and
            cpu["scope"] == CPU_ANALYSIS_SCOPE and cpu["algorithm_cpu_claim"] is False and
            cpu["full_production_cpu_breakdown"] is False,
            "analysis CPU does not match actual wrapped thread CPU")
    return measured



def engine_projection(config):
    value = copy.deepcopy({k: config[k] for k in ("model", "engine", "sampling")})
    extra = value["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
    for key in ("shared_storage_path", "prefix_io_observation_run_id", "prefix_io_stage_policy"):
        extra.pop(key, None)
    if "prefix_io_parent_admission" in extra:
        extra["prefix_io_parent_admission"].pop("run_id", None)
    return value


def model_run(evidence, entry, gpu_uuid):
    wrapper = gpu_wrapper(evidence, entry, gpu_uuid)
    raw = evidence.read(entry["result"])
    analysis = evidence.read(entry["analysis"])
    config = evidence.read(entry["config"])
    manifest = evidence.read(entry["workload_manifest"])
    reference = evidence.read(entry["reference"])
    control_config = evidence.read(entry["control_config"])
    require(entry["arm"] in ARMS and entry["policy_mode"] == MODES[entry["arm"]], "arm mode mismatch")
    require(entry["profile"] in ("mixed_readwrite", "gpu_hot", "mixed") and
            raw["profile"] == manifest["profile"] == entry["profile"], "profile mismatch")
    require(raw["status"] == "PASSED_NATIVE_C2_DEVELOPMENT_REPLAY" and
            analysis["status"] == "PASS_FULL_MODEL_SIMPLE_STAGE", "failed model input")
    require(config["gpu_uuid"] == gpu_uuid and raw["policy_mode"] == analysis["policy_mode"] == entry["policy_mode"],
            "model identity/mode mismatch")
    require(analysis["source_sha256"] == entry["result"]["sha256"], "analysis raw-source SHA mismatch")
    bound_path(evidence, analysis["source"], entry["result"]["path"])
    require(raw["manifest_sha256"] == entry["workload_manifest"]["sha256"] and
            same(config["manifest"], manifest), "workload binding changed")
    argv = wrapper["command"]
    require(any(Path(x).name in ("run_concurrent_pilot_p316.py", "run_concurrent_capacity_p316.py") for x in argv),
            "actual P316 model driver required")
    bound_path(evidence, flag(argv, "--output"), str(evidence.path(entry["result"]["path"]).parent))
    bound_path(evidence, flag(argv, "--manifest"), entry["workload_manifest"]["path"])
    bound_path(evidence, flag(argv, "--simple-stage-config"), entry["control_config"]["path"])
    require("--native-cpu-probe" in argv, "model CPU observation missing")
    if "capacity_domain_id" in entry:
        require(flag(argv, "--capacity-domain") == entry["capacity_domain_id"], "capacity replay domain mismatch")
        require(analysis["capacity_gate_audited"] is True and
                analysis["capacity_domain_id"] == entry["capacity_domain_id"] and
                same_capacity_domain(analysis["capacity_domain"], manifest["capacity_domain"]),
                "loaded capacity lacks independent NEW provenance/geometry audit")
        require(analysis["capacity_qualification"]["status"] == "PASSED_CAPACITY_UNLOADED_POINT_GATE",
                "loaded capacity has no independently qualified original gate")
    require(raw["cache_resets"] == 0 and type(raw["cache_resets"]) is int, "per-request cache reset changes development scope")
    require(raw["engine_shutdown"] == "completed" and
            same(raw["source_preservation"], {"checked_files": 3048, "changed": []}), "source/shutdown failure")
    actual_kv = integer(raw["native_kv"]["actual_gpu_kv_bytes"], "actual GPU KV", 1)
    require(actual_kv <= 2147483648, "actual GPU KV exceeds common 2GiB budget")
    require(raw["native_kv"]["groups"] == 1 and raw["native_kv"]["layers"] == 28,
            "model native group/layer geometry differs")
    for point in ("cohort_probe_start", "probe", "final_probe"):
        native = raw[point]
        require(type(native["kv_budget_bytes"]) is int and native["kv_budget_bytes"] == 2147483648 and
                type(native["staging_budget_bytes"]) is int and
                native["staging_budget_bytes"] == manifest["staging_bytes"], "declared probe resource budget differs")
    handlers = raw["probe"]["handlers"]
    require(type(handlers) is list and handlers, "actual staging allocation proof missing")
    for handler in handlers:
        require(integer(handler["staging_bytes"], "allocated staging", 1) <= manifest["staging_bytes"] and
                handler["pinned"] is True and handler["progress_enabled"] is True and
                integer(handler["observation_failures"], "native observation failure") == 0,
                "actual pinned staging or observer invalid")
    require(raw["formal_performance_claim"] is False and analysis["formal_goodput"] is False and
            analysis["slo"] is None and analysis["unseen_evaluation"] is False, "scope expanded")
    extra = config["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
    expected_policy = control_config["stage_policy"]
    require(same(expected_policy["mode"], entry["policy_mode"]), "frozen policy file mode mismatch")
    injected = copy.deepcopy(expected_policy)
    if entry["policy_mode"] != "off":
        injected["run_id"] = entry["label"]
    parent = dict(control_config["parent_admission"], run_id=entry["label"])
    require(same(extra["prefix_io_stage_policy"], injected) and
            same(extra["prefix_io_parent_admission"], parent) and
            extra["prefix_io_observation_run_id"] == entry["label"], "actual options differ from frozen controls")
    controls = actual_controls(raw, analysis)
    require(all((c["controller"] is None) if entry["policy_mode"] == "off" else
                (c["controller"] is not None and c["controller"]["mode"] == entry["policy_mode"])
                for c in controls), "off factory or actual controller mismatch")
    families = {f["name"]: f["tokens"] for f in manifest["families"]}
    golden = {(r["family"], r["kind"]): r for r in reference["rows"]}
    rows = raw["rows"]
    require(type(rows) is list and len(rows) == 10 and
            {r["request_id"] for r in rows} == {"c2-" + str(i) for i in range(10)}, "incomplete/duplicate cohort")
    for row in rows:
        kind = "cold" if integer(row["num_cached_tokens"], "cached tokens") == 0 else "gpu_hot"
        truth = golden[(row["family"], kind)]
        require(truth["prompt_token_ids"] == families[row["family"]] and
                row["output_tokens"] == truth["output_tokens"] and len(row["output_tokens"]) == 128,
                "full output/reference token mismatch")
        clocks, intervals = row["engine_token_timestamps"], row["itl_seconds"]
        require(row["per_token_complete"] is True and row["ambiguous_events"] == [] and
                len(clocks) == 128 and len(intervals) == 127 and row["metrics"]["is_corrupted"] is False,
                "ambiguous token metrics")
        for value in clocks:
            number(value, "token time")
        for delta, start, end in zip(intervals, clocks, clocks[1:]):
            number(delta, "ITL", positive=True)
            require(end > start and abs(delta - (end-start)) < 1e-9, "ITL/token clock mismatch")
    require(analysis["requests"] == 10 and analysis["output_tokens"] == 1280 and
            len(analysis["comparisons"]) == 10 and
            all(row["output_exact"] is True for row in analysis["comparisons"]), "analysis cohort incomplete")
    sampling = entry["native_observation"]
    require(sampling in ("on", "off") and raw["native_observation"] == analysis["native_observation"] == sampling,
            "optional observation mode mismatch")
    require(sampling != "off" or (entry["profile"] == "gpu_hot" and entry["policy_mode"] == "off"),
            "observation-off outside its approved GPU-hot U domain")
    for point in ("native_kv", "cohort_probe_start", "probe"):
        proof = raw[point]["optional_native_sampling"]
        require(proof["mode"] == sampling and proof["optional_sink_suppressed"] is (sampling == "off") and
                proof["physical_proof_requires_optional_sampling"] is False, "toggle affects hard proof")
    proof = raw["final_probe"]["optional_native_sampling"]
    require(proof["mode"] == sampling and proof["optional_sink_suppressed"] is False and
            proof["finish_restore_outside_cohort"] is True, "optional sink chain not restored")
    seconds = number(raw["cohort_seconds_including_drain"], "cohort seconds", positive=True)
    require(close(analysis["cohort_seconds_including_drain"], seconds), "analysis cohort differs")
    read_bytes = integer(raw["cohort_read_bytes"], "actual SSD read")
    write_bytes = integer(raw["cohort_write_bytes"], "actual SSD write")
    require(same(analysis["actual_read_bytes"], read_bytes) and same(analysis["actual_write_bytes"], write_bytes),
            "analysis omitted paid IO")
    if entry["profile"] == "gpu_hot":
        require(raw["gpu_hot_measurement_gate"] == "PASS_PREFIX16256_GOLDEN128_READ0_PAID_WRITE_CAP" and
                read_bytes == 0 and write_bytes <= 128 * 917504 and
                all(row["num_cached_tokens"] == 16256 for row in rows) and
                raw["gpu_hot_prewarm"]["external_pipeline_enabled"] is True,
                "GPU-hot negative control actually performed SSD loads or exceeded paid-write bound")
        require(all(c["native_io_size"] == 917504 for c in controls), "native IO unit changed")
    cpu = cpu_measurement(raw, analysis)
    pending_calls = integer(analysis["native_pending_flush_calls"], "pending flush calls")
    pending_seconds = number(analysis["native_pending_flush_seconds"], "pending flush seconds")
    unavailable = integer(analysis["foreground_slot_unavailable"], "slot unavailable")
    probe = raw["probe"]
    require(same(pending_calls, probe["pending_flush_wait_calls"]) and
            close(pending_seconds, probe["pending_flush_wait_seconds"]) and
            same(unavailable, probe["foreground_slot_unavailable"]), "wait analysis differs from actual probe")
    return dict(label=entry["label"], arm=entry["arm"], policy_mode=entry["policy_mode"],
                profile=entry["profile"], sampling=sampling, seconds=seconds,
                wrapped_thread_cpu_seconds=cpu, actual_read_bytes=read_bytes, actual_write_bytes=write_bytes,
                pending_flush_calls=pending_calls, pending_flush_seconds=pending_seconds,
                foreground_slot_unavailable=unavailable, projection=engine_projection(config),
                raw=raw, analysis=analysis, entry=entry)


def pressure_proof(run):
    live = run["raw"]["probe"]["simple_native_control"]
    require(type(live) is list and live, "pressure live owner samples absent")
    proofs = []
    for control in live:
        capacity = control["staging_capacity"]
        require(capacity["valid"] is True and capacity["observation_valid"] is True and
                capacity["error"] is None and capacity["physical_release_credit"] is False,
                "pressure capacity unknown/invalid")
        count = integer(capacity["cache_slots"], "cache registry") + integer(capacity["preload_cached_slots"], "preload registry")
        require(count > 64, "pressure production registry <=64")
        integer(capacity["free_reclaimable_staging_bytes"], "known clean capacity")
        require(control["owner_capture"] is True, "non-owner live capacity")
        policy = control["controller"]
        require(policy is not None and policy["mode"] == "pressure" and
                integer(policy["metadata_fallbacks"], "metadata fallback") == 0 and
                integer(policy["native_fallbacks"], "native fallback") == 0,
                "pressure fell back to native on unknown metadata")
        proofs.append(dict(registry_members=count, clean_capacity_bytes=capacity["free_reclaimable_staging_bytes"],
                           metadata_fallbacks=policy["metadata_fallbacks"], native_fallbacks=policy["native_fallbacks"]))
    for final in run["analysis"]["final_control"]:
        policy = final["controller"]
        require(policy["epoch_totals_are_lower_bounds"] is False and
                integer(policy["accepted_without_epoch_ops"], "accepted without known epoch") == 0 and
                integer(policy["metadata_fallbacks"], "final metadata fallback") == 0 and
                integer(policy["native_fallbacks"], "final native fallback") == 0, "final pressure fallback total nonzero")
    return proofs


def public_run(run):
    return {k: v for k, v in run.items() if k not in ("projection", "raw", "analysis", "entry")}


def arm_summary(runs):
    result = {}
    for arm in ARMS:
        values = [r for r in runs if r["arm"] == arm]
        require(len(values) == 2, "each common-domain arm needs two independent actual runs: " + arm)
        result[arm] = dict(labels=[v["label"] for v in values],
                           cohort_seconds=[v["seconds"] for v in values],
                           median_seconds=statistics.median(v["seconds"] for v in values))
    baseline = result["U"]["median_seconds"]
    for value in result.values():
        value["relative_duration_vs_U"] = value["median_seconds"]/baseline - 1
    winner = min(ARMS, key=lambda arm: (result[arm]["median_seconds"], ARMS.index(arm)))
    return result, winner


def audit_checks(evidence, entries, minimum=1):
    require(type(entries) is list and len(entries) >= minimum, "required producer receipts missing")
    result = []
    for entry in entries:
        value = evidence.read(entry["receipt"])
        require(type(entry.get("name")) is str and entry["name"], "named receipt required")
        checks = entry["checks"]
        require(type(checks) is list and checks, "producer predicates required")
        for check in checks:
            require(set(check) == {"field", "equals"} and same(lookup(value, check["field"]), check["equals"]),
                    "producer predicate failed: " + entry["name"] + "/" + str(check.get("field")))
        result.append(dict(name=entry["name"], receipt=entry["receipt"], checks=checks))
    return result


def native_qualifications(evidence, entries, gpu_uuid):
    require(type(entries) is list and entries, "actual P316 native qualification absent")
    result = []
    for entry in entries:
        gpu_wrapper(evidence, entry, gpu_uuid)
        value = evidence.read(entry["result"])
        require(type(value["status"]) is str and value["status"].startswith("PASS") and
                value["status"] == entry["required_status"], "native qualification failed")
        require(type(entry["checks"]) is list and entry["checks"], "physical qualification predicates absent")
        for check in entry["checks"]:
            require(same(lookup(value, check["field"]), check["equals"]), "native qualification predicate failed")
        result.append(dict(label=entry["label"], result=entry["result"],
                           status=value["status"], checks=entry["checks"]))
    return result


def normalized_capacity_domain(value):
    """Only the GiB JSON numeric representation is normalized; bytes/counts stay typed."""
    keys = {"capacity_domain_id", "staging_bytes", "staging_mem_gib",
            "preload_lookahead_requests", "io_depth", "slot_count", "cache_preload_ceiling_slots"}
    require(type(value) is dict and set(value) == keys, "capacity exact seven-field domain required")
    domain_id = value["capacity_domain_id"]
    require(type(domain_id) is str and domain_id in CAPACITY_IDS,
            "capacity declared outside two exact frozen domains")
    mib, horizon = (960, 1) if domain_id == "cap960-l1" else (1024, 2)
    budget = mib * 1024**2
    slots = (budget - 4095) // 917504
    expected = dict(capacity_domain_id=domain_id, staging_bytes=budget, staging_mem_gib=mib/1024,
                    preload_lookahead_requests=horizon, io_depth=8, slot_count=slots,
                    cache_preload_ceiling_slots=slots-8)
    gib = number(value["staging_mem_gib"], "capacity GiB", positive=True)
    require(gib == budget / 1024**3, "capacity GiB/physical bytes mismatch")
    normalized = dict(value, staging_mem_gib=float(gib))
    require(same(normalized, expected), "capacity declared outside two exact frozen domains")
    return normalized


def same_capacity_domain(left, right):
    return same(normalized_capacity_domain(left), normalized_capacity_domain(right))


def calibration_native_source_scope(full_sources, observed_sources, native_worktree):
    """Runtime package scope only here; all 31 files still bind CPU and patch proof."""
    require(type(native_worktree) is str and native_worktree and
            "\\" not in native_worktree, "calibration native worktree required")
    worktree = PurePosixPath(native_worktree)
    require(not worktree.is_absolute() and ".." not in worktree.parts and
            str(worktree) == native_worktree, "unsafe calibration worktree")
    require(type(full_sources) is dict and full_sources and type(observed_sources) is dict,
            "calibration source maps required")
    runtime, tests = {}, {}
    for path, digest in full_sources.items():
        require(type(path) is str and "\\" not in path and type(digest) is str and SHA.fullmatch(digest),
                "invalid full native source identity")
        rel = PurePosixPath(path)
        require(not rel.is_absolute() and ".." not in rel.parts and str(rel) == path and rel.suffix == ".py",
                "unsafe native source identity")
        try:
            tail = rel.relative_to(worktree)
        except ValueError as exc:
            raise AuditError("native identity outside calibration worktree") from exc
        require(len(tail.parts) >= 2 and tail.parts[0] in ("py_kvcache", "tests"),
                "unclassified native source scope")
        (runtime if tail.parts[0] == "py_kvcache" else tests)[path] = digest
    require(len(runtime) == 15 and len(tests) == 16, "P316 full native runtime/test coverage must be 15/16")
    observed_runtime = {}
    for path, digest in observed_sources.items():
        require(type(path) is str and type(digest) is str and SHA.fullmatch(digest),
                "invalid calibration source identity")
        if path == native_worktree or path.startswith(native_worktree + "/"):
            require("\\" not in path, "unsafe calibration native source path")
            rel = PurePosixPath(path)
            require(not rel.is_absolute() and ".." not in rel.parts and str(rel) == path and rel.suffix == ".py",
                    "unsafe calibration native source path")
            tail = rel.relative_to(worktree)
            require(len(tail.parts) >= 2 and tail.parts[0] in ("py_kvcache", "tests"),
                    "unclassified calibration native source")
            if tail.parts[0] == "py_kvcache":
                observed_runtime[path] = digest
            else:
                require(path in tests and tests[path] == digest, "calibration optional test source differs")
    require(same(observed_runtime, runtime), "calibration runtime native source coverage/hash differs")
    return dict(native_worktree=native_worktree, runtime_source_count=len(runtime),
                test_source_count=len(tests), calibration_scope="runtime_py_kvcache_only",
                full_test_hashes_remain_required_by_CPU_and_patch=True)


P316_EXECUTION_LOCK_SHA = "afd322a2795d045934e206334a3920fc38edf9dd2957bcaae87cabc9c81f00fd"
CALIBRATION_OTHER_SOURCES = {
    "experiments/prefix_io_v1/scripts/analyze_decode_interference_calibration_p316.py",
    "experiments/prefix_io_v1/scripts/decode_interference_worker_probe_p316.py",
    "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py",
    "experiments/prefix_io_v1/scripts/run_decode_interference_calibration_p316.py"}


def _relative_python_source(path):
    require(type(path) is str and path and "\\" not in path, "canonical Python source path required")
    rel = PurePosixPath(path)
    require(not rel.is_absolute() and ".." not in rel.parts and str(rel) == path and
            rel.suffix == ".py", "unsafe/noncanonical Python source path")
    return path


P316_AUTHENTICATED_HISTORY_REFERENCES = {
    "/root/prefix-io-v1-validation/runs/server07-p3-c2-fullref-01/details/result.json"}


def _locked_source_map(root, values):
    require(type(values) is dict and values, "locked input map required")
    normalized = {}
    for path, digest in values.items():
        require(type(path) is str and path and "\\" not in path and
                type(digest) is str and SHA.fullmatch(digest), "invalid locked path/SHA")
        rel = PurePosixPath(path)
        require(".." not in rel.parts and str(rel) == path, "noncanonical locked path")
        if rel.is_absolute():
            try:
                rel = rel.relative_to(PurePosixPath(str(root)))
            except ValueError:
                # This exact lock08 metadata reference is not executable/runtime source.
                # The caller pins and reads lock08 before comparing the complete producer map.
                require(path in P316_AUTHENTICATED_HISTORY_REFERENCES,
                        "locked runtime/unregistered reference outside project root")
        key = str(rel)
        require(key and key != "." and key not in normalized, "locked path alias/duplicate")
        normalized[key] = digest
    return normalized


def calibration_source_anchor(evidence, manifest):
    """Use recorded byte-verification producer and the independently frozen lock08."""
    entries = [x for x in manifest["delivery_receipts"] if x.get("name") == "source_lock"]
    require(len(entries) == 1, "unique source-lock producer required")
    receipt = evidence.read(entries[0]["receipt"])
    check = lookup(receipt, ["checks", "source_lock"])
    require(check["evidence_complete"] is True and type(check["producer_refs"]) is list and
            check["producer_refs"], "actual source-lock evidence incomplete")
    source_ref = check["producer_refs"][0]
    source = evidence.read(source_ref)
    require(same(source["schema_version"], 1) and
            source["status"] == "PASS_FROZEN_INPUT_BYTE_IDENTITIES_AT_STORAGE_BLOCK" and
            source["all_hashes_match"] is True and same(source["locked_input_count"], 2599),
            "source byte verification absent/failed")
    lock_ref = source["execution_lock"]
    require(type(lock_ref) is dict and lock_ref.get("sha256") == P316_EXECUTION_LOCK_SHA,
            "calibration source lock is not the frozen P316 lock08")
    bound_path(evidence, lock_ref["path"], "artifacts/prefix_io_v1/server08-p3-16/execution-lock-08.json")
    locked = _locked_source_map(evidence.root, evidence.read(lock_ref))
    verified = _locked_source_map(evidence.root, source["input_files"])
    require(len(locked) == 2599 and same(verified, locked), "verified input identities differ from frozen lock08")
    prefixes = {"native": "third_party/work/py-kvcache-p3-16-cpu/py_kvcache/",
                "control": "src/prefix_io_control/",
                "vllm": "third_party/work/vllm-author-build/vllm/"}
    expected, counts = {}, {}
    for name, prefix in prefixes.items():
        selected = {k: v for k, v in locked.items() if k.startswith(prefix) and k.endswith(".py")}
        for path in selected:
            _relative_python_source(path)
        expected.update(selected)
        counts[name] = len(selected)
    require(same(counts, {"native": 15, "control": 21, "vllm": 1844}),
            "frozen calibration runtime tree coverage changed")
    for path in CALIBRATION_OTHER_SOURCES:
        require(path in locked, "frozen calibration script source missing")
        expected[path] = locked[path]
    require(len(expected) == 1884, "complete frozen calibration source coverage missing")
    full = manifest["native_source_hashes"]
    require(type(full) is dict and len(full) == 31, "full P316 native source map must contain 31 files")
    for path, digest in full.items():
        require(path in locked and locked[path] == digest, "native full-map SHA differs from frozen lock08")
    return expected, dict(source_lock_producer=source_ref, execution_lock=lock_ref,
                          anchored_runtime_source_count=1884, verified_locked_input_count=2599)


def anchored_calibration_sources(expected, observed):
    require(type(observed) is dict, "recorded calibration source map required")
    for path, digest in observed.items():
        _relative_python_source(path)
        require(type(digest) is str and SHA.fullmatch(digest), "invalid calibration runtime SHA")
    require(same(observed, expected), "calibration full runtime source map differs from frozen lock08")


def phase_receipts(evidence, gate, gpu_uuid):
    disposition = gate["disposition"]
    phases = gate["phases"]
    require(type(phases) is list and len(phases) == (2 if disposition == "FAILED_NUMERIC_GATE" else 6),
            "capacity original phase count incomplete")
    stamps = []
    for phase in phases:
        wrapper = gpu_wrapper(evidence, phase, gpu_uuid)
        raw = evidence.read(phase["result"])
        config = evidence.read(phase["config"])
        require(type(raw["status"]) is str and raw["status"].startswith("PASS"), "actual capacity acquisition failed")
        require(config["gpu_uuid"] == gpu_uuid and
                raw["capacity_domain_id"] == config["capacity_domain_id"] == gate["capacity_domain_id"] and
                same_capacity_domain(raw["capacity_domain"], gate["capacity_domain"]) and
                same_capacity_domain(config["capacity_domain"], gate["capacity_domain"]), "actual capacity geometry differs")
        require(any(Path(x).name == "acquire_native_aio_costs_capacity_p316.py" for x in wrapper["command"]) and
                flag(wrapper["command"], "--capacity-domain") == gate["capacity_domain_id"],
                "capacity phase not guarded NEW original acquisition")
        stamps.append(dict(label=phase["label"], result_sha256=phase["result"]["sha256"],
                           config_sha256=phase["config"]["sha256"], wrapper_sha256=phase["wrapper"]["sha256"]))
    return stamps


def capacity_gate(evidence, gate, gpu_uuid):
    require(gate["capacity_domain_id"] in CAPACITY_IDS and
            gate["disposition"] in ("PASS", "FAILED_NUMERIC_GATE", "FAILED_COST_GATE"), "capacity disposition not an evidence-based gate/prune")
    domain = normalized_capacity_domain(gate["capacity_domain"])
    require(domain["capacity_domain_id"] == gate["capacity_domain_id"], "capacity domain mismatch")
    receipts = phase_receipts(evidence, gate, gpu_uuid)
    reference = evidence.read(gate["reference_result"])
    require(reference["capacity_domain_id"] == gate["capacity_domain_id"] and
            same_capacity_domain(reference["capacity_domain"], domain), "numeric reference source/domain differs")
    comparisons = reference["comparisons"]
    require(type(comparisons) is list and len(comparisons) == 6, "original numeric reference coverage missing")
    numeric_success = []
    for row in comparisons:
        require(type(row["output_equal"]) is bool and type(row["top5_exact_equal"]) is bool and
                type(row["token_set_equal"]) is bool, "numeric comparison not classified")
        error = number(row["max_common_logprob_delta"], "logprob delta")
        numeric_success.append(row["output_equal"] and row["top5_exact_equal"] and row["token_set_equal"] and error == 0)
    if gate["disposition"] == "FAILED_NUMERIC_GATE":
        require(reference["status"] != "PASSED_EXACT_CACHED_REFERENCE" and not all(numeric_success),
                "numeric prune has no actual failed original comparison")
        return dict(domain=domain, disposition=gate["disposition"], receipts=receipts,
                    reason="actual exact numeric gate failed; no loaded screen required")
    require(reference["status"] == "PASSED_EXACT_CACHED_REFERENCE" and all(numeric_success) and
            reference["cached_output_tolerance"] == 0 and reference["latency_fit_allowed"] is False and
            reference["runtime_curve_exported"] is False, "original capacity reference gate failed")
    cost = evidence.read(gate["cost_result"])
    require(cost["capacity_domain_id"] == gate["capacity_domain_id"] and same_capacity_domain(cost["capacity_domain"], domain) and
            cost["runtime_curve_updated"] is False and cost["formal_performance_claim"] is False and
            cost["independent_content_validation"] is True, "capacity gate domain/scope changed")
    require(set(cost["checks"]) == {"f", "g_ssd", "g_mem"} and integer(cost["checked_rows"], "cost rows") == 18,
            "cost point paths/rows incomplete")
    success = []
    summaries = {}
    for path, lengths in cost["checks"].items():
        require(set(lengths) == {"16256"}, "unqualified cost point extrapolation")
        row = lengths["16256"]
        samples = row["samples_s"]
        require(type(samples) is list and len(samples) == 4 and integer(row["n"], "cost repetitions") == 4,
                "cost independent four-session coverage missing")
        samples = [number(v, "cost sample", positive=True) for v in samples]
        prediction = number(row["prediction_s"], "prediction", positive=True)
        observed = statistics.median(samples)
        signed_error = (observed-prediction)/prediction
        passed = abs(signed_error) <= .25
        require(type(row["relative_error"]) in (int, float) and math.isfinite(row["relative_error"]), "invalid signed point error")
        require(close(row["observed_median_s"], observed) and
                math.isclose(row["relative_error"], signed_error, rel_tol=1e-10, abs_tol=1e-12) and
                close(row["absolute_relative_error"], abs(signed_error)) and row["passed"] is passed,
                "cost point error/classification differs from original 25% threshold")
        success.append(passed)
        summaries[path] = dict(prediction_s=prediction, observed_median_s=observed,
                               absolute_relative_error=abs(signed_error), passed=passed)
    if gate["disposition"] == "FAILED_COST_GATE":
        require(cost["status"] == "FAILED_HELDOUT_POINT_PREDICTION" and not all(success),
                "cost prune lacks actual original-threshold failure")
    else:
        require(cost["status"] == "PASSED_HELDOUT_POINT_PREDICTION" and all(success), "capacity point did not pass")
        permit = evidence.read(gate["permit"])
        require(permit["status"] == "PASSED_CAPACITY_UNLOADED_POINT_GATE" and
                permit["capacity_domain_id"] == gate["capacity_domain_id"] and
                same_capacity_domain(permit["capacity_domain"], domain) and
                permit["maximum_absolute_relative_median_error"] == .25 and
                permit["cost_result_sha256"] == gate["cost_result"]["sha256"] and
                permit["reference_result_sha256"] == gate["reference_result"]["sha256"] and
                permit["ordinary_quotas_installed"] is False and permit["formal_goodput"] is False and
                same(permit["guarded_gpu_receipts"], cost["guarded_gpu_receipts"]),
                "capacity permit changed/lacks original qualification")
        bound_path(evidence, permit["cost_result"], gate["cost_result"]["path"])
    recorded = cost["guarded_gpu_receipts"]
    require(type(recorded) is list and len(recorded) == 6 and
            {x["label"] for x in recorded} == {x["label"] for x in receipts}, "cost receipt provenance differs")
    sources = reference["sources"] + cost["sources"]
    require(len(sources) == 6 and {x["label"] for x in sources} == {x["label"] for x in receipts}, "capacity source set differs")
    for source in sources:
        actual = next(x for x in receipts if x["label"] == source["label"])
        require(source["result_sha256"] == actual["result_sha256"] and
                source["config_sha256"] == actual["config_sha256"], "capacity source SHA not exact")
    return dict(domain=domain, disposition=gate["disposition"], receipts=receipts,
                point_checks=summaries, loaded_prediction_qualified=False)


def conditional_table(evidence, manifest):
    runs = manifest["calibration_runs"]
    require(type(runs) is list and len(runs) == 3 and
            [r["partition"] for r in runs] == ["calibration", "calibration", "validation"], "sparse calibration independent split missing")
    actual = []
    expected_sources, source_anchor = calibration_source_anchor(evidence, manifest)
    for run in runs:
        gpu_wrapper(evidence, run, manifest["gpu_uuid"])
        value = evidence.read(run["result"])
        require(type(value["status"]) is str and value["status"].startswith("PASS") and
                value["status"] == run["required_status"], "calibration input failed or producer status changed")
        require(value["spec"]["partition"] == run["partition"] and
                value["spec"]["session_id"] == run["label"] and
                value["spec"]["gpu_uuid"] == manifest["gpu_uuid"] and
                value["engine_shutdown"] == "completed", "calibration partition/identity/drain mismatch")
        anchored_calibration_sources(expected_sources, value["spec"]["source_sha256"])
        source_scope = calibration_native_source_scope(manifest["native_source_hashes"],
            value["spec"]["source_sha256"], value["spec"]["native_worktree"])
        actual.append(dict(label=run["label"], path=str(evidence.path(run["result"]["path"])),
                           sha256=run["result"]["sha256"], partition=run["partition"], source_scope=source_scope))
    table = evidence.read(manifest["conditional_table"])
    require(table["status"] == "PASS_COMPLETE_CONDITIONAL_TABLE" and table["complete_conditional_table"] is True and
            table["production_state_table_complete"] is False and table["p4_connected"] is False and
            table["formal_performance_claim"] is False and table["confidence_interval"] is None,
            "conditional calibration incomplete or promoted to production/formal claim")
    require(table["domain"]["gpu_uuid"] == manifest["gpu_uuid"], "conditional table GPU domain differs")
    anchored_calibration_sources(expected_sources, table["domain"]["source_sha256"])
    table_scope = calibration_native_source_scope(manifest["native_source_hashes"],
        table["domain"]["source_sha256"], table["domain"]["native_worktree"])
    require(all(same(run["source_scope"], table_scope) for run in actual),
            "calibration/table native runtime scopes differ")
    table_inputs = table["inputs"]
    require(len(table_inputs) == 3, "table input coverage")
    for run, recorded in zip(actual, table_inputs):
        require(recorded["sha256"] == run["sha256"] and recorded["partition"] == run["partition"], "table actual run provenance differs")
        bound_path(evidence, recorded["path"], run["path"])
    cells = table["cells"]
    require(len(cells) == 7 and {str(c["cell"]["anchor"])+str(c["cell"]["units"]) for c in cells} == CELL_IDS, "seven sparse cells missing")
    result = []
    for cell in cells:
        values = [number(v, "calibration loaded ITL", positive=True) for v in cell["calibration_run_loaded_values"]]
        require(len(values) == 2, "independent calibration repetitions missing")
        predicted = statistics.median(values)
        observed = number(cell["validation_loaded_itl_seconds"], "validation ITL", positive=True)
        error = abs(predicted-observed)/observed
        spread = abs(values[0]-values[1])/predicted
        drifts = [number(v, "baseline drift") for v in cell["three_run_baseline_relative_drifts"]]
        require(len(drifts) == 3, "baseline drift coverage missing")
        supported = error <= .25 and spread <= .25 and max(drifts) <= .25
        require(close(cell["validation_loaded_relative_error"], error) and
                close(cell["calibration_loaded_relative_spread"], spread) and cell["supported"] is supported,
                "conditional stability classification differs from three frozen 25% thresholds")
        if supported:
            require(close(cell["conditional_loaded_itl_seconds"], predicted), "supported prediction differs")
        else:
            require(cell["conditional_loaded_itl_seconds"] is None, "unsupported lookup fabricated")
        require(cell["production_value"] is None and cell["p4_connected"] is False and
                cell["confidence_interval"] is None and cell["delta_prediction"] is None,
                "conditional estimate became a production/effect estimate")
        result.append(dict(cell=cell["cell"], supported=supported, loaded_error=error,
                           calibration_spread=spread, baseline_drifts=drifts))
    require(all(c["supported"] for c in result), "sparse table has unsupported cells")
    return dict(cells=result, conditional_domain_only=True, production_state_table_complete=False,
                p4_connected=False, independent_calibrations=2, independent_validation=1,
                span_and_handler_quantity_independently_regated=False,
                native_source_scope=table_scope, frozen_source_anchor=source_anchor)


def guard(value):
    require(value["cuda_initialized"] is False and integer(value["gpu_workloads_run"], "CPU GPU work") == 0 and
            type(value["pytest_exit"]) is int and value["pytest_exit"] == 0, "CPU guard/test failed")


def cpu_receipts(evidence, manifest):
    entries = manifest["cpu_receipts"]
    require(type(entries) is list and entries, "CPU evidence absent")
    summaries = []
    for entry in entries:
        value = evidence.read(entry["receipt"])
        if entry["kind"] == "matrix":
            require(value["status"] == "PASS_CPU_SIMPLE_STAGE" and value["complete_historical_matrix"] is True and
                    value["cuda_initialized"] is False and integer(value["gpu_runs"], "CPU GPU count") == 0,
                    "complete historical CPU matrix absent")
            counts = value["unique_counts"]
            integer(counts["passed"], "unique CPU passed", 1)
            integer(counts["skipped"], "CPU skips")
            require(integer(counts["failed"], "CPU failures") == 0, "CPU matrix failure")
            processes = value["processes"]
            require(processes and len(value["cuda_guard_proofs"]) == len(processes), "CPU process guard coverage")
            for process in processes:
                require(type(process["exit"]) is int and process["exit"] == 0 and process["error"] is None and
                        process["cleanup"]["session_drained"] is True and
                        process["cleanup"]["session_members_after_cleanup"] == [], "CPU process not successful/drained")
            for proof in value["cuda_guard_proofs"]:
                require(proof["verified"] is True, "unverified guard")
                guard(proof["proof"])
            supplemental = value.get("supplemental_pilot_contract")
            if supplemental is not None:
                guard(supplemental["cuda_guard"])
                require(integer(supplemental["failed"], "CPU supplemental failures") == 0 and
                        integer(supplemental["unique_passed"], "CPU supplemental passed", 1) > 0,
                        "matrix includes failed supplemental pilot contracts")
            for path, digest in manifest["native_source_hashes"].items():
                require(value["latest_native_sha256"][path] == digest, "CPU source identity differs")
            summaries.append(dict(kind="matrix", receipt=entry["receipt"], unique_counts=counts,
                                  claimed_total_unique_passed=value.get("total_unique_passed")))
        elif entry["kind"] == "command":
            require(type(value["exit"]) is int and value["exit"] == 0 and value.get("error") is None, "supplemental CPU command failed")
            proof = evidence.read(entry["guard"])
            guard(proof)
            summaries.append(dict(kind="command", receipt=entry["receipt"], guard=entry["guard"],
                                  count_not_inferred_from_stdout=True))
        else:
            raise AuditError("unknown CPU receipt kind")
    require(any(e["kind"] == "matrix" for e in entries), "historical/source-bound CPU matrix required")
    return dict(receipts=summaries, counts_not_summed_across_duplicate_suites=True)


def patch_roundtrip(evidence, manifest):
    value = evidence.read(manifest["patch_roundtrip"])
    require(value["status"] == "PASS_PATCH_ROUNDTRIP" and
            type(value["check_exit"]) is int and value["check_exit"] == 0 and
            type(value["apply_exit"]) is int and value["apply_exit"] == 0 and value["exact_content"] is True,
            "actual patch roundtrip not successful")
    files = value["files"]
    require(type(files) is list and files and len({f["path"] for f in files}) == len(files), "patch exact file coverage")
    for row in files:
        require(type(row["expected_sha256"]) is str and SHA.fullmatch(row["expected_sha256"]) and
                row["actual_sha256"] == row["expected_sha256"], "roundtrip content mismatch")
    for path, digest in manifest["native_source_hashes"].items():
        require(any(f["path"] == path and f["expected_sha256"] == digest for f in files),
                "roundtrip did not prove actual P316 native source")
    checks = value["individual_checks"]
    require(set(checks) == {"common_only", "full", "default_off", "reverse"} and
            all(v["passed"] is True for v in checks.values()), "composition/default-off/reverse patch check incomplete")
    return dict(files=files, actual_roundtrip=True,
                individual_checks={k: v["passed"] for k, v in checks.items()},
                CPU_composition_is_not_GPU_qualification=True)


def analyze(root, manifest):
    evidence = Evidence(root)
    output = dict(schema_version=1, status="P3_INCOMPLETE", p3_phase_closed=False,
                  scope="Finite P3 engineering/development closeout; final root review remains required",
                  formal_efficacy_claim=False, global_optimum_claim=False, confidence_interval=None,
                  p4_enabled=False, gates=[], unmet_gates=[], evidence_inputs={})
    results = {}
    def gate(name, action):
        try:
            results[name] = action()
            output["gates"].append(dict(name=name, status="PASS", evidence=results[name]))
        except (AuditError, KeyError, TypeError, ValueError, OSError, StopIteration, ZeroDivisionError) as exc:
            reason = type(exc).__name__ + ": " + str(exc)
            output["gates"].append(dict(name=name, status="INCOMPLETE", reason=reason))
            output["unmet_gates"].append(dict(name=name, reason=reason))
    def header():
        require(manifest["schema_version"] == 1 and type(manifest["schema_version"]) is int, "manifest schema")
        require(type(manifest["gpu_uuid"]) is str and manifest["gpu_uuid"].startswith("GPU-"), "owned GPU UUID required")
        require(type(manifest["native_source_hashes"]) is dict and manifest["native_source_hashes"] and
                all(type(k) is str and type(v) is str and SHA.fullmatch(v)
                    for k, v in manifest["native_source_hashes"].items()), "native source identity absent")
        review = evidence.read(manifest["requirements_review"])
        require(len(review["mandatory_p3_gates"]) == 11, "complete author/user P3 requirements absent")
        prereg = evidence.read(manifest["preregistration"])
        require(prereg["core_order"] == CORE_ORDER and prereg["supplement_order"] == FINITE_ORDER and
                prereg["capacity_order"] == list(CAPACITY_IDS), "frozen finite schedule changed")
        require(prereg["formal_performance_claim"] is False and prereg["p4_enabled"] is False,
                "preregistration research scope expanded")
        for ref in prereg["configs"]:
            evidence.read(ref)
        for ref in prereg["plans"]:
            evidence.read(ref)
        results["prereg_private"] = prereg
        return dict(preregistration=manifest["preregistration"], requirements=manifest["requirements_review"],
                    candidate_rule="2 runs per common 1GiB arm; minimum cohort median; tie U,F16,P4,F8,P512; capacity excluded")
    gate("requirements_and_preregistration", header)
    def model_group(key, expected, profile=None):
        entries = manifest[key]
        require(type(entries) is list and len(entries) == len(expected) and
                [r["arm"] for r in entries] == expected, "missing/frozen run order changed: " + key)
        if key == "finite_runs":
            prereg = results["prereg_private"]
            expected_configs = {"F8": prereg["configs"][0]["sha256"], "P512": prereg["configs"][1]["sha256"]}
            require(all(e["control_config"]["sha256"] == expected_configs[e["arm"]] for e in entries),
                    "finite controls differ from prospective registration")
        runs = [model_run(evidence, e, manifest["gpu_uuid"]) for e in entries]
        if profile is not None:
            require(all(r["profile"] == profile for r in runs), "group profile changed")
        results[key + "_private"] = runs
        return [public_run(r) for r in runs]
    gate("core_models", lambda: model_group("core_runs", CORE_ORDER))
    gate("finite_models", lambda: model_group("finite_runs", FINITE_ORDER))
    def common():
        runs = results["core_runs_private"] + results["finite_runs_private"]
        require(all(r["profile"] != "gpu_hot" and r["sampling"] == "on" for r in runs), "normal cohorts not normal sampled domain")
        require(all(same(r["projection"], runs[0]["projection"]) and
                    r["entry"]["workload_manifest"]["sha256"] == runs[0]["entry"]["workload_manifest"]["sha256"]
                    for r in runs), "common arm resources/model/source workload differ")
        summary, winner = arm_summary(runs)
        results["winner"] = winner
        return dict(arms=summary, selected_arm=winner, selection_is_descriptive_development=True,
                    independent_unit="run", capacity_sensitivity_pooled=False, confidence_interval=None)
    gate("comparable_common_domain_and_selection", common)
    def pressures():
        runs = results["core_runs_private"] + results["finite_runs_private"]
        selected = [r for r in runs if r["policy_mode"] == "pressure"]
        require(len(selected) == 4, "four actual pressure runs absent")
        return [dict(label=r["label"], live_capacity=pressure_proof(r),
                     denied_performance=[c["controller"]["denied_performance"] for c in r["analysis"]["final_control"]],
                     nonbinding_is_not_an_efficacy_failure=True) for r in selected]
    gate("pressure_known_over64_no_unknown_fallback", pressures)
    def waits():
        runs = [r for r in results["core_runs_private"] if r["arm"] == "U"]
        require(len(runs) == 2, "ordinary independent U repeats missing")
        description = []
        for run in runs:
            diag = run["raw"]["probe"]["flush_diagnostic"]
            require(diag["faulted"] is False and type(diag["records"]) is list, "flush witness invalid")
            records = diag["records"]
            description.append(dict(label=run["label"], pending_flush_calls=run["pending_flush_calls"],
                pending_flush_seconds=run["pending_flush_seconds"],
                foreground_slot_unavailable=run["foreground_slot_unavailable"],
                bounded_records=len(records), dropped_records=diag["dropped_records"],
                retired_parent_records=sum(r.get("kind") == "native_parent_retired" for r in records),
                flush_batch_records=sum(r.get("kind") == "flush_batch" for r in records),
                immediately_reusable_bytes=diag["immediately_reusable_bytes"],
                resource_release_credit=False))
        repeated = all(r["pending_flush_calls"] > 0 and r["pending_flush_seconds"] > 0 for r in runs)
        results["repeatable_wait"] = repeated
        return dict(runs=description, repeated_pending_flush_target=repeated,
                    no_repeatable_target_stop_justified=False,
                    key_path="native pending parent flush -> original copy/AIO completion -> verified full parent retirement; source active refs/generation can prevent GPU block reuse",
                    source_protection_is_not_new_free_GPU_capacity=True,
                    bounded_samples_are_not_integrated_wait_or_release_credit=True)
    gate("ordinary_wait_and_resource_path", waits)
    def capacities():
        entries = manifest["capacity_gates"]
        require(type(entries) is list and len(entries) == 2 and
                [e["capacity_domain_id"] for e in entries] == list(CAPACITY_IDS), "two finite capacity domains absent")
        handled = [capacity_gate(evidence, e, manifest["gpu_uuid"]) for e in entries]
        passed = {x["domain"]["capacity_domain_id"] for x in handled if x["disposition"] == "PASS"}
        screens = manifest["capacity_runs"]
        require(type(screens) is list and len(screens) <= 2 and
                {e["capacity_domain_id"] for e in screens} == passed and len(screens) == len(passed),
                "each passed capacity requires one loaded U sensitivity; failed capacity forbids a loaded screen")
        runs = [model_run(evidence, e, manifest["gpu_uuid"]) for e in screens]
        require(all(r["arm"] == "U" and r["profile"] != "gpu_hot" and r["sampling"] == "on" for r in runs), "capacity screen is not independent normal native U")
        return dict(domains=handled, independent_loaded_U_sensitivities=[public_run(r) for r in runs],
                    pooled_with_winner=False, full_domain_cost_prediction_qualified=False,
                    resource_blockage_is_not_scientific_pruning=True)
    gate("finite_capacity_and_preload_disposition", capacities)
    gate("sparse_decode_conditional_calibration", lambda: conditional_table(evidence, manifest))
    def hot():
        winner = results["winner"]
        runs = model_group("hot_runs", ["U", winner, winner, "U"], "gpu_hot")
        private = results["hot_runs_private"]
        require(all(r["sampling"] == "on" for r in private), "hot winner screen mixes optional toggles")
        require(all(same(r["projection"], private[0]["projection"]) for r in private), "hot common resources differ")
        baselines = [r["seconds"] for r in private if r["entry"] is private[0]["entry"] or r["entry"] is private[-1]["entry"]]
        candidate = [private[1]["seconds"], private[2]["seconds"]]
        degrade = statistics.median(candidate)/statistics.median(baselines)-1
        return dict(runs=runs, selected_arm=winner, relative_duration_degradation=degrade,
                    internal_engineering_target=.02, internal_target_met=degrade <= .02,
                    target_is_not_formal_SLO=True, gain_claim=False)
    gate("gpu_hot_selected_baseline_negative_control", hot)
    def observer():
        model_group("observation_runs", ["U"] * 4, "gpu_hot")
        runs = results["observation_runs_private"]
        require([r["sampling"] for r in runs] == ["off", "on", "on", "off"], "observer frozen order differs")
        require(all(same(r["projection"], runs[0]["projection"]) for r in runs), "observer arm resources differ")
        values = {}
        for mode in ("off", "on"):
            arm = [r for r in runs if r["sampling"] == mode]
            values[mode] = dict(cohort_median_seconds=statistics.median(r["seconds"] for r in arm),
                wrapped_thread_cpu_median_seconds=statistics.median(r["wrapped_thread_cpu_seconds"] for r in arm),
                labels=[r["label"] for r in arm])
        degradation = values["on"]["cohort_median_seconds"]/values["off"]["cohort_median_seconds"]-1
        return dict(runs=[public_run(r) for r in runs], modes=values, relative_duration_degradation=degradation,
                    wrapped_thread_cpu_difference_seconds=values["on"]["wrapped_thread_cpu_median_seconds"]-values["off"]["wrapped_thread_cpu_median_seconds"],
                    internal_engineering_target=.02, internal_target_met=degradation <= .02,
                    isolated_scope="optional native sink suppression; common admission/capacity and unwrapped bookkeeping remain in both arms",
                    wrapped_CPU_scope="outermost metadata/controller/observer hooks; timer overhead included; backend/model/unwrapped bookkeeping excluded",
                    whole_model_CPU_quantified=False, all_common_observation_cost_isolated=False)
    gate("optional_observation_off_on_CPU_and_end_to_end", observer)
    gate("CPU_matrix_and_supplements", lambda: cpu_receipts(evidence, manifest))
    gate("actual_patch_roundtrip", lambda: patch_roundtrip(evidence, manifest))
    gate("real_native_transfer_shutdown_qualification", lambda: native_qualifications(evidence, manifest["native_qualifications"], manifest["gpu_uuid"]))
    def lineage():
        names = {entry["name"] for entry in manifest["lineage_receipts"]}
        require({"batch", "iodepth", "cache_capacity", "preload_capacity", "reserve", "stage_quota"} <= names,
                "all six finite baseline dimensions need actual named lineage dispositions")
        return audit_checks(evidence, manifest["lineage_receipts"], minimum=6)
    gate("bounded_batch_depth_capacity_reserve_lineage", lineage)
    def delivery():
        names = {entry["name"] for entry in manifest["delivery_receipts"]}
        require({"source_lock", "version_lock", "GPU_budget_idle", "storage_and_source_preserved",
                 "problem_report", "reproduction"} <= names,
                "final source/version/budget/storage/problem-report/reproduction evidence required")
        return audit_checks(evidence, manifest["delivery_receipts"], minimum=6)
    gate("source_locks_budget_idle_and_final_delivery", delivery)
    output["evidence_inputs"] = evidence.inputs
    output["descriptive_arm_medians"] = results.get("comparable_common_domain_and_selection")
    output["actual_GPU_sessions_audited"] = len(evidence.successful_gpu_sessions)
    output["actual_GPU_session_receipts"] = evidence.successful_gpu_sessions
    output["target_waits"] = results.get("ordinary_wait_and_resource_path")
    output["pressure_observation"] = results.get("pressure_known_over64_no_unknown_fallback")
    output["conditional_interference_table"] = results.get("sparse_decode_conditional_calibration")
    output["optional_toggle_and_CPU_boundaries"] = results.get("optional_observation_off_on_CPU_and_end_to_end")
    complete = not output["unmet_gates"]
    negative_flags = []
    for key in ("gpu_hot_selected_baseline_negative_control", "optional_observation_off_on_CPU_and_end_to_end"):
        if key in results and results[key]["internal_target_met"] is False:
            negative_flags.append(key + ": frozen descriptive 2% engineering target not met")
    output["engineering_negative_flags"] = negative_flags
    output["status"] = "P3_EVIDENCE_COMPLETE_FOR_ROOT_REVIEW" if complete else "P3_INCOMPLETE"
    output["research_effect_disposition"] = "UNPROVEN: finite descriptive development only; no formal unseen gain/CI or P4 policy evaluated"
    output["p4_next_allowed"] = dict(
        automatic_activation=False, requirements_evidence_complete=complete,
        repeated_ordinary_wait_target=results.get("repeatable_wait"),
        engineering_negative_flags=negative_flags,
        next_stage_requires_root_review=complete,
        reason=("Review bounded ordinary remaining target and engineering costs before an explicitly authorized P4 dependency policy"
                if complete and not negative_flags and results.get("repeatable_wait") is True else
                "No positive P4 gate: finish unmet evidence or resolve/document negative engineering findings; P4 remains disabled"),
        production_state_interference_table_available=False)
    output["limitations"] = [
        "This stdlib reader verifies typed recorded evidence/hash bindings, not native physical behavior by executing it.",
        "Model producer performed full exact reference audit; this reader cross-checks raw outputs/drain/timelines and accepted counters.",
        "Capacity producer qualified original numeric/medium/cost paths; no loaded or full-domain cost extrapolation.",
        "Condition table supported values are conditional loaded ITL, not an effect delta or production state lookup.",
        "Span/handler ordinal producer was frozen; quantity is not independently regated here.",
        "Host query gate can return False after real Event is already True: software protection qualification, not true DMA delay/performance.",
        "Run medians are descriptive with two runs per common arm; correlated token/windows do not provide independent replication.",
        "Common instrumentation remains in both optional-off/on arms; wrapped thread CPU is not all model CPU.",
        "A quota that did not bind is recorded without manufacturing contention or changing a frozen candidate."
    ]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = json.loads(args.manifest.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(AuditError("nonfinite manifest")))
        result = analyze(args.root, value)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        result = dict(schema_version=1, status="P3_INCOMPLETE", p3_phase_closed=False,
                      unmet_gates=[dict(name="closeout_manifest_input", reason=type(exc).__name__ + ": " + str(exc))],
                      p4_enabled=False, formal_efficacy_claim=False)
    result["closeout_manifest"] = str(args.manifest.resolve())
    if args.manifest.is_file():
        result["closeout_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], unmet_gates=len(result["unmet_gates"]),
                          output=str(args.output)), ensure_ascii=False))
    return 0 if result["status"] == "P3_EVIDENCE_COMPLETE_FOR_ROOT_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
