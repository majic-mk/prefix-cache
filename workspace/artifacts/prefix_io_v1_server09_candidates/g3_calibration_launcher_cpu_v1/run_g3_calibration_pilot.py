"""Finite original-acquirer pilot; GPU entry requires a new human scope and guard.

Importing this file or performing source preflight is pure CPU. The original
model, cache, asynchronous I/O, and cumulative budget runner remain authoritative.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DELIVERY = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
SCRIPT = DELIVERY + "/run_g3_calibration_pilot.py"
PLAN_MODULE = DELIVERY + "/g3_calibration_plan.py"
RUNTIME_MODULE = DELIVERY + "/g3_calibration_runtime.py"
RESULT_MODULE = DELIVERY + "/g3_calibration_result.py"
PLAN = DELIVERY + "/G3_CALIBRATION_CPU_PLAN.json"
LOCK = DELIVERY + "/gpu-source-lock-candidate.json"
RUNS = "experiments/prefix_io_v1/runs"
LEDGER = "experiments/prefix_io_v1/gpu-budget-ledger.json"
GUARD = "experiments/prefix_io_v1/scripts/run_gpu_stage.py"
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server09.g1.yaml"
SHARED = RUNS + "/server09-g3-calibration-shared-01/storage"
BOOTSTRAP_PLAN_REF = dict(path=PLAN_MODULE, bytes=24475,
    sha256="196700d96733ec07c5a4c08c6104ab31a2a36548cf005fbb8e11b6a769f6f0c0")
BOOTSTRAP_RUNTIME_REF = dict(path=RUNTIME_MODULE, bytes=27641,
    sha256="a32842723fb9871696173c257d3276d156545d519a7aefeea449b367bd79a294")
BOOTSTRAP_RESULT_REF = dict(path=RESULT_MODULE, bytes=34174,
    sha256="83ae8bb14181d51e9d4b0e40bca4123d30578cce43f581d851f97033f155eb9e")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and
            not Path(relative).is_absolute() and
            all(part not in ("", ".", "..") for part in relative.split("/")),
            "project-relative POSIX path required")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "symlink in project evidence path")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), "path outside project")
    return path


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), "missing regular evidence file: " + relative)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=relative, bytes=path.stat().st_size, sha256=h.hexdigest())


def read_json(path):
    require(path.is_file() and path.stat().st_size <= 16 * 1024 * 1024,
            "bounded regular JSON evidence required")
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def load_module(root, relative, expected=None):
    path = safe(root, relative)
    require(path.is_file() and path.stat().st_size <= 4 * 1024 * 1024, "bounded helper source")
    raw = path.read_bytes()
    actual = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if expected is not None:
        require(actual == expected, "loaded helper source differs from lock")
    name = "_g3_pilot_" + actual["sha256"]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, safe(root, relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def new_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def child_command(root, mode, source_lock, scope_record):
    return [".venv/bin/python", "-B", SCRIPT, "--project", str(root),
            "--mode", mode, "--source-lock", source_lock,
            "--scope-record", scope_record, "--execute"]


def storage_snapshot(root):
    path = safe(root, RUNS)
    require(path.is_dir(), "original PRIMARY experiment directory required")
    st = os.statvfs(path)
    return dict(origin="actual_statvfs", captured_unix=time.time(),
                primary_root=str(root), free_bytes=st.f_bavail * st.f_frsize)


def full_sources(root, source_lock):
    plan = load_module(root, PLAN_MODULE, BOOTSTRAP_PLAN_REF)
    lock_ref = ref(root, source_lock)
    refs = plan.verify_source_lock(root, lock_ref)
    required = {SCRIPT, PLAN_MODULE, RUNTIME_MODULE, RESULT_MODULE, PLAN, GUARD, PERMISSIONS}
    require(required <= set(refs), "full pilot source/plan/original guard pins missing")
    require(refs[GUARD] == dict(path=GUARD, bytes=13013,
            sha256="3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a"),
            "original current budget guard must remain unchanged")
    require(read_json(safe(root, PLAN))["qualification_context"] == plan.fixed_context(),
            "frozen pilot CPU plan context differs")
    require(refs[RUNTIME_MODULE] == BOOTSTRAP_RUNTIME_REF, "exact audited runtime bootstrap ref")
    require(refs[RESULT_MODULE] == BOOTSTRAP_RESULT_REF, "exact audited result validator ref")
    runtime = load_module(root, RUNTIME_MODULE, BOOTSTRAP_RUNTIME_REF)
    runtime.preflight_runtime(root, refs)
    return plan, refs, lock_ref


def prior_results(root, plan, refs, scope, scope_sha, ledger, source_lock, *, verify_publication=True):
    checker = load_module(root, RESULT_MODULE, refs[RESULT_MODULE])
    receipts = []
    identities = []
    for mode in plan.MODES:
        label = plan.JOB_NAMES[mode]
        events = [e for e in ledger["events"] if type(e) is dict and e.get("label") == label]
        require(len(events) <= 1, "pilot label repeated; new authorization required")
        if not events:
            require(not safe(root, RUNS + "/" + label).exists() or
                    (type(ledger.get("active_reservation")) is dict and
                     ledger["active_reservation"].get("label") == label),
                    "unaccounted existing pilot path blocks replay")
            continue
        event = events[0]
        details = safe(root, RUNS + "/" + label + "/details")
        report = read_json(details / "acquisition/result.json")
        runtime = read_json(details / "calibration-runtime-result.json")
        config = read_json(details / "acquisition/frozen-config.json")
        require(read_json(safe(root, RUNS + "/" + label + "/result.json")) == event,
                "durable original guard result differs from ledger")
        source_closure = read_json(details / "post-source-verification.json")
        require(source_closure.get("status") == "PASS_ACTUAL_PARENT_FULL_SOURCE_BYTES" and
                source_closure.get("source_lock") == scope["source_lock"] and
                type(source_closure.get("source_count")) is int and
                source_closure["source_count"] == len(refs) and
                source_closure.get("scope_sha256") == scope_sha and
                source_closure.get("label") == label,
                "actual parent complete source closure required")
        binding = dict(expected_source_lock_sha256=scope["source_lock"]["sha256"],
            expected_scope_sha256=scope_sha, expected_gpu_uuid=scope["gpu_uuid"],
            expected_label=label, expected_command=child_command(root, mode, source_lock,
                scope["_actual_scope_path"]), expected_storage=str(safe(root, SHARED)),
                prior_process_identities=list(identities))
        verdict = checker.validate_job_result(mode, report, runtime, event, binding, config=config)
        require(verdict.get("status") == "PASS_BOUNDED_ORIGINAL_ACQUISITION_PATH_CHECKS",
                "previous original path pilot did not qualify: " + mode)
        if mode == "populate" and verify_publication:
            publication = read_json(details / "storage-publication.json")
            checker.verify_storage_publication(safe(root, SHARED), publication)
        identities.append(runtime["process_identity"])
        receipts.append(dict(mode=mode, label=label, guard_exit=event["exit"],
            child_exit=event["child_exit"], timed_out=event["timed_out"], error=event["error"],
            session_drained=event["session_drained"],
            session_members_before_cleanup=event["session_members_before_cleanup"],
            session_members_after_cleanup=event["session_members_after_cleanup"],
            original_exit_code=runtime["original_exit_code"],
            original_acquisition_status=report["status"],
            original_shutdown_completed=report.get("engine_shutdown") == "completed"))
    return receipts


def publish_after_original_guard(root, plan, refs, scope, scope_ref, source_lock, mode, ledger):
    """Only the parent can bind original shutdown to the completed OS session."""
    if mode != "populate":
        return
    checker = load_module(root, RESULT_MODULE, refs[RESULT_MODULE])
    label = plan.JOB_NAMES[mode]
    events = [e for e in ledger["events"] if type(e) is dict and e.get("label") == label]
    require(len(events) == 1, "one original guarded populate event required")
    details = safe(root, RUNS + "/" + label + "/details")
    report = read_json(details / "acquisition/result.json")
    runtime = read_json(details / "calibration-runtime-result.json")
    config = read_json(details / "acquisition/frozen-config.json")
    binding = dict(expected_source_lock_sha256=scope["source_lock"]["sha256"],
        expected_scope_sha256=scope_ref["sha256"], expected_gpu_uuid=scope["gpu_uuid"],
        expected_label=label, expected_command=child_command(root, mode, source_lock,
            scope["_actual_scope_path"]), expected_storage=str(safe(root, SHARED)))
    verdict = checker.validate_job_result(mode, report, runtime, events[0], binding, config=config)
    require(verdict.get("status") == "PASS_BOUNDED_ORIGINAL_ACQUISITION_PATH_CHECKS",
            "failed populate cannot publish SSD target")
    publication = checker.build_storage_publication(safe(root, SHARED))
    new_json(details / "storage-publication.json", publication)


def gates(root, args, *, executing=False):
    # Reject missing approval before even the full model/source CPU reads.
    require(args.scope_record is not None, "NO_NEW_G3_HUMAN_SCOPE")
    scope_ref = ref(root, args.scope_record)
    scope = read_json(safe(root, args.scope_record))
    plan, refs, lock_ref = full_sources(root, args.source_lock)
    require(scope.get("source_lock") == lock_ref and scope.get("cpu_plan_ref") == refs[PLAN],
            "new human scope must bind actual pilot source lock and CPU plan")
    plan.validate_scope(root, scope, refs)
    ledger = read_json(safe(root, LEDGER))
    scope["_actual_scope_path"] = args.scope_record
    receipts = prior_results(root, plan, refs, scope, scope_ref["sha256"], ledger, args.source_lock)
    require(plan.next_job(receipts) == args.mode, "strict cold/populate/paired order; no replay")
    command = child_command(root, args.mode, args.source_lock, args.scope_record)
    if executing:
        # Scope validation above precedes adding only an internal diagnostic path.
        clean_scope = {k: v for k, v in scope.items() if k != "_actual_scope_path"}
        witness = plan.validate_execution_guard(clean_scope, ledger, args.mode,
            actual_session_id=os.getsid(0), expected_command=command,
            scope_sha256=scope_ref["sha256"])
    else:
        require(ledger.get("active_reservation") is None, "unresolved original guard reservation")
        witness = None
    snapshot = storage_snapshot(root)
    budget_ledger = dict(ledger)
    if executing:
        # This reservation was just independently matched to our real OS session.
        budget_ledger["active_reservation"] = None
    plan.validate_budget_and_storage(budget_ledger, snapshot,
                                    remaining_jobs=len(plan.MODES) - len(receipts))
    out = safe(root, RUNS + "/" + plan.JOB_NAMES[args.mode] + "/details")
    require(not out.exists(), "new result details only; prior jobs cannot be retried")
    storage = safe(root, SHARED)
    if args.mode != "paired":
        require(not storage.exists(), "cold/populate must not consume old SSD data")
    else:
        require(storage.is_dir(), "paired requires newly published original SSD cache")
    return plan, refs, scope, scope_ref, ledger, out, storage, witness


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--mode", choices=("cold", "populate", "paired"), default="cold")
    p.add_argument("--source-lock", default=LOCK)
    p.add_argument("--scope-record")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--scope-template", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    root = args.project.resolve()
    guard_launch_attempted = False
    try:
        if args.scope_template:
            plan = load_module(root, PLAN_MODULE, BOOTSTRAP_PLAN_REF)
            print(json.dumps(plan.scope_template(ref(root, args.source_lock),
                plan_ref=ref(root, PLAN)), indent=2)); return 0
        if args.preflight and args.scope_record is None:
            plan, refs, lock_ref = full_sources(root, args.source_lock)
            ledger = read_json(safe(root, LEDGER))
            budget = plan.validate_budget_and_storage(ledger, storage_snapshot(root))
            print(json.dumps(dict(status="CPU_READY_NO_NEW_G3_HUMAN_SCOPE",
                source_count=len(refs), source_lock=lock_ref, budget_and_storage=budget,
                actual_gpu_runs=0, framework_imported=False, GPU_initialized=False,
                production_qualified=False, performance_claim=False))); return 0
        plan, refs, scope, scope_ref, ledger, out, storage, witness = gates(
            root, args, executing=args.execute)
        if args.preflight or not (args.launch or args.execute):
            print(json.dumps(dict(status="AUTHORIZED_CPU_PREFLIGHT_NO_LAUNCH", actual_gpu_runs=0)))
            return 0
        if args.launch:
            command = [".venv/bin/python", "-B", GUARD, "--permissions-path", PERMISSIONS,
                "--label", plan.JOB_NAMES[args.mode], "--seconds", "300", "--",
                *child_command(root, args.mode, args.source_lock, args.scope_record)]
            guard_launch_attempted = True
            code = subprocess.run(command, cwd=root, env=dict(os.environ,
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                PYTHONHASHSEED="0")).returncode
            if code != 0:
                return code
            # Read original guard closure after the child is gone; no GPU work.
            plan, refs, post_lock = full_sources(root, args.source_lock)
            require(post_lock == scope["source_lock"], "post-run complete frozen source lock changed")
            details = safe(root, RUNS + "/" + plan.JOB_NAMES[args.mode] + "/details")
            new_json(details / "post-source-verification.json", dict(
                status="PASS_ACTUAL_PARENT_FULL_SOURCE_BYTES", source_lock=post_lock,
                source_count=len(refs), scope_sha256=scope_ref["sha256"],
                label=plan.JOB_NAMES[args.mode], GPU_operations_in_source_check=0))
            done = read_json(safe(root, LEDGER))
            publish_after_original_guard(root, plan, refs, scope, scope_ref, args.source_lock, args.mode, done)
            receipts = prior_results(root, plan, refs, scope, scope_ref["sha256"], done,
                args.source_lock, verify_publication=args.mode != "paired")
            plan.next_job(receipts)
            return 0
        require(witness is not None, "actual original guard witness required")
        out.mkdir(exist_ok=False)
        if args.mode == "populate":
            storage.parent.mkdir(exist_ok=False)
        runtime = load_module(root, RUNTIME_MODULE, refs[RUNTIME_MODULE])
        outcome = runtime.execute_original_acquisition(root, args.mode, out, storage, refs, witness)
        require(type(outcome) is dict, "bounded original runtime outcome required")
        return outcome["original_exit_code"]
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps(dict(status="BLOCKED_G3_CPU_OR_AUTHORITY_GATE", reason=str(exc),
            actual_gpu_runs=0 if not (args.execute or guard_launch_attempted) else "UNKNOWN_READ_ORIGINAL_GUARD_LEDGER",
            GPU_initialized=False if not (args.execute or guard_launch_attempted) else "UNKNOWN_READ_RUNTIME_RECEIPT",
            production_qualified=False, performance_claim=False)))
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
