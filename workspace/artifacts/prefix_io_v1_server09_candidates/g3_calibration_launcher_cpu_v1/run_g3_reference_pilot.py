"""Two original diagnostic acquisitions; CPU preparation never grants GPU scope."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DELIVERY = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
SCRIPT = DELIVERY + "/run_g3_reference_pilot.py"
PLAN_MODULE = DELIVERY + "/g3_reference_plan.py"
RUNTIME_MODULE = DELIVERY + "/g3_reference_runtime.py"
RESULT_MODULE = DELIVERY + "/g3_reference_result.py"
PLAN = DELIVERY + "/G3_REFERENCE_CPU_PLAN.json"
INPUT = DELIVERY + "/G3_REFERENCE_INPUT_MANIFEST.json"
ANALYZER_PLAN = DELIVERY + "/G3_REFERENCE_ANALYZER_PLAN.json"
MODEL_CONTEXT = DELIVERY + "/G3_REFERENCE_MODEL_CONTEXT.json"
LOCK = DELIVERY + "/gpu-source-lock-reference-v1.json"
RUNS = "experiments/prefix_io_v1/runs"
LEDGER = "experiments/prefix_io_v1/gpu-budget-ledger.json"
GUARD = "experiments/prefix_io_v1/scripts/run_gpu_stage.py"
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server09.g1.yaml"
PARENT_PATH = Path(__file__).with_name("run_g3_calibration_pilot_v3.py")
raw = PARENT_PATH.read_bytes()
if PARENT_PATH.is_symlink() or len(raw) != 18729 or hashlib.sha256(raw).hexdigest() != "095ca8cf7ab2393c1b42b6ecb59e00bff409052fe7a49c7979e511fc11f2eadc":
    raise ValueError("exact frozen parent CPU helper required")
namespace = dict(__name__="_reference_parent_cpu_only", __file__=str(PARENT_PATH))
exec(compile(raw, str(PARENT_PATH), "exec", dont_inherit=True), namespace)
require, safe, ref = (namespace[n] for n in ("require", "safe", "ref"))
read_json, load_module, new_json = (namespace[n] for n in ("read_json", "load_module", "new_json"))
storage_snapshot = namespace["storage_snapshot"]
check_common_stats_dictionary_fix = namespace["check_common_stats_dictionary_fix"]
del raw, namespace
BOOTSTRAP_PLAN_REF = dict(path=PLAN_MODULE, bytes=19355, sha256="600d1cf449c964339c4515de3e2f4a6fe9dbf21c715acf164f102c90eedce017")
BOOTSTRAP_RUNTIME_REF = dict(path=RUNTIME_MODULE, bytes=14377, sha256="4adc621057fde015e4483d2140601c1d8f68936c1e2b4dd2f828f3c04ef50ab3")
BOOTSTRAP_RESULT_REF = dict(path=RESULT_MODULE, bytes=18442, sha256="46efa7026f66da7b44bb0d10df96b5a4f6aaef3e0f7b0073654144ca604415a8")
ANALYZERS = dict(cached="experiments/prefix_io_v1/scripts/analyze_cached_references.py",
    repeated="experiments/prefix_io_v1/scripts/analyze_repeated_native_costs.py",
    storage="experiments/prefix_io_v1/scripts/experiment_storage.py")


def child_command(root, mode, lock, scope):
    return [".venv/bin/python", "-B", SCRIPT, "--project", str(root), "--mode", mode,
            "--source-lock", lock, "--scope-record", scope, "--execute"]


def input_check(root, plan, refs):
    require(ref(root, INPUT) == refs[INPUT], "frozen existing reference input manifest")
    record = read_json(safe(root, INPUT))
    require(record.get("status") == "CPU_FROZEN_EXISTING_G3_REFERENCE_CACHE_INPUT_NOT_AUTHORIZED"
        and record.get("source_lock_ref") == plan.BASELINE_REF
        and record.get("storage_relative") == plan.PAIRED_STORAGE_RELATIVE
        and record.get("old_publication_rows_equal_current") is True
        and record.get("GPU_authorized") is False, "original existing cache input ancestry")
    original = load_module(root, DELIVERY + "/g3_calibration_result.py", refs[DELIVERY + "/g3_calibration_result.py"])
    publication = record["publication"]
    require(publication["file_count"] == 24 and publication["total_bytes"] == 22020096,
            "fixed published reference cache input size")
    original.verify_storage_publication(safe(root, plan.PAIRED_STORAGE_RELATIVE), publication)
    return dict(status="PASS_ACTUAL_EXISTING_CACHE_BYTES", file_count=24, total_bytes=22020096,
                GPU_operations=0, logical_KV_identity_verified=False)


def model_context(root, refs):
    require(ref(root, MODEL_CONTEXT) == refs[MODEL_CONTEXT], "frozen independent model context")
    context = read_json(safe(root, MODEL_CONTEXT))
    require(context.get("status") == "CPU_FROZEN_CURRENT_REFERENCE_MODEL_AND_ANALYZER_DEPENDENCIES"
            and context.get("authorizes_gpu") is False, "model context is CPU evidence only")
    previous = context["original_config_ref"]
    require(previous["path"] == RUNS + "/server09-g3-calibration-cold-02/details/acquisition/frozen-config.json"
            and ref(root, previous["path"]) == previous, "actual prior native config identity")
    actual = read_json(safe(root, previous["path"]))
    for name in ("model", "alias_target", "model_alias"):
        require(context[name] == actual[name], "prior current model binding: " + name)
    require(context["engine_without_connector"] == actual["engine"]
            and context["original_config_gpu_uuid"] == actual["gpu_uuid"], "original model engine/GPU context")
    for name in ("cpu_config_ref", "cpu_init_ref"):
        row = context[name]
        require(refs.get(row["path"]) == row and ref(root, row["path"]) == row, "original CPU config dependency")
    return context


def yaml_sources(root, context):
    base = root / ".venv/lib/python3.12/site-packages/yaml"
    require(context["yaml_root"] == str(base) and base.is_dir() and not base.is_symlink(),
            "fixed installed CPU YAML dependency")
    rows = context["yaml_files"]
    require(type(rows) is list and 1 <= len(rows) <= 32, "bounded YAML dependency manifest")
    actual = sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    require([str(p) for p in actual] == [r["path"] for r in rows], "YAML dependency inventory drift")
    for row, p in zip(rows, actual):
        require(not p.is_symlink() and p.resolve().is_relative_to(base.resolve()), "YAML source path")
        require(type(row["bytes"]) is int and 0 < row["bytes"] <= 4 * 1024**2, "bounded YAML dependency")
        data = p.read_bytes()
        require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "YAML dependency bytes drift")
    require(base / "__init__.py" in actual, "original YAML initializer")
    return base


@contextmanager
def original_analyzer_context(root, refs):
    """Load the three unmodified original CPU modules and restore all bindings."""
    context = model_context(root, refs)
    base = yaml_sources(root, context)
    prefixes = ("yaml", "prefix_io_control", "experiment_storage", "analyze_repeated_native_costs", "analyze_cached_references")
    names = lambda n: any(n == p or n.startswith(p + ".") for p in prefixes)
    saved = {n: m for n, m in list(sys.modules.items()) if names(n)}
    before = set(sys.modules)
    for n in saved:
        sys.modules.pop(n)
    storage = None
    permission = None
    try:
        spec = importlib.util.spec_from_file_location("yaml", base / "__init__.py", submodule_search_locations=[str(base)])
        yaml = importlib.util.module_from_spec(spec)
        sys.modules["yaml"] = yaml
        spec.loader.exec_module(yaml)
        def named(name, row, package=False):
            path = safe(root, row["path"])
            require(ref(root, row["path"]) == row, "original CPU module source bytes")
            spec = importlib.util.spec_from_file_location(name, path,
                submodule_search_locations=[str(path.parent)] if package else None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
            return module
        named("prefix_io_control", context["cpu_init_ref"], True)
        config = named("prefix_io_control.config", context["cpu_config_ref"])
        storage = named("experiment_storage", refs[ANALYZERS["storage"]])
        permission = storage.permission
        effective = config.validate_permissions(config.read_yaml(safe(root, PERMISSIONS)))
        require(effective["approved_experiment_root"] == str(root / RUNS)
                and effective["approved_gpu_ids"] == [context["original_config_gpu_uuid"]]
                and effective.get("approved_auxiliary_storage") is None, "unchanged PRIMARY effective permission")
        def current_permission(project=storage.ROOT):
            require(Path(project).resolve() == root, "original authorization project identity")
            return dict(effective)
        storage.permission = current_permission
        named("analyze_repeated_native_costs", refs[ANALYZERS["repeated"]])
        analyzer = named("analyze_cached_references", refs[ANALYZERS["cached"]])
        require(not any(n in sys.modules and n not in before for n in ("torch", "vllm", "py_kvcache", "pytorch")),
                "CPU analyzer must not import model/backend")
        yield analyzer
    finally:
        if storage is not None and permission is not None:
            storage.permission = permission
        for n in list(sys.modules):
            if names(n):
                sys.modules.pop(n)
        sys.modules.update(saved)
        yaml_sources(root, context)


def full_sources(root, source_lock):
    plan = load_module(root, PLAN_MODULE, BOOTSTRAP_PLAN_REF)
    lock_ref = ref(root, source_lock)
    refs = plan.verify_source_lock(root, lock_ref)
    required = {SCRIPT, PLAN_MODULE, RUNTIME_MODULE, RESULT_MODULE, PLAN, INPUT, ANALYZER_PLAN, MODEL_CONTEXT,
                GUARD, PERMISSIONS, *ANALYZERS.values()}
    require(required <= refs.keys(), "complete reference source/plan/input/analyzer/model/guard pins")
    for row in (BOOTSTRAP_PLAN_REF, BOOTSTRAP_RUNTIME_REF, BOOTSTRAP_RESULT_REF, plan.GUARD_REF):
        require(refs[row["path"]] == row, "exact audited bootstrap/guard source")
    frozen = read_json(safe(root, PLAN))
    expected = plan.build_reference_plan(frozen["new_source_refs"], refs[INPUT], analyzer_plan_ref=refs[ANALYZER_PLAN])
    plan.same(frozen, expected, "exact typed frozen CPU reference plan")
    for row in frozen["new_source_refs"]:
        require(refs.get(row["path"]) == row, "CPU plan own source/input dependency pins")
    context = model_context(root, refs)
    require(context["alias_target"] == str(root / plan.MODEL_RELATIVE)
            and context["original_config_gpu_uuid"] == plan.GPU_UUID, "fixed current model/GPU context")
    checker = load_module(root, RESULT_MODULE, BOOTSTRAP_RESULT_REF)
    analyzer_plan = read_json(safe(root, ANALYZER_PLAN))
    checker.validate_reference_plan(analyzer_plan, expected_gpu_uuid=plan.GPU_UUID,
        expected_native_storage=str(safe(root, plan.STORAGE_RELATIVE)),
        expected_paired_storage=str(safe(root, plan.PAIRED_STORAGE_RELATIVE)))
    require(analyzer_plan["run_details"] == {label: str(safe(root, RUNS + "/" + label + "/details/acquisition"))
        for label in plan.JOB_NAMES.values()}, "actual two reference details locations")
    input_check(root, plan, refs)
    with original_analyzer_context(root, refs):
        pass
    load_module(root, RUNTIME_MODULE, BOOTSTRAP_RUNTIME_REF).preflight_runtime(root, refs)
    return plan, refs, lock_ref


def reference_receipt(root, details, runtime, refs):
    require(runtime.get("reference_delegate_overrides_restored") is True
            and runtime.get("numerical_reference_only") is True
            and runtime.get("diagnostic_latency_excluded_from_fit") is True, "reference delegate restoration")
    original_ref = runtime["original_delegate_receipt_ref"]
    original_rel = str((details / "calibration-runtime-result.json").relative_to(root))
    require(original_ref == ref(root, original_rel), "original raw delegate receipt bytes")
    original = read_json(details / "calibration-runtime-result.json")
    allowed = {"raw_calibration_only"}
    for key, value in original.items():
        if key not in allowed:
            require(runtime.get(key) == value, "reference original closure/core field lost: " + key)
    report_ref = ref(root, str((details / "acquisition/result.json").relative_to(root)))
    require(runtime["original_report_ref"] == report_ref
            and runtime["delegate_runtime_ref"] == refs[DELIVERY + "/g3_calibration_runtime_metrics_v2.py"],
            "original diagnostic report/delegate bytes binding")
    check_common_stats_dictionary_fix(root, runtime)


def prior_results(root, plan, refs, scope, scope_sha, ledger, source_lock):
    checker = load_module(root, RESULT_MODULE, refs[RESULT_MODULE])
    context = model_context(root, refs)
    receipts, checks, identities = [], {}, []
    for mode in plan.MODES:
        label = plan.JOB_NAMES[mode]
        events = [e for e in ledger["events"] if type(e) is dict and e.get("label") == label]
        require(len(events) <= 1, "reference label repeated; this scope ends")
        if not events:
            require(not safe(root, RUNS + "/" + label).exists() or (type(ledger.get("active_reservation")) is dict
                and ledger["active_reservation"].get("label") == label), "unaccounted reference output path blocks replay")
            continue
        require(len(receipts) == plan.MODES.index(mode), "reference job order")
        event = events[0]
        details = safe(root, RUNS + "/" + label + "/details")
        require(read_json(safe(root, RUNS + "/" + label + "/result.json")) == event, "original guard durable ledger identity")
        closure = read_json(details / "post-source-verification.json")
        require(closure.get("status") == "PASS_ACTUAL_PARENT_FULL_SOURCE_BYTES"
                and closure.get("source_lock") == scope["source_lock"] and closure.get("source_count") == len(refs)
                and closure.get("scope_sha256") == scope_sha and closure.get("label") == label, "actual parent source closure")
        report = read_json(details / "acquisition/result.json")
        config = read_json(details / "acquisition/frozen-config.json")
        runtime = read_json(details / "reference-runtime-result.json")
        reference_receipt(root, details, runtime, refs)
        binding = dict(expected_source_lock_sha256=scope["source_lock"]["sha256"], expected_scope_sha256=scope_sha,
            expected_gpu_uuid=scope["gpu_uuid"], expected_label=label,
            expected_command=child_command(root, mode, source_lock, scope["_actual_scope_path"]),
            expected_storage=str(safe(root, plan.storage_relative(mode))), expected_model=context["model"],
            expected_alias_target=context["alias_target"], prior_process_identities=list(identities))
        check = checker.validate_reference_job(mode, report, runtime, event, binding, config)
        require(check["status"] == "PASS_REFERENCE_SIX_REQUEST_PATH_CLOSURE_ONLY", "six original reference request closure")
        checks[mode] = check
        identities.append(runtime["process_identity"])
        receipts.append(dict(mode=mode, label=label, guard_exit=event["exit"], child_exit=event["child_exit"],
            timed_out=event["timed_out"], error=event["error"], session_drained=event["session_drained"],
            session_members_before_cleanup=event["session_members_before_cleanup"],
            session_members_after_cleanup=event["session_members_after_cleanup"], original_exit_code=runtime["original_exit_code"],
            original_acquisition_status=report["status"], original_shutdown_completed=report.get("engine_shutdown") == "completed"))
    plan.next_job(receipts)
    return receipts, checks


def analyze_pair(root, refs, checks):
    require(set(checks) == {"cold", "paired"}, "both real six-request reference closures")
    checker = load_module(root, RESULT_MODULE, refs[RESULT_MODULE])
    with original_analyzer_context(root, refs) as analyzer:
        result = checker.verify_reference_pair(root, safe(root, ANALYZER_PLAN), expected_plan_ref=refs[ANALYZER_PLAN],
            native_check=checks["cold"], paired_check=checks["paired"],
            analyzer_refs={k: refs[v] for k, v in ANALYZERS.items()}, original_analyzer=analyzer)
    return result


def gates(root, args, executing=False):
    require(args.scope_record is not None, "NO_NEW_REFERENCE_HUMAN_SCOPE")
    scope_ref = ref(root, args.scope_record)
    scope = read_json(safe(root, args.scope_record))
    require(scope.get("purpose") == "CURRENT_CONTEXT_CACHED_NUMERICAL_REFERENCE_ONLY"
            and scope.get("status") == "USER_AUTHORIZED_G3_REFERENCE_DIAGNOSTIC", "OLD_SCOPE_CANNOT_AUTHORIZE_REFERENCE")
    plan, refs, lock_ref = full_sources(root, args.source_lock)
    require(scope["source_lock"] == lock_ref and scope["cpu_plan_ref"] == refs[PLAN]
            and scope["input_manifest_ref"] == refs[INPUT] and scope["analyzer_plan_ref"] == refs[ANALYZER_PLAN],
            "new exact source/plan/input/analyzer scope bindings")
    plan.validate_scope(root, scope, refs)
    scope["_actual_scope_path"] = args.scope_record
    ledger = read_json(safe(root, LEDGER))
    receipts, checks = prior_results(root, plan, refs, scope, scope_ref["sha256"], ledger, args.source_lock)
    require(plan.next_job(receipts) == args.mode, "strict native then paired order; no retry")
    if executing:
        clean = {k: v for k, v in scope.items() if k != "_actual_scope_path"}
        witness = plan.validate_execution_guard(clean, ledger, args.mode, actual_session_id=os.getsid(0),
            expected_command=child_command(root, args.mode, args.source_lock, args.scope_record), scope_sha256=scope_ref["sha256"])
    else:
        require(ledger.get("active_reservation") is None, "original GPU reservation unresolved")
        witness = None
    budget = dict(ledger)
    if executing:
        budget["active_reservation"] = None
    plan.validate_budget_and_storage(budget, storage_snapshot(root), remaining_jobs=2 - len(receipts))
    out = safe(root, RUNS + "/" + plan.JOB_NAMES[args.mode] + "/details")
    require(not out.exists(), "new reference details only; no overwrite/retry")
    storage = safe(root, plan.storage_relative(args.mode))
    require(not storage.exists() if args.mode == "cold" else storage.is_dir(), "native unused/paired existing storage boundary")
    return plan, refs, scope, scope_ref, ledger, out, storage, witness


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--mode", choices=("cold", "paired"), default="cold")
    p.add_argument("--source-lock", default=LOCK)
    p.add_argument("--scope-record")
    group = p.add_mutually_exclusive_group()
    for option in ("preflight", "scope-template", "launch", "execute"):
        group.add_argument("--" + option, action="store_true")
    args = p.parse_args(argv)
    root = args.project.resolve()
    attempted = False
    executed = False
    try:
        if args.scope_template:
            plan = load_module(root, PLAN_MODULE, BOOTSTRAP_PLAN_REF)
            print(json.dumps(plan.scope_template(ref(root, args.source_lock), ref(root, PLAN), ref(root, INPUT), ref(root, ANALYZER_PLAN)), indent=2))
            return 0
        if args.preflight and args.scope_record is None:
            plan, refs, lock_ref = full_sources(root, args.source_lock)
            budget = plan.validate_budget_and_storage(read_json(safe(root, LEDGER)), storage_snapshot(root))
            for mode in plan.MODES:
                require(not safe(root, RUNS + "/" + plan.JOB_NAMES[mode]).exists(), "unused reference names required")
            require(not safe(root, plan.STORAGE_RELATIVE).exists(), "native unused storage required")
            print(json.dumps(dict(status="CPU_READY_NO_NEW_REFERENCE_HUMAN_SCOPE", source_count=len(refs), source_lock=lock_ref,
                budget_and_storage=budget, actual_gpu_runs=0, framework_imported=False, GPU_initialized=False,
                numerical_reference_verified=False, production_qualified=False, performance_claim=False)))
            return 0
        plan, refs, scope, scope_ref, ledger, out, storage, witness = gates(root, args, executing=args.execute)
        if not (args.launch or args.execute):
            print(json.dumps(dict(status="AUTHORIZED_CPU_PREFLIGHT_NO_LAUNCH", actual_gpu_runs=0)))
            return 0
        if args.launch:
            command = [".venv/bin/python", "-B", GUARD, "--permissions-path", PERMISSIONS,
                "--label", plan.JOB_NAMES[args.mode], "--seconds", "300", "--",
                *child_command(root, args.mode, args.source_lock, args.scope_record)]
            attempted = True
            code = subprocess.run(command, cwd=root, env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")).returncode
            if code:
                return code
            plan, refs, post_lock = full_sources(root, args.source_lock)
            require(post_lock == scope["source_lock"], "post-run complete source lock unchanged")
            new_json(out / "post-source-verification.json", dict(status="PASS_ACTUAL_PARENT_FULL_SOURCE_BYTES", source_lock=post_lock,
                source_count=len(refs), scope_sha256=scope_ref["sha256"], label=plan.JOB_NAMES[args.mode], GPU_operations_in_source_check=0))
            done = read_json(safe(root, LEDGER))
            require(done.get("active_reservation") is None, "guard reservation must close")
            receipts, checks = prior_results(root, plan, refs, scope, scope_ref["sha256"], done, args.source_lock)
            if plan.next_job(receipts) is None:
                result = analyze_pair(root, refs, checks)
                new_json(out / "reference-analysis-result.json", result)
                return 0 if result["status"] == "PASSED_EXACT_CACHED_REFERENCE" else 1
            return 0
        require(witness is not None, "actual original guard witness required")
        out.mkdir(exist_ok=False)
        runtime = load_module(root, RUNTIME_MODULE, refs[RUNTIME_MODULE])
        executed = True
        outcome = runtime.execute_original_acquisition(root, args.mode, out, storage, refs, witness)
        require(type(outcome) is dict and type(outcome["original_exit_code"]) is int, "original acquisition outcome")
        return outcome["original_exit_code"]
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps(dict(status="BLOCKED_REFERENCE_CPU_OR_AUTHORITY_GATE", reason=str(exc),
            actual_gpu_runs=0 if not (executed or attempted) else "UNKNOWN_READ_ORIGINAL_GUARD_LEDGER",
            GPU_initialized=False if not (executed or attempted) else "UNKNOWN_READ_ORIGINAL_RUNTIME",
            production_qualified=False, performance_claim=False)))
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
