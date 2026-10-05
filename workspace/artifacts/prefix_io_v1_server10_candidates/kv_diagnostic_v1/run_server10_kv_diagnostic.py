"""Two bounded original populate/paired jobs with real KV byte observation.

This file binds the existing acquisition, publication, result and GPU guard
functions to a fresh diagnostic directory. LoadPlanner stays off. Copies made
by the separate byte observer exclude all collected latency from performance.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
OLD = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
DELIVERY = "artifacts/prefix_io_v1/server10-kv-diagnostic-v1-20261003"
SCRIPT = DELIVERY + "/run_server10_kv_diagnostic.py"
CONFIG = DELIVERY + "/KV_DIAGNOSTIC_CONFIG.json"
PLAN = DELIVERY + "/KV_DIAGNOSTIC_CPU_PLAN.json"
LOCK = DELIVERY + "/gpu-source-lock-kv-diagnostic.json"
SCOPE = DELIVERY + "/KV_DIAGNOSTIC_AUTHORIZED_SCOPE.json"
GPU_UUID = "GPU-4b4d17ec-95a3-4efd-2bc4-1613333e949f"
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server10.reference.yaml"
MODES = ("populate", "paired")
JOBS = {mode: "server10-kv-byte-" + mode + "-01" for mode in MODES}
STORAGE = "experiments/prefix_io_v1/runs/server10-kv-byte-01-private-storage"
PURPOSE = "ACTUAL_PRODUCTION_KV_BYTE_DIAGNOSTIC_ONLY"
PINNED = {
    "entry": dict(path=OLD + "/run_g3_calibration_pilot_v3.py", bytes=18729, sha256="095ca8cf7ab2393c1b42b6ecb59e00bff409052fe7a49c7979e511fc11f2eadc"),
    "plan": dict(path=OLD + "/g3_calibration_plan_metrics_v2.py", bytes=24591, sha256="96abdd444c187b70c81de82a63f24b990e3fabe65ecd37307909602181648543"),
    "runtime": dict(path=OLD + "/g3_calibration_runtime_metrics_v2.py", bytes=29978, sha256="0c04287325fd84c31aaea35dc6ff7431229511a04f3ab71bf9aa96a2ce83f1cc"),
    "reference": dict(path=OLD + "/g3_reference_runtime.py", bytes=14377, sha256="4adc621057fde015e4483d2140601c1d8f68936c1e2b4dd2f828f3c04ef50ab3"),
    "result": dict(path=OLD + "/g3_calibration_result.py", bytes=34174, sha256="83ae8bb14181d51e9d4b0e40bca4123d30578cce43f581d851f97033f155eb9e"),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def safe(root, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and "\\" not in relative and all(p not in ("", ".", "..") for p in relative.split("/")), "safe diagnostic path")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "diagnostic path symlink")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), "diagnostic path outside root")
    return path


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), "regular diagnostic source")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest.hexdigest())


def checked(root, row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
            and type(row["bytes"]) is int and 0 < row["bytes"] <= 4 * 1024**2, "bounded diagnostic bootstrap ref")
    require(ref(root, row["path"]) == row, "diagnostic bootstrap source drift: " + row["path"])
    return safe(root, row["path"])


def read_json(path):
    require(path.is_file() and path.stat().st_size <= 4 * 1024**2, "bounded diagnostic JSON")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate diagnostic JSON key")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def load(root, row, suffix):
    path = checked(root, row)
    name = "_server10_kv_" + suffix + "_" + row["sha256"][:20]
    require(name not in sys.modules, "fresh private diagnostic module")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def two_job_next_function(plan, raw):
    """Only the two finite sequence-length constants change; checks stay whole."""
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "next_job")
    hits = [n for n in ast.walk(node) if isinstance(n, ast.Constant) and type(n.value) is int and n.value == 3]
    require(len(hits) == 2, "exact original two sequence bound constants")
    for hit in hits:
        hit.value = 2
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, plan.__file__, "exec", dont_inherit=True), plan.__dict__)


def configure_plan(root, plan, config):
    plan.DELIVERY, plan.GPU_UUID, plan.PURPOSE = DELIVERY, GPU_UUID, PURPOSE
    plan.MODES, plan.JOB_NAMES, plan.STORAGE_RELATIVE = MODES, dict(JOBS), STORAGE
    plan.PERMISSIONS_REF = copy.deepcopy(config["permissions_ref"])
    plan.BASELINE_REF = copy.deepcopy(config["ancestor_ref"])
    plan.BASELINE_SHA = config["ancestor_ref"]["sha256"]
    plan.TOTAL_RESERVED_SECONDS = 640
    plan.JOB_STORAGE_RESERVE_BYTES = 512 * 1024**2
    plan.TOTAL_STORAGE_RESERVE_BYTES = 2 * plan.JOB_STORAGE_RESERVE_BYTES
    old_context, old_scope = plan.fixed_context, plan.scope_template
    old_build = plan.build_calibration_plan
    def fixed_context():
        result = old_context()
        result.update(requests_by_mode=dict(populate=9, paired=6), total_requests=15, total_outputs=15,
            actual_production_KV_byte_capture=True, latency_fit_allowed=False,
            performance_claim=False, production_qualified=False)
        return result
    def scope_template(source_lock_ref=None, *, plan_ref=None):
        result = old_scope(source_lock_ref, plan_ref=plan_ref)
        result.update(maximum_jobs=2, maximum_total_planned_reserve_seconds=640)
        return result
    def baseline(raw):
        require(len(raw) == config["ancestor_ref"]["bytes"]
            and hashlib.sha256(raw).hexdigest() == config["ancestor_ref"]["sha256"], "exact immutable diagnostic ancestor")
        value = plan.parse_json(raw)
        rows = value.get("files")
        require(type(rows) is list and 2052 <= len(rows) <= 8192, "complete ancestor closure")
        refs = {row["path"]: row for row in rows}
        require(len(refs) == len(rows), "unique ancestor refs")
        for row in (plan.GUARD_REF, plan.ACQUIRE_REF, plan.PERMISSIONS_REF):
            require(refs.get(row["path"]) == row, "original guard/acquisition and server10 permission preserved")
        return value
    def new_rows(rows, old_paths):
        require(type(rows) is list and 1 <= len(rows) <= 32, "finite diagnostic additions")
        seen = set(old_paths)
        for row in rows:
            plan.ref_shape(row)
            name = row["path"]
            require(name.startswith("artifacts/prefix_io_v1/server10-kv-") and name not in seen
                and Path(name).suffix in (".py", ".json") and row["bytes"] <= 4*1024**2
                and not any(word in Path(name).name.lower() for word in ("authorization", "human", "scope", "source-lock")),
                "append-only diagnostic sources without authorization cycles")
            seen.add(name)
        return copy.deepcopy(rows)
    def build(project_root=plan.PROJECT_ROOT, *, new_source_refs):
        result = old_build(project_root, new_source_refs=new_source_refs)
        result["jobs"][0]["requires_prior_success_original_shutdown_and_OS_drain"] = False
        return result
    plan.fixed_context, plan.scope_template = fixed_context, scope_template
    plan._baseline, plan._new_rows, plan.build_calibration_plan = baseline, new_rows, build
    two_job_next_function(plan, Path(plan.__file__).read_bytes())
    return plan


def validate_capture_cohort(capture, report, mode):
    """Bind every real pre-forward capture to its original acquisition row."""
    require(type(report) is dict and report.get("mode") == mode and type(report.get("rows")) is list,
        "actual original acquisition report for KV cohort")
    for stage, kinds, count in (("before_model_forward", ("g_mem", "g_ssd"), 3 if mode == "populate" else 6),
                               ("producer_model_forward", ("store",), 3 if mode == "populate" else 0)):
        expected = [row for row in report["rows"] if row["kind"] in kinds]
        observed = [row for row in capture["records"] if row["stage"] == stage]
        require(len(expected) == len(observed) == count, "complete same-job original/capture cohort: " + stage)
        originals = {row["request_id"]: row for row in expected}
        actual = {row["request_id"]: row for row in observed}
        require(len(originals) == len(actual) == count and set(actual) == set(originals),
            "same actual original request IDs: " + stage)
        for request_id, row in originals.items():
            prompt = row["prompt_token_ids"]
            require(type(prompt) is list and len(prompt) == 129, "complete original acquisition prompt")
            digest = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            require(actual[request_id]["prompt_sha256"] == digest and actual[request_id]["prompt_tokens"] == 129,
                "same actual original prompt bytes for captured model forward")
    return dict(status="PASS_ACTUAL_ORIGINAL_REQUEST_CAPTURE_COHORT", warmup_included=True,
                latency_fit_allowed=False, performance_claim=False)


class DiagnosticRuntime:
    def __init__(self, root, refs, config):
        self.root, self.refs, self.config = root, refs, config
        self.last_state = None

    @contextmanager
    def session(self, out=None, mode=None):
        before = dict(sys.modules)
        changes = []
        capture_handle = None
        capture_module = None
        state = dict(status="ACTIVE", GPU_UUID=GPU_UUID, model_executor_changed=False,
            original_acquisition_changed=False, latency_fit_allowed=False, performance_claim=False,
            production_qualified=False, capture_installed=False, capture_receipt=None)
        def patch(module, key, value):
            changes.append((module, key, getattr(module, key)))
            setattr(module, key, value)
        runtime = load(self.root, PINNED["runtime"], "runtime")
        reference = load(self.root, PINNED["reference"], "guard")
        try:
            for key, value in (("GPU_UUID", GPU_UUID), ("PURPOSE", PURPOSE), ("MODES", MODES), ("JOB_NAMES", dict(JOBS))):
                patch(reference, key, value)
            patch(runtime, "GPU_UUID", GPU_UUID)
            patch(runtime, "PURPOSE", PURPOSE)
            patch(runtime, "MODES", MODES)
            patch(runtime, "validate_guard", reference.validate_guard)
            original_load = runtime.load_source
            def load_source(root, refs, relative):
                nonlocal capture_module
                module = original_load(root, refs, relative)
                if relative == runtime.COMMON:
                    replacement = {module.SDK_INVENTORY: self.config["sdk_inventory_ref"], module.SDK_PROOF: self.config["sdk_proof_ref"]}
                    sdk = tuple((replacement[p]["path"], replacement[p]["bytes"], replacement[p]["sha256"])
                        if p in replacement else (p, n, sha) for p, n, sha in module.SDK_SOURCE_REFS)
                    for key, value in (("GPU_UUID", GPU_UUID), ("PERMISSIONS", PERMISSIONS),
                        ("SDK_INVENTORY", self.config["sdk_inventory_ref"]["path"]),
                        ("SDK_PROOF", self.config["sdk_proof_ref"]["path"]), ("SDK_SOURCE_REFS", sdk)):
                        patch(module, key, value)
                if relative == runtime.ACQUIRE and out is not None:
                    capture_module = load(root, self.config["capture_ref"], "capture")
                    original_control = module.worker_control
                    def control(worker, *args, **kwargs):
                        nonlocal capture_handle
                        result = original_control(worker, *args, **kwargs)
                        if capture_handle is None:
                            kv = sys.modules.get("vllm.distributed.kv_transfer.kv_transfer_state")
                            connector = None if kv is None else kv.__dict__.get("_KV_CONNECTOR_AGENT")
                            if connector is not None:
                                handlers = connector.connector_worker.worker.handlers
                                if type(handlers) is set and len(handlers) == 1:
                                    producer = None if mode == "populate" else self.root / "experiments/prefix_io_v1/runs" / JOBS["populate"] / "details/kv-capture/native-kv-byte-receipt.json"
                                    state["producer_capture_ref"] = None if producer is None else ref(root, producer.relative_to(root).as_posix())
                                    publication = None if producer is None else producer.parent.parent / "storage-publication.json"
                                    state["producer_publication_ref"] = None if publication is None else ref(root, publication.relative_to(root).as_posix())
                                    capture_handle = capture_module.install_capture(worker, root, refs, out / "kv-capture", mode,
                                        producer_manifest=producer)
                                    state["capture_installed"] = True
                        return result
                    patch(module, "worker_control", control)
                return module
            patch(runtime, "load_source", load_source)
            yield runtime
        finally:
            try:
                if capture_handle is not None:
                    capture_handle.detach()
                    state["capture_receipt"] = capture_handle.finish()
                    receipt_path = out / "kv-capture/native-kv-byte-receipt.json"
                    state["capture_receipt_ref"] = ref(self.root, receipt_path.relative_to(self.root).as_posix())
                    require(read_json(receipt_path) == state["capture_receipt"], "actual capture receipt matches installed observer result")
                    capture_module.validate_receipt(state["capture_receipt"], mode=mode)
                    state["cohort_binding"] = validate_capture_cohort(state["capture_receipt"],
                        read_json(out / "acquisition/result.json"), mode)
            finally:
                for module, key, original in reversed(changes):
                    setattr(module, key, original)
                for name in tuple(sys.modules):
                    if name not in before and name.startswith(("_server10_kv_", "_g2_lifecycle_", "_g3_calibration_")):
                        sys.modules.pop(name)
                state.update(all_bindings_restored=all(getattr(module, key) is original for module, key, original in changes),
                    preexisting_modules_preserved=all(sys.modules.get(name) is value for name, value in before.items()))
                state["status"] = "RESTORED" if state["all_bindings_restored"] and state["preexisting_modules_preserved"] else "RESTORE_FAILED"
                self.last_state = state

    def preflight_runtime(self, root, refs):
        require(root == self.root and refs == self.refs, "same diagnostic source mapping")
        with self.session() as runtime:
            result = runtime.preflight_runtime(root, refs)
        require(self.last_state["status"] == "RESTORED", "CPU private migration restore")
        return result

    def execute_original_acquisition(self, root, mode, out, storage, refs, witness):
        require(root == self.root and refs == self.refs and mode in MODES and storage == root / STORAGE,
            "fixed actual diagnostic acquisition")
        try:
            with self.session(out, mode) as runtime:
                result = runtime.execute_original_acquisition(root, mode, out, storage, refs, witness)
        finally:
            if self.last_state is not None:
                self.last_state.update(mode=mode, label=JOBS[mode], source_lock_sha256=witness["source_lock_sha256"],
                    scope_sha256=witness["scope_sha256"], process_identity=dict(pid=os.getpid(), sid=os.getsid(0)),
                    capture_module_ref=self.config["capture_ref"], storage=str(storage))
            with (out / "kv-diagnostic-runtime-state.json").open("x", encoding="utf-8") as stream:
                json.dump(self.last_state, stream, indent=2, allow_nan=False)
                stream.write("\n")
        require(self.last_state["status"] == "RESTORED" and self.last_state["capture_installed"], "actual capture installed and restored")
        return result


def build_facade(root, source_lock=LOCK):
    root = Path(root).resolve(strict=True)
    require(root == ROOT and source_lock == LOCK, "fixed diagnostic project/lock")
    lock = read_json(safe(root, source_lock))
    refs = {row["path"]: row for row in lock["files"]}
    require(len(refs) == len(lock["files"]), "unique diagnostic source refs")
    require(Path(__file__).resolve() == checked(root, refs[SCRIPT]).resolve(), "actual diagnostic entry source")
    config = read_json(checked(root, refs[CONFIG]))
    require(config["gpu_uuid"] == GPU_UUID and config["job_names"] == JOBS and config["storage"] == STORAGE,
            "frozen diagnostic machine identity")
    require(config["seconds_limit_per_job"] == 300 and config["reserved_seconds_per_job"] == 320
        and config["storage_reserve_bytes_per_job"] == 512*1024**2
        and config["total_storage_reserve_bytes"] == 1024**3 and config["storage_floor_bytes"] == 8*1024**3
        and config["load_planner"] == "off" and config["latency_fit_allowed"] is False
        and config["production_qualified"] is False and config["performance_claim"] is False,
        "fixed bounded diagnostic-only resources and strategy")
    for row in PINNED.values():
        require(refs.get(row["path"]) == row, "unchanged original diagnostic module")
    for name in ("permissions_ref", "sdk_inventory_ref", "sdk_proof_ref", "capture_ref", "ancestor_ref"):
        require(refs.get(config[name]["path"]) == config[name], "frozen diagnostic asset mapping")
        checked(root, config[name])
    prior = dict(sys.modules)
    entry = load(root, PINNED["entry"], "entry")
    plan = configure_plan(root, load(root, PINNED["plan"], "plan"), config)
    runtime = DiagnosticRuntime(root, refs, config)
    original_loader = entry.load_module
    # These are private routing identities; the original sources remain pinned.
    entry.SCRIPT, entry.LOCK, entry.PLAN, entry.PERMISSIONS = SCRIPT, LOCK, PLAN, PERMISSIONS
    entry.PLAN_MODULE, entry.RUNTIME_MODULE = SCRIPT, CONFIG
    entry.BOOTSTRAP_PLAN_REF, entry.BOOTSTRAP_RUNTIME_REF = refs[SCRIPT], refs[CONFIG]
    def loader(project, relative, expected=None):
        if relative == SCRIPT:
            require(expected == refs[SCRIPT], "private plan routing pin")
            return plan
        if relative == CONFIG:
            require(expected == refs[CONFIG], "private runtime routing pin")
            return runtime
        return original_loader(project, relative, expected)
    entry.load_module = loader
    original_prior = entry.prior_results
    def prior_results(project, the_plan, source_refs, scope, scope_sha, ledger, lock_name, *, verify_publication=True):
        receipts = original_prior(project, the_plan, source_refs, scope, scope_sha, ledger, lock_name,
                                  verify_publication=verify_publication)
        for receipt in receipts:
            state_path = project / entry.RUNS / receipt["label"] / "details/kv-diagnostic-runtime-state.json"
            state = read_json(state_path)
            require(state["status"] == "RESTORED" and state["capture_installed"] is True
                and state["capture_receipt"] is not None, "actual KV capture runtime evidence required")
            original_runtime = read_json(state_path.parent / "calibration-runtime-result.json")
            require(state["mode"] == receipt["mode"] and state["label"] == receipt["label"]
                and state["source_lock_sha256"] == scope["source_lock"]["sha256"] and state["scope_sha256"] == scope_sha
                and state["process_identity"] == original_runtime["process_identity"]
                and state["capture_module_ref"] == config["capture_ref"] and state["storage"] == str(project / STORAGE),
                "capture belongs to actual original process/source/scope/private storage")
            receipt_path = checked(project, state["capture_receipt_ref"])
            require(receipt_path == state_path.parent / "kv-capture/native-kv-byte-receipt.json"
                and read_json(receipt_path) == state["capture_receipt"], "same byte-bound capture file and inlined producer evidence")
            if receipt["mode"] == "paired":
                producer = project / entry.RUNS / JOBS["populate"] / "details/kv-capture/native-kv-byte-receipt.json"
                require(state["producer_capture_ref"] == ref(project, producer.relative_to(project).as_posix()),
                    "actual paired installer consumed the exact verified producer receipt")
                publication = producer.parent.parent / "storage-publication.json"
                require(state["producer_publication_ref"] == ref(project, publication.relative_to(project).as_posix()),
                    "paired capture bound to current producer cache publication")
                actual_producer = state["capture_receipt"]["producer_manifest"]
                expected_producer = dict(state["producer_capture_ref"], path=str(producer))
                require(actual_producer == expected_producer, "capture helper read exactly the wrapper-verified producer receipt")
            capture = load(project, config["capture_ref"], "validate")
            try:
                capture.validate_receipt(state["capture_receipt"], mode=receipt["mode"])
                require(validate_capture_cohort(state["capture_receipt"],
                    read_json(state_path.parent / "acquisition/result.json"), receipt["mode"]) == state["cohort_binding"],
                    "same actual original request cohort after original guard closure")
            finally:
                sys.modules.pop(capture.__name__, None)
        require(entry.storage_snapshot(project)["free_bytes"] >= 8*1024**3,
            "actual post-result PRIMARY storage remains above 8GiB")
        return receipts
    entry.prior_results = prior_results
    def cleanup():
        for name in tuple(sys.modules):
            if name not in prior and name.startswith(("_server10_kv_", "_g3_pilot_", "_g2_lifecycle_")):
                sys.modules.pop(name)
    return entry, plan, runtime, cleanup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=MODES, default="populate")
    parser.add_argument("--source-lock", default=LOCK)
    parser.add_argument("--scope-record", default=SCOPE)
    actions = parser.add_mutually_exclusive_group()
    for action in ("preflight", "launch", "execute"):
        actions.add_argument("--" + action, action="store_true")
    args = parser.parse_args(argv)
    require(args.scope_record == SCOPE, "distinct KV diagnostic authorization scope")
    entry, plan, runtime, cleanup = build_facade(args.project, args.source_lock)
    try:
        command = ["--project", str(args.project), "--mode", args.mode, "--source-lock", args.source_lock,
                   "--scope-record", args.scope_record]
        command += ["--" + action for action in ("preflight", "launch", "execute") if getattr(args, action)]
        return entry.main(command)
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
