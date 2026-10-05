"""Server10 machine binding around the frozen original reference launcher.

The original acquirer, cache, model executor, numerical comparator and budget
guard remain in charge. Historical evidence keeps its original machine UUID.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
OLD = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
DELIVERY = "artifacts/prefix_io_v1/server10-reference-migration-v1-20261003"
SCRIPT = DELIVERY + "/run_server10_reference.py"
PLAN_MODULE = DELIVERY + "/migration_contract.py"
RUNTIME_MODULE = DELIVERY + "/server10_reference_runtime.py"
RESULT_MODULE = OLD + "/g3_reference_result.py"
PLAN = DELIVERY + "/G3_REFERENCE_CPU_PLAN.json"
INPUT = OLD + "/G3_REFERENCE_INPUT_MANIFEST.json"
ANALYZER_PLAN = DELIVERY + "/G3_REFERENCE_ANALYZER_PLAN.json"
MODEL_CONTEXT = DELIVERY + "/G3_REFERENCE_MODEL_CONTEXT.json"
CONFIG = DELIVERY + "/MIGRATION_RUNTIME_CONFIG.json"
LOCK = DELIVERY + "/gpu-source-lock-server10-reference.json"
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server10.reference.yaml"
ENTRY_REF = dict(path=OLD + "/run_g3_reference_pilot.py", bytes=22725,
    sha256="1bd502be997fd1a784ab49aa0262d45dabeca2374a53312d4c96dbdc504ea7a6")
# Filled from the finalized CPU-only adapters before freezing the source lock.
CONTRACT_REF = dict(path=PLAN_MODULE, bytes=20885,
    sha256="1ab4f1a694f695f3736c8d9f1e417a6918f548800af4a4e99f3b7bf54d0e491d")
RUNTIME_REF = dict(path=RUNTIME_MODULE, bytes=12079,
    sha256="9b154ac69f2981f097810a5c824fb10e8f8cbbcce48d2fb0fba471e7c9a678c3")
RESULT_REF = dict(path=RESULT_MODULE, bytes=18442,
    sha256="46efa7026f66da7b44bb0d10df96b5a4f6aaef3e0f7b0073654144ca604415a8")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked(root, row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact source ref")
    name = row["path"]
    require(type(name) is str and name and not Path(name).is_absolute()
            and all(p not in ("", ".", "..") for p in name.split("/"))
            and "\\" not in name, "relative source path")
    path = root / name
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "source path symlink")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root) and path.is_file()
            and path.stat().st_size == row["bytes"], "source bytes/path drift")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == row["sha256"], "source SHA drift: " + name)
    return path, raw


def private_module(root, row, suffix):
    path, raw = checked(root, row)
    name = "_server10_reference_" + suffix + "_" + row["sha256"]
    require(name not in sys.modules, "fresh migration module required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def analyzer_context_ast(raw):
    """Change only the execution-UUID permission comparison, never old evidence."""
    tree = ast.parse(raw)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "original_analyzer_context"]
    require(len(nodes) == 1, "exact original analyzer context")
    node = copy.deepcopy(nodes[0])
    hits = []
    for part in ast.walk(node):
        if (isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name)
                and part.value.id == "context" and isinstance(part.slice, ast.Constant)
                and part.slice.value == "original_config_gpu_uuid"):
            hits.append(part)
    require(len(hits) == 1, "one original permission UUID comparison")
    hits[0].slice.value = "execution_gpu_uuid"
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def build_facade(root, source_lock):
    require(root == ROOT and root.resolve(strict=True) == root, "fixed cloned project root")
    require(CONTRACT_REF is not None and RUNTIME_REF is not None, "final adapter pins required")
    source_path = root / source_lock
    require(source_lock == LOCK and source_path.is_file() and not source_path.is_symlink()
            and source_path.stat().st_size < 2 * 1024**2, "fixed bounded migration lock")
    candidate = json.loads(source_path.read_bytes())
    refs = {row["path"]: row for row in candidate["files"]}
    require(len(refs) == len(candidate["files"]), "no duplicate migration refs")
    for row in (CONTRACT_REF, RUNTIME_REF, ENTRY_REF, RESULT_REF):
        require(refs.get(row["path"]) == row, "fixed original/bootstrap source")
    require(CONFIG in refs and refs[CONFIG]["bytes"] < 65536, "bounded frozen machine config")
    _, cfg_bytes = checked(root, refs[CONFIG])
    cfg = json.loads(cfg_bytes)
    contract = private_module(root, CONTRACT_REF, "contract")
    # Structural ancestry is also rechecked during every actual full-source gate.
    require(cfg["gpu_uuid"] == contract.GPU_UUID
            and cfg["job_names"] == contract.JOB_NAMES
            and cfg["storage"] == {mode: contract.storage_relative(mode) for mode in contract.MODES}
            and cfg["permissions_ref"] == contract.PERMISSIONS_REF, "same machine plan/runtime identity")
    for key in ("permissions_ref", "sdk_inventory_ref", "sdk_proof_ref"):
        require(refs.get(cfg[key]["path"]) == cfg[key], "source-bound runtime asset ref")
        checked(root, cfg[key])
    entry_path, entry_raw = checked(root, ENTRY_REF)
    entry = types.ModuleType("_server10_private_original_entry")
    entry.__file__ = str(entry_path)
    exec(compile(entry_raw, str(entry_path), "exec", dont_inherit=True), entry.__dict__)
    original_loader = entry.load_module
    original_model_context = entry.model_context
    runtime_module = private_module(root, RUNTIME_REF, "runtime")
    migrated_runtime = runtime_module.create_runtime(root, refs, cfg)
    class RuntimeWithEvidence:
        def preflight_runtime(self, project, source_refs):
            return migrated_runtime.preflight_runtime(project, source_refs)

        def execute_original_acquisition(self, project, mode, out, storage, source_refs, witness):
            try:
                return migrated_runtime.execute_original_acquisition(
                    project, mode, out, storage, source_refs, witness)
            finally:
                evidence = dict(schema_version=1, adapter_ref=RUNTIME_REF,
                    migration_config_ref=refs[CONFIG], source_lock_sha256=witness["source_lock_sha256"],
                    scope_sha256=witness["scope_sha256"], gpu_uuid=contract.GPU_UUID,
                    label=contract.JOB_NAMES[mode],
                    state=migrated_runtime.last_migration_evidence)
                entry.new_json(Path(out) / "server10-migration-runtime-state.json", evidence)
    runtime = RuntimeWithEvidence()
    checker = private_module(root, RESULT_REF, "result")
    original_labels = checker.LABELS
    checker.LABELS = copy.deepcopy(contract.JOB_NAMES)
    private_names = [contract.__name__, runtime_module.__name__, checker.__name__]
    before_names = set(sys.modules)
    original_prior_results = entry.prior_results

    def prior_results(project, plan, source_refs, scope, scope_sha, ledger, lock_name):
        receipts, checks = original_prior_results(project, plan, source_refs, scope, scope_sha, ledger, lock_name)
        for receipt in receipts:
            path = project / entry.RUNS / receipt["label"] / "details/server10-migration-runtime-state.json"
            state = entry.read_json(path)
            require(state["adapter_ref"] == RUNTIME_REF and state["migration_config_ref"] == source_refs[CONFIG]
                    and state["source_lock_sha256"] == scope["source_lock"]["sha256"]
                    and state["scope_sha256"] == scope_sha and state["label"] == receipt["label"]
                    and state["gpu_uuid"] == contract.GPU_UUID, "actual migrated runtime evidence identity")
            restoration = state["state"]
            require(type(restoration) is dict and restoration["status"] == "RESTORED_PRIVATE_GLOBAL_BINDINGS"
                    and all(restoration[k] is True for k in ("all_bindings_restored",
                        "all_new_private_modules_unloaded", "preexisting_modules_preserved")),
                    "migration overrides must be restored after real execution")
        return receipts, checks

    def loader(project, relative, expected=None):
        require(project == root, "fixed migration helper project")
        if relative == PLAN_MODULE:
            require(expected == CONTRACT_REF, "contract load pin")
            checked(root, CONTRACT_REF)
            return contract
        if relative == RUNTIME_MODULE:
            require(expected == RUNTIME_REF, "runtime load pin")
            checked(root, RUNTIME_REF)
            return runtime
        if relative == RESULT_MODULE:
            require(expected == RESULT_REF, "result load pin")
            checked(root, RESULT_REF)
            return checker
        return original_loader(project, relative, expected)

    def context(project, source_refs):
        value = original_model_context(project, source_refs)
        require(value["source_gpu_uuid"] == value["original_config_gpu_uuid"]
                == "GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2", "historical source GPU retained")
        require(value["execution_gpu_uuid"] == contract.GPU_UUID, "new execution GPU binding")
        source_ref = value["source_context_ref"]
        require(source_refs.get(source_ref["path"]) == source_ref, "historical context lock binding")
        _, old_bytes = checked(root, source_ref)
        old = json.loads(old_bytes)
        require({k: value[k] for k in old} == old, "historical model context cannot be rewritten")
        return value

    def full_sources(project, lock_name):
        require(project == root and lock_name == LOCK, "fixed migrated source closure")
        lock_ref = entry.ref(project, lock_name)
        actual_refs = contract.verify_source_lock(project, lock_ref)
        require(actual_refs == refs, "pre-read source refs must match actual full source closure")
        required = {SCRIPT, PLAN_MODULE, RUNTIME_MODULE, RESULT_MODULE, PLAN, INPUT, ANALYZER_PLAN,
                    MODEL_CONTEXT, CONFIG, entry.GUARD, PERMISSIONS, *entry.ANALYZERS.values()}
        require(required <= actual_refs.keys(), "complete migrated source and control dependencies")
        for row in (CONTRACT_REF, RUNTIME_REF, RESULT_REF, contract.GUARD_REF):
            require(actual_refs[row["path"]] == row, "exact bootstrap and original guard")
        frozen = entry.read_json(entry.safe(root, PLAN))
        expected = contract.build_reference_plan(frozen["new_source_refs"], actual_refs[INPUT],
                    analyzer_plan_ref=actual_refs[ANALYZER_PLAN])
        contract.same(frozen, expected, "exact typed migrated CPU plan")
        for row in frozen["new_source_refs"]:
            require(actual_refs.get(row["path"]) == row, "CPU plan source pin")
        current_context = context(project, actual_refs)
        require(current_context["alias_target"] == str(root / contract.MODEL_RELATIVE), "original model path")
        analysis = entry.read_json(entry.safe(root, ANALYZER_PLAN))
        checker.validate_reference_plan(analysis, expected_gpu_uuid=contract.GPU_UUID,
            expected_native_storage=str(root / contract.STORAGE_RELATIVE),
            expected_paired_storage=str(root / contract.PAIRED_STORAGE_RELATIVE))
        require(analysis["run_details"] == {label: str(root / entry.RUNS / label / "details/acquisition")
            for label in contract.JOB_NAMES.values()}, "new exact result directories")
        entry.input_check(root, contract, actual_refs)
        with entry.original_analyzer_context(root, actual_refs):
            pass
        runtime.preflight_runtime(root, actual_refs)
        require(not any(n in sys.modules and n not in before_names for n in ("torch", "vllm", "py_kvcache")),
                "full-source preflight is CPU only")
        return contract, actual_refs, lock_ref

    entry.__dict__.update(SCRIPT=SCRIPT, PLAN_MODULE=PLAN_MODULE, RUNTIME_MODULE=RUNTIME_MODULE,
        RESULT_MODULE=RESULT_MODULE, PLAN=PLAN, INPUT=INPUT, ANALYZER_PLAN=ANALYZER_PLAN,
        MODEL_CONTEXT=MODEL_CONTEXT, LOCK=LOCK, PERMISSIONS=PERMISSIONS,
        BOOTSTRAP_PLAN_REF=CONTRACT_REF, BOOTSTRAP_RUNTIME_REF=RUNTIME_REF, BOOTSTRAP_RESULT_REF=RESULT_REF,
        load_module=loader, model_context=context, full_sources=full_sources, prior_results=prior_results)
    exec(compile(analyzer_context_ast(entry_raw), str(entry_path), "exec"), entry.__dict__)

    def restore():
        checker.LABELS = original_labels
        for name in private_names:
            sys.modules.pop(name, None)
    return entry, restore


def main(argv=None):
    pre = argparse.ArgumentParser(description=__doc__)
    pre.add_argument("--project", type=Path, default=ROOT)
    pre.add_argument("--source-lock", default=LOCK)
    pre.add_argument("--mode", choices=("cold", "paired"), default="cold")
    pre.add_argument("--scope-record")
    options = pre.add_mutually_exclusive_group()
    for name in ("preflight", "scope-template", "launch", "execute"):
        options.add_argument("--" + name, action="store_true")
    args = pre.parse_args(argv)
    restore = None
    initial_modules = set(sys.modules)
    try:
        entry, restore = build_facade(args.project.resolve(), args.source_lock)
        return entry.main(argv)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps(dict(status="BLOCKED_SERVER10_MIGRATION_CPU_BOOTSTRAP", reason=str(exc),
                              actual_GPU_runs=0, GPU_initialized=False, performance_claim=False)))
        return 78
    finally:
        if restore is not None:
            restore()
        for name in set(sys.modules) - initial_modules:
            if name.startswith("_server10_reference_"):
                sys.modules.pop(name, None)


if __name__ == "__main__":
    raise SystemExit(main())

