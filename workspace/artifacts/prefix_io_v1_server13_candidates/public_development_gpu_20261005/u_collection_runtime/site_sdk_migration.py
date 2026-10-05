"""Reuse the original CUDA13 layout with separately audited current driver bytes.

Historical compiler evidence remains bound to 580; it does not prove compiling,
linking, loading or GPU execution on 595. No original receipt/source is changed.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
import types

SOURCE = "artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/sdk_migration/site_sdk_migration.py"
ORIGINAL_SITE = "artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/site_sdk_binding.py"
ORIGINAL_SITE_SHA = "35c3d68ff2c53e0213806331d98e94ac0dba7c3c3d97b20e6e9f7fdfc2a274e8"
HELPER_SHA = "ed846361788e9bdde853e6e0aa16e361329e3f2dffbfbe2c4f396acf9e9d1ecf"
MIGRATION_REF = dict(path="artifacts/prefix_io_v1/server13-public-development-gpu-20261005/ACTUAL_DRIVER_MIGRATION_AUDIT_01.json",
    bytes=1413, sha256="9874640f0ab80d847d6adcf77dd71c7f6e5c540ed782d855473cf0cfa3818a3c")
CURRENT_DRIVER = dict(path="/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05", bytes=91501576,
    sha256="76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6")
HISTORICAL_DRIVER = dict(path="/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05", bytes=96276264,
    sha256="f27223c58d4c0d2ead3c2d747eb30a530c449f89c42e16b23e9bef04f3e6dc2e")
ALIASES = ("/usr/lib/x86_64-linux-gnu/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def require(value, reason):
    if not value:
        raise ValueError("SDK_DRIVER_MIGRATION_REJECTED: " + reason)


def _load_source(root, refs, relative, expected_sha=None):
    row = refs.get(relative)
    require(type(row) is dict and row.get("path") == relative and
            (expected_sha is None or row.get("sha256") == expected_sha), "exact closed migration dependency")
    path = Path(root) / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),
            "contained regular source")
    raw = path.read_bytes()
    require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "actual dependency bytes")
    name = "_source_only_migrated_SDK_" + str(id(raw))
    require(name not in sys.modules, "fresh private source module")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
        require(path.read_bytes() == raw, "source changed during load")
        return module, raw, dict(row)
    finally:
        sys.modules.pop(name, None)


def _validate_audit(document, root):
    require(type(document) is dict and set(document) == {"old_driver", "old_cudart", "current_driver_paths",
            "SDK_size_drift", "actual_driver_dynamic_soname"}, "exact actual driver migration audit")
    require(document["old_driver"] == HISTORICAL_DRIVER and document["SDK_size_drift"] == [] and
            document["actual_driver_dynamic_soname"] == [], "unaltered historical driver/audit scope")
    cudart = dict(path=str(Path(root).resolve()/".venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"),
                 bytes=704288, sha256="96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0")
    require(document["old_cudart"] == cudart, "same actual installed CUDA13 runtime")
    paths = document["current_driver_paths"]
    require(type(paths) is dict and set(paths) == set(ALIASES) | {CURRENT_DRIVER["path"], HISTORICAL_DRIVER["path"]},
            "complete actual driver alias/target audit")
    for path in ALIASES + (CURRENT_DRIVER["path"],):
        require(paths[path] == dict(resolved_path=CURRENT_DRIVER["path"], bytes=CURRENT_DRIVER["bytes"],
                sha256=CURRENT_DRIVER["sha256"], is_symlink=path in ALIASES), "real current driver audit identity")
    require(paths[HISTORICAL_DRIVER["path"]] == dict(resolved_path=HISTORICAL_DRIVER["path"], bytes=0,
            sha256=hashlib.sha256(b"").hexdigest(), is_symlink=False), "retain original empty host-injected old path evidence")
    return dict(schema="source_closed_existing_CUDA13_driver_migration_v1", migration_ref=dict(MIGRATION_REF),
        current_driver_ref=dict(CURRENT_DRIVER), historical_compiler_driver_ref=dict(HISTORICAL_DRIVER),
        compiler_proof_current_driver_matched=False, current_driver_compile_link_executions=0,
        actual_GPU_operations=0, GPU_model_or_JIT_runtime_qualified=False)


def validate_migration_binding(root, refs, migration_ref):
    """Small source/audit closure only; does not rehash the SDK or load a library."""
    require(migration_ref == MIGRATION_REF and refs.get(MIGRATION_REF["path"]) == MIGRATION_REF,
            "actual fixed migration audit in current source closure")
    site, _, _ = _load_source(root, refs, ORIGINAL_SITE, ORIGINAL_SITE_SHA)
    require(site.checked(root, refs, MIGRATION_REF["path"]) == MIGRATION_REF, "actual migration bytes")
    return _validate_audit(site.read(root, MIGRATION_REF["path"]), root)


def _derive_asset_verifier(helper, raw):
    """Keep every original check; replace only the historical proof/driver join."""
    require(hashlib.sha256(raw).hexdigest() == HELPER_SHA, "exact immutable original SDK helper")
    function = next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name == "verify_assets")
    derived = deepcopy(function)
    derived.name = "_verify_current_assets_with_historical_compile_evidence"
    target = ast.dump(ast.parse('proof.get("driver_ref") == pin["driver"]', mode="eval").body)
    changed = 0
    for node in ast.walk(derived):
        if isinstance(node,ast.Compare) and ast.dump(node) == target:
            node.comparators[0] = ast.Name(id="_actual_historical_compile_driver",ctx=ast.Load())
            changed += 1
    require(changed == 1, "one original driver/proof association; all SDK/command/output checks retained")
    environment = dict(helper.__dict__)
    environment["_actual_historical_compile_driver"] = dict(HISTORICAL_DRIVER)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[])),
                 "<original_asset_verifier_one_historical_driver_join>", "exec", dont_inherit=True),environment)
    return environment[derived.name]


def _verify_current_driver(helper):
    path = Path(CURRENT_DRIVER["path"])
    require("stubs" not in path.parts and path.parent == Path("/usr/lib/x86_64-linux-gnu"),
            "real host driver, never compiler-only stubs")
    helper.digest_file(path, CURRENT_DRIVER["bytes"], CURRENT_DRIVER["sha256"])
    for alias in ALIASES:
        link = Path(alias)
        require(link.is_symlink() and link.resolve(strict=True) == path, "actual runtime driver alias targets")


def _assets(root, refs, migration_ref):
    root = Path(root).resolve(strict=True)
    migration = validate_migration_binding(root, refs, migration_ref)
    site, _, site_ref = _load_source(root, refs, ORIGINAL_SITE, ORIGINAL_SITE_SHA)
    rows = {relative:site.checked(root,refs,relative) for relative in site.PINS}
    inventory, proof, result = (site.read(root,name) for name in site.SDK_EVIDENCE)
    inventory_ref, proof_ref = site.validate_receipts(root, rows, inventory, proof, result)
    helper, raw, helper_ref = _load_source(root, refs, site.HELPER, HELPER_SHA)
    actual_verify = _derive_asset_verifier(helper, raw)
    state = {"reading_historical_inventory":False}
    def verify(pin):
        require(pin["driver"] == (HISTORICAL_DRIVER if state["reading_historical_inventory"] else CURRENT_DRIVER),
                "historical inventory and current layout driver remain separately identified")
        require(pin["inventory_ref"] == inventory_ref and pin["compiler_cpu_proof"] == proof_ref,
                "same actual unmodified historical compiler/inventory leaves")
        current = dict(pin,driver=dict(CURRENT_DRIVER))
        _verify_current_driver(helper)
        return actual_verify(current)
    helper.verify_assets = verify
    state["reading_historical_inventory"] = True
    try:
        historical_pin = helper.load_audited_assets(inventory_ref,proof_ref)
    finally:
        state["reading_historical_inventory"] = False
    require(historical_pin["driver"] == HISTORICAL_DRIVER, "never relabel original inventory or compiler proof")
    current_pin = dict(historical_pin,driver=dict(CURRENT_DRIVER))
    require(sum(len(tree["files"]) for tree in current_pin["trees"].values()) == 1934 and
            sum(tree["bytes"] for tree in current_pin["trees"].values()) == 217111529,
            "same complete original 1934 CUDA13 toolchain leaves")
    for relative,row in rows.items():
        require(site.checked(root,refs,relative) == row, "historical source/receipt drift after asset replay")
    require(validate_migration_binding(root,refs,migration_ref) == migration, "migration audit unchanged")
    return helper,current_pin,site,rows,site_ref,helper_ref,migration


def preflight_site_sdk(root, refs, migration_ref):
    """Full CPU asset read only. Returned JSON cannot authorize CUDA execution."""
    _, _, _, rows, site_ref, helper_ref, migration = _assets(root,refs,migration_ref)
    return dict(migration,schema="existing_CUDA13_current_driver_CPU_asset_preflight_v1",
        SDK_source_files=1934,SDK_source_bytes=217111529,CPU_assets_verified=True,
        historical_inventory_ref=rows[next(k for k in rows if k.endswith("/CUDA13_SOURCE_INVENTORY.json"))],
        historical_compiler_proof_ref=rows[next(k for k in rows if k.endswith("/CPU_COMPILE_LINK_RESULT.json"))],
        original_site_sdk_adapter_ref=site_ref,helper_ref=helper_ref,
        shared_objects_loaded=False,overlay_created=False)


preflight_migrated_site_sdk = preflight_site_sdk


def prepare_site_sdk(root, refs, out, *, migration_ref):
    """Original private layout/environment only; actual guarded runtime must qualify CUDA."""
    root = Path(root).resolve(strict=True)
    out = Path(out)
    require(out.is_absolute() and out.resolve(strict=True).is_relative_to(root) and
            out.name == "details" and not out.is_symlink(), "actual guarded child details directory")
    helper,pin,site,rows,site_ref,helper_ref,migration = _assets(root,refs,migration_ref)
    evidence = helper.prepare_overlay(approved_run_root=out.parent,runtime_cache=out/"runtime-cache",
        pin=pin,inherited_environment=dict(os.environ))
    selected = dict(evidence.environment)
    os.environ.update(selected)
    return dict(overlay=evidence.overlay,links=evidence.links,
        selected_environment={key:selected[key] for key in site.PROCESS_KEYS},
        inventory_ref=rows[site.INVENTORY],compiler_proof_ref=rows[site.PROOF],helper_ref=helper_ref,
        original_site_sdk_adapter_ref=site_ref,site_sdk_adapter_ref=refs[SOURCE],
        existing_rebind_receipt_ref=rows[site.RESULT],current_driver_ref=dict(CURRENT_DRIVER),
        historical_compiler_driver_ref=dict(HISTORICAL_DRIVER),compiler_proof_current_driver_matched=False,
        migration_ref=dict(migration_ref),CPU_assets_verified=True,GPU_model_or_JIT_runtime_qualified=False,
        production_qualified=False,stubs_on_runtime_library_path=False,
        compiler_executions_this_action=0,GPU_operations_this_action=0,shared_objects_loaded_this_action=False)
