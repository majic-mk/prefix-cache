"""Three-argument SDK binding for raw original A/B unit-cost observations.

Only the existing source-closed current-driver migration adapter prepares the
guarded child's private SDK. No compiler, model, table or policy is implemented.
Historical compiler receipts remain historical; raw SDK reuse grants no I or
effect qualification. Imports are standard library and source-only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

PREFIX = "artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004"
SOURCE = PREFIX + "/raw_unit_cost/raw_sdk_binding.py"
SDK_REF = dict(path=PREFIX + "/sdk_migration/site_sdk_migration.py", bytes=11359,
    sha256="3fd20b032c16178da8cd2a46e89a74f93cec8ff7464fbf4e5fe689e6be5b1d0f")
MIGRATION_REF = dict(
    path="artifacts/prefix_io_v1/server13-public-development-gpu-20261005/ACTUAL_DRIVER_MIGRATION_AUDIT_01.json",
    bytes=1413, sha256="9874640f0ab80d847d6adcf77dd71c7f6e5c540ed782d855473cf0cfa3818a3c")
CURRENT_DRIVER = dict(path="/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05", bytes=91501576,
    sha256="76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6")


def require(value, reason):
    if not value:
        raise ValueError("RAW_UNIT_COST_SDK_REJECTED: " + reason)


def _checked(root, refs, row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and
            type(row["path"]) is str and type(row["bytes"]) is int and row["bytes"] > 0 and
            type(row["sha256"]) is str and len(row["sha256"]) == 64 and
            all(value in "0123456789abcdef" for value in row["sha256"]) and
            refs.get(row["path"]) == row, "exact current frozen SDK source leaf")
    relative = row["path"]
    require(relative and not Path(relative).is_absolute() and ":" not in relative and
            "\\" not in relative and all(part not in ("", ".", "..") for part in relative.split("/")),
            "safe project-relative SDK source")
    path = root
    for part in relative.split("/"):
        path /= part
        require(not path.is_symlink(), "SDK source symlink refused")
    require(path.is_file() and path.resolve().is_relative_to(root), "regular SDK source inside project")
    raw = path.read_bytes()
    require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
            "actual frozen SDK source bytes unchanged")
    return path, raw


def _binding(root, refs):
    root = Path(root).resolve(strict=True)
    own = refs.get(SOURCE)
    path, _ = _checked(root, refs, own)
    require(path.resolve() == Path(__file__).resolve(), "actual raw SDK binding source, never an ambient module")
    sdk_path, sdk_raw = _checked(root, refs, SDK_REF)
    _checked(root, refs, MIGRATION_REF)
    return root, own, sdk_path, sdk_raw


def _load_source(raw, path):
    name = "_raw_unit_current_driver_sdk_" + str(time.monotonic_ns())
    require(name not in sys.modules, "fresh private SDK source namespace")
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
        return module
    finally:
        sys.modules.pop(name, None)


def preflight_binding(root, refs):
    """Small source/audit replay only; full actual assets remain the SDK's gate."""
    root, own, path, raw = _binding(root, refs)
    sdk = _load_source(raw, path)
    summary = sdk.validate_migration_binding(root, refs, migration_ref=MIGRATION_REF)
    _binding(root, refs)
    return dict(schema="raw_original_unit_cost_SDK_source_binding_v1",
        raw_unit_cost_SDK_adapter_ref=deepcopy(own), site_sdk_adapter_ref=deepcopy(SDK_REF),
        migration_ref=deepcopy(MIGRATION_REF), migration_binding=summary,
        actual_GPU_operations=0, private_cost_table_issued=False, ordinary_I_strategy_authorized=False,
        formal_strategy_effect_qualified=False, historical_compiler_receipts_refit=False)


def prepare_site_sdk(root, refs, out):
    """Retain the original raw probe's three-argument preparation interface."""
    root, own, path, raw = _binding(root, refs)
    sdk = _load_source(raw, path)
    evidence = sdk.prepare_site_sdk(root, refs, out, migration_ref=MIGRATION_REF)
    require(type(evidence) is dict and evidence.get("current_driver_ref") == CURRENT_DRIVER and
            evidence.get("migration_ref") == MIGRATION_REF and
            evidence.get("compiler_proof_current_driver_matched") is False and
            evidence.get("production_qualified") is False and
            evidence.get("GPU_model_or_JIT_runtime_qualified") is False and
            evidence.get("stubs_on_runtime_library_path") is False and
            type(evidence.get("compiler_executions_this_action")) is int and
            evidence["compiler_executions_this_action"] == 0 and
            type(evidence.get("GPU_operations_this_action")) is int and
            evidence["GPU_operations_this_action"] == 0 and
            evidence.get("shared_objects_loaded_this_action") is False,
            "original current-driver SDK layout only, never a new compiler or GPU qualification")
    _binding(root, refs)
    return dict(evidence, raw_unit_cost_SDK_adapter_ref=deepcopy(own),
        private_cost_table_issued=False, ordinary_I_strategy_authorized=False,
        formal_strategy_effect_qualified=False, historical_compiler_receipts_refit=False)
