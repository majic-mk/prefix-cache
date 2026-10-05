"""Strict project configuration; these fields are NOT native vLLM options."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
from typing import Any
import yaml

MODES = frozenset(("off", "shadow", "fixed", "pressure", "interference", "dependency_only", "joint"))
class ConfigError(ValueError):
    pass

class UniqueLoader(yaml.SafeLoader):
    pass

def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ConfigError("mapping keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result
UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)

def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ConfigError("configuration is too large")
    value = yaml.load(path.read_text(), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ConfigError("configuration must be a mapping")
    return value

def _positive(value, name, *, integer=False, nullable=False):
    if value is None and nullable:
        return
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be a finite positive {'integer' if integer else 'number'}")

def _keys(value, names, name):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ConfigError(f"{name} keys must be exactly: {', '.join(sorted(names))}")

def _fixed(value, expected, name):
    if type(value) is not type(expected) or value != expected:
        raise ConfigError(f"{name} must remain {expected!r}")

TOP_FIXED = {
 "schema_version": 1, "inherit_upstream_load_planner": True,
 "inherit_upstream_write_admission": True, "preserve_existing_pipeline": True,
 "preserve_native_completion_semantics": True,
 "allow_early_source_release_protocol_change": False,
 "allow_mid_transfer_recompute_switch": False}
LIMITS = ("max_parent_jobs_observed", "max_ready_work_observed",
          "max_release_closures", "max_dependency_stage_depth")
CREDIT_INTS = (
 "max_issued_h2d_bytes_per_epoch", "max_issued_d2h_bytes_per_epoch",
 "max_inflight_h2d_bytes", "max_inflight_d2h_bytes",
 "max_new_ssd_read_ops_per_epoch", "max_new_ssd_write_ops_per_epoch",
 "max_inflight_ssd_ops", "max_accepted_parent_jobs", "actual_staging_budget_bytes")
PROGRESS_FIXED = {"must_preserve_native_continuations": True,
 "allow_performance_budget_override_for_progress": True,
 "may_override_memory_or_event_safety": False}
OBSERVABILITY = {"log_full_kv_in_hot_path": False, "scan_full_kv_pool_each_step": False,
 "global_gpu_sync_for_statistics": False, "record_actual_allocated_bytes": True,
 "record_parent_completion_granularity": True}

def validate_controller(raw: dict) -> dict:
    _keys(raw, set(TOP_FIXED) | {"mode", "candidate_limits", "credits",
          "feedback", "progress", "observability"}, "controller")
    for key, value in TOP_FIXED.items():
        _fixed(raw[key], value, key)
    mode = raw["mode"]
    if type(mode) is not str or mode not in MODES:
        raise ConfigError("unknown controller mode")
    limits = raw["candidate_limits"]
    _keys(limits, set(LIMITS) | {"storage_units_per_batch_candidates"}, "candidate_limits")
    for key, cap in zip(LIMITS, (32, 64, 8, 3)):
        _positive(limits[key], key, integer=True)
        if limits[key] > cap:
            raise ConfigError(f"{key} exceeds the V1 search bound {cap}")
    units = limits["storage_units_per_batch_candidates"]
    if not isinstance(units, list) or not units or len(units) > 4:
        raise ConfigError("at most four storage-unit candidates are required")
    for unit in units:
        _positive(unit, "storage_units_per_batch_candidates", integer=True)
        if unit not in (1, 2, 4, 8):
            raise ConfigError("V1 batch candidates must come from 1,2,4,8")
    if units != sorted(set(units)):
        raise ConfigError("batch candidates must be sorted and unique")
    credits = raw["credits"]
    _keys(credits, set(CREDIT_INTS) | {"grant_once_per_epoch", "snapshot_max_age_ms"}, "credits")
    _fixed(credits["grant_once_per_epoch"], True, "grant_once_per_epoch")
    _positive(credits["snapshot_max_age_ms"], "snapshot_max_age_ms", nullable=True)
    for key in CREDIT_INTS:
        _positive(credits[key], key, integer=True, nullable=True)
    feedback = raw["feedback"]
    _keys(feedback, ("calibration_manifest", "internal_step_budget_ms",
          "uncertainty_margin_policy", "table_miss_action", "stale_snapshot_action"), "feedback")
    _positive(feedback["internal_step_budget_ms"], "internal_step_budget_ms", nullable=True)
    for key in ("calibration_manifest", "uncertainty_margin_policy"):
        if feedback[key] is not None and (type(feedback[key]) is not str or not feedback[key].strip()):
            raise ConfigError(f"{key} must be null or a nonempty string")
    _fixed(feedback["table_miss_action"], "validated_conservative_fallback", "table_miss_action")
    _fixed(feedback["stale_snapshot_action"], "validated_progress_preserving_fallback", "stale_snapshot_action")
    progress = raw["progress"]
    _keys(progress, set(PROGRESS_FIXED) | {"mandatory_signal_verified", "validated_min_progress_unit"}, "progress")
    for key, value in PROGRESS_FIXED.items():
        _fixed(progress[key], value, key)
    if type(progress["mandatory_signal_verified"]) is not bool:
        raise ConfigError("mandatory_signal_verified must be boolean")
    _positive(progress["validated_min_progress_unit"], "validated_min_progress_unit", integer=True, nullable=True)
    _keys(raw["observability"], OBSERVABILITY, "observability")
    for key, value in OBSERVABILITY.items():
        _fixed(raw["observability"][key], value, key)
    if mode not in ("off", "shadow"):
        if any(credits[k] is None for k in (*CREDIT_INTS, "snapshot_max_age_ms")):
            raise ConfigError("active mode requires frozen credits; null is not zero")
        if not progress["mandatory_signal_verified"] or progress["validated_min_progress_unit"] is None:
            raise ConfigError("mandatory progress must be validated before ordinary quotas")
        if mode in ("interference", "joint") and any(feedback[k] is None for k in
             ("calibration_manifest", "internal_step_budget_ms", "uncertainty_margin_policy")):
            raise ConfigError("interference modes require measured calibration")
    return deepcopy(raw)

PERMISSION_BOOLS = ("allow_project_local_edits", "allow_cpu_tests", "allow_public_source_read",
 "allow_remote_push", "allow_new_cloud_rental", "allow_payment", "allow_driver_or_system_changes",
 "allow_shared_data_deletion", "allow_gpu_runs", "allow_model_downloads")
PERMISSION_ROOTS = ("approved_experiment_root", "approved_dependency_root")
def validate_permissions(raw: dict) -> dict:
    optional = {"approved_auxiliary_storage"} if "approved_auxiliary_storage" in raw else set()
    _keys(raw, set(PERMISSION_BOOLS) | set(PERMISSION_ROOTS) |
          {"schema_version", "max_gpu_hours", "max_model_download_gib", "approved_gpu_ids"} | optional, "permissions")
    if optional:
        aux = raw["approved_auxiliary_storage"]
        _keys(aux, ("root", "max_bytes", "minimum_free_bytes", "authorization_record"), "auxiliary storage")
        if type(aux["root"]) is not str or not Path(aux["root"]).is_absolute() or len(Path(aux["root"]).parts) < 3:
            raise ConfigError("auxiliary storage root must be a dedicated absolute directory")
        for key in ("max_bytes", "minimum_free_bytes"):
            _positive(aux[key], key, integer=True)
        if aux["max_bytes"] > 20 * 1024**3 or aux["minimum_free_bytes"] < 8 * 1024**3:
            raise ConfigError("auxiliary storage exceeds the user-approved 20 GiB / 8 GiB contract")
        record = aux["authorization_record"]
        if type(record) is not str or not record or Path(record).is_absolute() or ".." in Path(record).parts:
            raise ConfigError("authorization record must be project relative")
    _fixed(raw["schema_version"], 1, "schema_version")
    for key in PERMISSION_BOOLS:
        if type(raw[key]) is not bool:
            raise ConfigError(f"{key} must be boolean")
    for key in ("max_gpu_hours", "max_model_download_gib"):
        _positive(raw[key], key, nullable=True)
    ids = raw["approved_gpu_ids"]
    if not isinstance(ids, list) or any(type(x) is not str or not x.strip() for x in ids) or len(ids) != len(set(ids)):
        raise ConfigError("approved_gpu_ids must be unique nonempty strings")
    for key in PERMISSION_ROOTS:
        if raw[key] is not None and (type(raw[key]) is not str or not Path(raw[key]).is_absolute()):
            raise ConfigError(f"{key} must be null or absolute")
    return deepcopy(raw)

def require_gpu_permission(raw: dict, *, gpu_id: str, experiment_root: Path, hours: float) -> None:
    """Validate a proposed run only. This function never starts a process."""
    p = validate_permissions(raw)
    _positive(hours, "requested hours")
    if not p["allow_gpu_runs"] or p["max_gpu_hours"] is None or hours > p["max_gpu_hours"]:
        raise ConfigError("GPU execution is not explicitly authorized within a finite budget")
    if gpu_id not in p["approved_gpu_ids"] or any(p[k] is None for k in PERMISSION_ROOTS):
        raise ConfigError("GPU identity and both roots must be explicitly approved")
    root = Path(p["approved_experiment_root"]).resolve()
    target = experiment_root.resolve()
    if target == root or not target.is_relative_to(root):
        raise ConfigError("run directory must be a child of the approved experiment root")
