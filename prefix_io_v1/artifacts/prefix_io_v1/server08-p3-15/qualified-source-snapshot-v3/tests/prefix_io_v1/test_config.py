from copy import deepcopy
from pathlib import Path
import pytest
from prefix_io_control.config import (ConfigError, MODES, read_yaml, validate_controller,
    validate_permissions, require_gpu_permission)
from prefix_io_control.preflight import inspect
ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "docs/prefix_io_v1/templates"

@pytest.fixture
def config():
    return read_yaml(T / "controller_spec.yaml")

def test_default_off_retains_unknown_values(config):
    parsed = validate_controller(config)
    assert parsed["mode"] == "off"
    assert parsed["credits"]["snapshot_max_age_ms"] is None
    parsed["credits"]["snapshot_max_age_ms"] = 1
    assert config["credits"]["snapshot_max_age_ms"] is None

@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_inactive_modes_can_be_prepared_without_calibration(config, mode):
    config["mode"] = mode
    validate_controller(config)

@pytest.mark.parametrize("mode", sorted(MODES - {"off", "shadow"}))
def test_active_mode_rejects_null_credits(config, mode):
    config["mode"] = mode
    with pytest.raises(ConfigError, match="null"):
        validate_controller(config)

@pytest.mark.parametrize("value", [True, -1, 0, float("nan"), float("inf"), "20"])
def test_strict_numeric_types(config, value):
    config["credits"]["snapshot_max_age_ms"] = value
    with pytest.raises(ConfigError):
        validate_controller(config)

@pytest.mark.parametrize("section,key,value", [
    (None, "preserve_existing_pipeline", False),
    (None, "allow_early_source_release_protocol_change", True),
    ("observability", "global_gpu_sync_for_statistics", True),
    ("progress", "may_override_memory_or_event_safety", True),
    ("candidate_limits", "max_ready_work_observed", 65),
    ("candidate_limits", "max_dependency_stage_depth", 4),
    ("candidate_limits", "storage_units_per_batch_candidates", [1, 1]),
])
def test_contract_boundaries(config, section, key, value):
    target = config if section is None else config[section]
    target[key] = value
    with pytest.raises(ConfigError):
        validate_controller(config)

def test_unknown_keys_fail(config):
    config["credits"]["silent_quota"] = 100
    with pytest.raises(ConfigError):
        validate_controller(config)

def test_duplicate_yaml_key_fails(tmp_path):
    p = tmp_path / "duplicate.yaml"
    p.write_text("mode: off\nmode: joint\n")
    with pytest.raises(ConfigError, match="unique"):
        read_yaml(p)

def test_mandatory_before_throttling(config):
    config["mode"] = "fixed"
    for key in config["credits"]:
        if config["credits"][key] is None:
            config["credits"][key] = 100
    with pytest.raises(ConfigError, match="mandatory"):
        validate_controller(config)

def test_permission_is_not_implied_by_ssh(tmp_path):
    p = read_yaml(T / "permissions.yaml")
    validate_permissions(p)
    with pytest.raises(ConfigError, match="authorized"):
        require_gpu_permission(p, gpu_id="GPU-0", experiment_root=tmp_path / "run", hours=.1)

@pytest.mark.parametrize("value", [1, "true", None])
def test_permission_flags_are_actual_booleans(value):
    p = read_yaml(T / "permissions.yaml")
    p["allow_gpu_runs"] = value
    with pytest.raises(ConfigError):
        validate_permissions(p)

def test_preflight_cannot_promote_mock_to_gpu_evidence(config):
    result = inspect(config, read_yaml(T / "permissions.yaml"), {
        "runtime": {"VLLM_AVAILABLE": False, "PLAN_API_AVAILABLE": False,
                    "TORCH_COPY_AVAILABLE": False},
        "io_uring": {"available": False}})
    assert result["status"] == "BLOCKED"
    assert result["gpu_executed"] is False and result["gpu_verified"] is False
    assert any("runtime" in x for x in result["blockers"])
