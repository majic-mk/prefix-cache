"""CPU contract tests only; these tests execute no GPU acquisition or experiment."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "experiments/prefix_io_v1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import concurrent_capacity_contract_p316 as c

@pytest.fixture
def frozen_inputs(tmp_path):
    baseline = PROJECT / c.BASELINE_MANIFEST
    target = tmp_path / c.BASELINE_MANIFEST
    target.parent.mkdir(parents=True)
    target.write_bytes(baseline.read_bytes())
    candidate = json.loads((PROJECT / c.CANDIDATE).read_text())
    return tmp_path, candidate, candidate["provenance"]["gpu_uuid"]

@pytest.mark.parametrize("domain_id,slots,ceiling,lookahead", [
    ("cap960-l1", 1097, 1089, 1), ("cap1024-l2", 1170, 1162, 2)])
def test_exact_native_capacity_domains(domain_id, slots, ceiling, lookahead):
    domain = c.capacity_domain(domain_id)
    assert domain["slot_count"] == slots
    assert domain["cache_preload_ceiling_slots"] == ceiling >= 1016
    assert domain["preload_lookahead_requests"] == lookahead
    assert domain["io_depth"] == 8
    assert slots * c.QUANTUM + 4095 <= domain["staging_bytes"]

@pytest.mark.parametrize("domain_id", [None, True, 1, "", "cap512-l1", "cap960-l2", "cap1024-l1"])
def test_unknown_capacity_domains_cannot_expand_scope(domain_id):
    with pytest.raises(ValueError):
        c.capacity_domain(domain_id)

@pytest.mark.parametrize("domain_id", c.DOMAIN_IDS)
def test_frozen_manifest_preserves_all_requests_and_tokens(frozen_inputs, domain_id):
    root, candidate, gpu_uuid = frozen_inputs
    baseline = c.load_baseline(root=root)
    manifest = c.make_manifest(domain_id, root=root)
    assert manifest["requests"] == baseline["requests"]
    assert manifest["families"] == baseline["families"]
    assert manifest["engine"] == baseline["engine"]
    assert c.validate(manifest, candidate, gpu_uuid, root=root) == baseline["engine"]

@pytest.mark.parametrize("domain_id", c.DOMAIN_IDS)
@pytest.mark.parametrize("change", [
    "old_schema", "wrong_bytes", "wrong_lookahead", "wrong_slots", "bool_lookahead",
    "arrival", "token", "engine", "output", "baseline_hash", "extra", "all_hit"])
def test_changed_workload_or_geometry_is_rejected(frozen_inputs, domain_id, change):
    root, candidate, gpu_uuid = frozen_inputs
    manifest = c.make_manifest(domain_id, root=root)
    if change == "old_schema":
        manifest["schema_version"] = 1
    elif change == "wrong_bytes":
        manifest["staging_bytes"] -= 4096
    elif change == "wrong_lookahead":
        manifest["preload_lookahead_requests"] += 1
    elif change == "wrong_slots":
        manifest["capacity_domain"]["slot_count"] += 1
    elif change == "bool_lookahead":
        manifest["capacity_domain"]["preload_lookahead_requests"] = True
    elif change == "arrival":
        manifest["requests"][0]["scheduled_time"] += .001
    elif change == "token":
        manifest["families"][3]["tokens"][100] += 1
    elif change == "engine":
        manifest["engine"]["max_num_seqs"] = 1
    elif change == "output":
        manifest["output_tokens"] = 127
    elif change == "baseline_hash":
        manifest["baseline_manifest_sha256"] = "0" * 64
    elif change == "extra":
        manifest["unregistered_change"] = True
    else:
        manifest["profile"] = "all_hit"
    with pytest.raises(ValueError):
        c.validate(manifest, candidate, gpu_uuid, root=root)

def test_baseline_manifest_hash_is_a_real_content_boundary(tmp_path):
    path = tmp_path / "changed.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="baseline manifest changed"):
        c.load_baseline(path)

@pytest.mark.parametrize("domain_id", c.DOMAIN_IDS)
def test_connector_change_preserves_every_other_original_field(frozen_inputs, domain_id):
    _, candidate, _ = frozen_inputs
    original = candidate["provenance"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
    snapshot = copy.deepcopy(original)
    result = c.expected_connector_variant(original, domain_id)
    delta = c.connector_delta(domain_id)
    assert original == snapshot
    assert result == dict(original, **delta)
    assert {k:v for k,v in result.items() if k not in delta} == {
        k:v for k,v in original.items() if k not in delta}

@pytest.mark.parametrize("field,value", [
    ("iodepth", 8), ("iodepth", True), ("staging_mem", .5),
    ("staging_mem", True), ("preload_lookahead_requests", 2),
    ("preload_lookahead_requests", True)])
def test_connector_requires_original_registered_baseline(frozen_inputs, field, value):
    _, candidate, _ = frozen_inputs
    original = copy.deepcopy(candidate["provenance"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"])
    original[field] = value
    with pytest.raises(ValueError):
        c.expected_connector_variant(original, "cap960-l1")

def test_missing_or_cross_domain_declaration_rejected():
    good = dict(capacity_domain_id="cap960-l1", capacity_domain=c.capacity_domain("cap960-l1"))
    with pytest.raises(ValueError):
        c.validate_declaration(good, "cap1024-l2")
    with pytest.raises(ValueError):
        c.validate_declaration({"capacity_domain_id":"cap960-l1"})

def _args(**overrides):
    args = dict(capacity_domain="cap960-l1", domain=16384, max_num_seqs=2,
                iodepth=8, sizes="16256", reps=3, mode="paired",
                prompt_manifest=Path("manifest.json"), curves=None,
                cached_reference_logprobs=False, native_hot_diagnostic=False,
                diagnostic_logprobs=False)
    args.update(overrides)
    return SimpleNamespace(**args)

@pytest.mark.parametrize("domain_id", c.DOMAIN_IDS)
def test_original_point_shape_and_standards_are_fixed(domain_id):
    assert c.validate_acquisition_args(_args(capacity_domain=domain_id)) == c.capacity_domain(domain_id)

@pytest.mark.parametrize("overrides", [
    {"domain":8192}, {"max_num_seqs":1}, {"iodepth":4}, {"sizes":"8192"},
    {"reps":2}, {"mode":"populate"}, {"mode":"planned"},
    {"prompt_manifest":None}, {"curves":Path("curve.json")},
    {"native_hot_diagnostic":True}, {"diagnostic_logprobs":True},
    {"cached_reference_logprobs":True, "mode":"cold"}])
def test_acquisition_cannot_smuggle_unqualified_modes(overrides):
    with pytest.raises(ValueError):
        c.validate_acquisition_args(_args(**overrides))

def test_only_mixed_capacity_development_has_a_storage_contract():
    assert c.required_new_bytes("mixed_readwrite") == 3 * 1024**3
    with pytest.raises(ValueError):
        c.required_new_bytes("all_hit")
