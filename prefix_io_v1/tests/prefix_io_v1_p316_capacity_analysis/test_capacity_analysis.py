"""Pure CPU contracts for capacity analysis.

All runtime-shaped records below are explicitly synthetic and live in pytest
tmp_path. No real GPU receipt/permit is created, no qualification is issued, and
no fixture proves physical execution. They test parser/guard rejection only.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "experiments/prefix_io_v1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_simple_stage_capacity_p316 as a
import concurrent_capacity_contract_p316 as c


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def replace_path(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


@pytest.fixture
def cpu_defined_manifest(tmp_path):
    # Actual frozen CPU workload/candidate definitions, never a GPU qualification.
    baseline = tmp_path / c.BASELINE_MANIFEST
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_bytes((PROJECT / c.BASELINE_MANIFEST).read_bytes())
    candidate = json.loads((PROJECT / c.CANDIDATE).read_text())
    manifest = c.make_manifest("cap960-l1", root=tmp_path)
    return tmp_path, manifest, candidate, candidate["provenance"]["gpu_uuid"]


def entry_files(root, manifest):
    details = root / "cpu-synthetic-run" / "details"
    manifest_path = write_json(root / "cpu-defined-manifest.json", manifest)
    write_json(details / "manifest.json", manifest)
    report = dict(synthetic_cpu_metadata_not_gpu_evidence=True,
                  manifest_sha256=a.digest(manifest_path),
                  profile=manifest["profile"],
                  capacity_domain_id=manifest["capacity_domain_id"],
                  capacity_domain=manifest["capacity_domain"])
    result_path = write_json(details / "cpu-synthetic-result.json", report)
    # Intentionally unqualified. This must never reach numerical/golden analysis.
    qualification = write_json(root / "cpu-synthetic-unqualified.json",
        dict(schema_version=2, status="CPU_SYNTHETIC_NOT_GPU_QUALIFIED",
             synthetic_cpu_metadata_not_gpu_evidence=True))
    return result_path, manifest_path, qualification


def block_baseline(monkeypatch):
    called = []
    def tripwire(*args, **kwargs):
        called.append(True)
        raise AssertionError("unqualified synthetic input reached baseline audit")
    monkeypatch.setattr(a, "baseline_audit", tripwire)
    return called


def test_entry_rejects_wrong_domain_before_gate(cpu_defined_manifest, monkeypatch):
    root, manifest, _, _ = cpu_defined_manifest
    result, manifest_path, qualification = entry_files(root, manifest)
    called = block_baseline(monkeypatch)
    gate_calls = []
    def gate_tripwire(*args, **kwargs):
        gate_calls.append(True)
        raise AssertionError("mismatched domain reached permit reader")
    monkeypatch.setattr(a, "verify_gate", gate_tripwire)
    with pytest.raises(ValueError, match="capacity domain mismatch"):
        a.audit(result, root / "unused-reference.json", manifest,
                capacity_domain_id="cap1024-l2", qualification=qualification,
                manifest_path=manifest_path, root=root)
    assert called == gate_calls == []


@pytest.mark.parametrize("which", ["argument", "run_snapshot", "reported_hash"])
def test_entry_rejects_manifest_substitution(cpu_defined_manifest, monkeypatch, which):
    root, manifest, _, _ = cpu_defined_manifest
    result, manifest_path, qualification = entry_files(root, manifest)
    argument = copy.deepcopy(manifest)
    if which == "argument":
        argument["requests"][0]["scheduled_time"] += .001
    elif which == "run_snapshot":
        snapshot = copy.deepcopy(manifest)
        snapshot["families"][0]["tokens"][0] += 1
        write_json(result.parent / "manifest.json", snapshot)
    else:
        report = json.loads(result.read_text())
        report["manifest_sha256"] = "0" * 64
        write_json(result, report)
    called = block_baseline(monkeypatch)
    with pytest.raises(ValueError, match="exact frozen run manifest"):
        a.audit(result, root / "unused-reference.json", argument,
                capacity_domain_id="cap960-l1", qualification=qualification,
                manifest_path=manifest_path, root=root)
    assert called == []


@pytest.mark.parametrize("schema", [1, True, "2", None])
def test_real_manifest_provider_rejects_non_schema2(cpu_defined_manifest, schema):
    root, manifest, candidate, gpu = cpu_defined_manifest
    manifest["schema_version"] = schema
    with pytest.raises(ValueError, match="schema"):
        a.validate_manifest(manifest, candidate, gpu, root=root)


@pytest.mark.parametrize("mutation", ["tokens", "arrival", "staging", "lookahead", "engine"])
def test_real_manifest_provider_rejects_changed_frozen_domain(cpu_defined_manifest, mutation):
    root, manifest, candidate, gpu = cpu_defined_manifest
    if mutation == "tokens":
        manifest["families"][0]["tokens"][42] += 1
    elif mutation == "arrival":
        manifest["requests"][1]["scheduled_time"] += .001
    elif mutation == "staging":
        manifest["staging_bytes"] = 1073741824
    elif mutation == "lookahead":
        manifest["preload_lookahead_requests"] = 2
    else:
        manifest["engine"]["max_num_seqs"] = 1
    with pytest.raises(ValueError):
        a.validate_manifest(manifest, candidate, gpu, root=root)


def test_entry_cannot_treat_cpu_definition_as_gpu_qualification(cpu_defined_manifest, monkeypatch):
    root, manifest, _, _ = cpu_defined_manifest
    result, manifest_path, qualification = entry_files(root, manifest)
    called = block_baseline(monkeypatch)
    with pytest.raises(ValueError, match="old or unqualified permit"):
        a.audit(result, root / "unused-reference.json", manifest,
                capacity_domain_id="cap960-l1", qualification=qualification,
                manifest_path=manifest_path, root=root)
    assert called == []


def synthetic_geometry(domain_id="cap960-l1"):
    domain = c.capacity_domain(domain_id)
    capacity = dict(valid=True, observation_valid=True, error=None,
                    physical_release_credit=False, slot_count=domain["slot_count"],
                    io_size=c.QUANTUM)
    live = dict(owner_capture=True, native_shutdown_read=False,
                physical_drain_inferred=False, gpu_release_credit=False,
                native_io_size=c.QUANTUM, staging_capacity=capacity)
    handler = dict(staging_bytes=domain["staging_bytes"], pinned=True,
                   progress_enabled=True, observation_failures=0)
    point = dict(simple_native_control=[live], kv_budget_bytes=2147483648,
                 staging_budget_bytes=domain["staging_bytes"], handlers=[handler])
    report = dict(synthetic_cpu_metadata_not_gpu_evidence=True,
                  native_kv=dict(actual_gpu_kv_bytes=2147483648, groups=1,
                                 layers=28, num_blocks=100,
                                 simple_native_control=[copy.deepcopy(live)]),
                  cohort_probe_start=copy.deepcopy(point), probe=copy.deepcopy(point),
                  final_probe=dict(kv_budget_bytes=2147483648,
                    staging_budget_bytes=domain["staging_bytes"], handlers=[copy.deepcopy(handler)],
                    simple_native_control=[dict(physical_drained=True,
                      admission=dict(count_valid=True, native_drain_unknown=False,
                                     accepted_parents=0))]))
    return domain, report


@pytest.mark.parametrize("domain_id", c.DOMAIN_IDS)
def test_geometry_parser_accepts_only_registered_cpu_shapes(domain_id):
    domain, report = synthetic_geometry(domain_id)
    parsed = a.native_geometry(report, domain)
    assert parsed["slot_count"] == domain["slot_count"]
    assert parsed["staging_budget_bytes"] == domain["staging_bytes"]
    assert parsed["physical_release_credit"] is False
    assert [x["point"] for x in parsed["live_owner_observations"]] == [
        "native_kv", "cohort_probe_start", "probe"]


@pytest.mark.parametrize("path,value", [
    (("native_kv", "actual_gpu_kv_bytes"), 2147483649),
    (("native_kv", "actual_gpu_kv_bytes"), True),
    (("native_kv", "groups"), 2),
    (("native_kv", "layers"), 27),
    (("native_kv", "num_blocks"), 0),
    (("native_kv", "simple_native_control", 0, "owner_capture"), False),
    (("native_kv", "simple_native_control", 0, "native_shutdown_read"), True),
    (("native_kv", "simple_native_control", 0, "physical_drain_inferred"), True),
    (("native_kv", "simple_native_control", 0, "gpu_release_credit"), True),
    (("native_kv", "simple_native_control", 0, "staging_capacity", "valid"), False),
    (("native_kv", "simple_native_control", 0, "staging_capacity", "observation_valid"), False),
    (("native_kv", "simple_native_control", 0, "staging_capacity", "error"), "synthetic-underflow"),
    (("native_kv", "simple_native_control", 0, "staging_capacity", "physical_release_credit"), True),
    (("native_kv", "simple_native_control", 0, "staging_capacity", "slot_count"), 1170),
    (("native_kv", "simple_native_control", 0, "staging_capacity", "slot_count"), True),
    (("native_kv", "simple_native_control", 0, "staging_capacity", "io_size"), 917505),
    (("native_kv", "simple_native_control", 0, "native_io_size"), 917505),
    (("cohort_probe_start", "staging_budget_bytes"), 1073741824),
    (("probe", "kv_budget_bytes"), 2147483649),
    (("probe", "handlers", 0, "staging_bytes"), 1006632961),
    (("probe", "handlers", 0, "pinned"), False),
    (("probe", "handlers", 0, "progress_enabled"), False),
    (("probe", "handlers", 0, "observation_failures"), 1),
    (("final_probe", "staging_budget_bytes"), 1073741824),
    (("final_probe", "simple_native_control", 0, "admission", "count_valid"), False),
    (("final_probe", "simple_native_control", 0, "admission", "native_drain_unknown"), True),
    (("final_probe", "simple_native_control", 0, "admission", "accepted_parents"), 1),
    (("final_probe", "simple_native_control", 0, "physical_drained"), False)])
def test_unknown_overbudget_or_unproved_geometry_is_rejected(path, value):
    domain, report = synthetic_geometry()
    replace_path(report, path, value)
    with pytest.raises(ValueError):
        a.native_geometry(report, domain)


def synthetic_receipt_shape(tmp_path):
    # Field-shape fixture only; never an issued real GPU qualification.
    root = tmp_path
    details = root / "cpu-synthetic-cohort" / "details"
    details.mkdir(parents=True)
    result_path = details / "cpu-synthetic-result.json"
    manifest = root / "cpu-synthetic-manifest.json"
    qualification = root / "cpu-synthetic-unqualified.json"
    config = {"gpu_uuid": "GPU-SYNTHETIC-CPU-FIXTURE"}
    command = [".venv/bin/python", "experiments/prefix_io_v1/scripts/" + a.DRIVER,
               "--capacity-domain", "cap960-l1", "--output", str(details),
               "--manifest", str(manifest), "--qualification", str(qualification),
               "--native-cpu-probe"]
    receipt = dict(synthetic_cpu_metadata_not_gpu_evidence=True,
        label=details.parent.name, gpu_job_attempted=True,
        exit=0, child_exit=0, timed_out=False, error=None, interrupted_signal=None,
        session_drained=True, session_members_after_cleanup=[],
        gpu_uuid=config["gpu_uuid"], command=command)
    path = write_json(details.parent / "result.json", receipt)
    return root, result_path, manifest, qualification, config, path, receipt


@pytest.mark.parametrize("key,value", [
    ("label", "other-cpu-fixture"), ("gpu_job_attempted", False),
    ("exit", 1), ("exit", False), ("child_exit", 1), ("child_exit", False),
    ("timed_out", True), ("error", "synthetic-error"),
    ("interrupted_signal", "synthetic-SIGTERM"), ("session_drained", False),
    ("session_members_after_cleanup", ["synthetic-member"]),
    ("gpu_uuid", "GPU-DIFFERENT-SYNTHETIC-CPU-FIXTURE")])
def test_gpu_receipt_rejects_failed_partial_or_wrong_identity_cpu_shapes(tmp_path, key, value):
    root, result, manifest, qualification, config, path, receipt = synthetic_receipt_shape(tmp_path)
    receipt[key] = value
    write_json(path, receipt)
    with pytest.raises(ValueError, match="completed real guarded GPU receipt"):
        a.cohort_receipt(root, result, manifest, qualification, "cap960-l1", config)


@pytest.mark.parametrize("mutation", [
    "old_driver", "wrong_domain", "wrong_output", "wrong_manifest",
    "wrong_qualification", "no_cpu_probe", "duplicate_domain", "duplicate_cpu_probe"])
def test_gpu_receipt_cannot_reuse_other_driver_domain_or_inputs(tmp_path, mutation):
    root, result, manifest, qualification, config, path, receipt = synthetic_receipt_shape(tmp_path)
    command = receipt["command"]
    if mutation == "old_driver":
        command[1] = "experiments/prefix_io_v1/scripts/run_concurrent_pilot_p316.py"
    elif mutation == "wrong_domain":
        command[command.index("--capacity-domain") + 1] = "cap1024-l2"
    elif mutation == "wrong_output":
        command[command.index("--output") + 1] = str(root / "other-details")
    elif mutation == "wrong_manifest":
        command[command.index("--manifest") + 1] = str(root / "other-manifest.json")
    elif mutation == "wrong_qualification":
        command[command.index("--qualification") + 1] = str(root / "other-permit.json")
    elif mutation == "no_cpu_probe":
        command.remove("--native-cpu-probe")
    elif mutation == "duplicate_domain":
        command.extend(["--capacity-domain", "cap960-l1"])
    else:
        command.append("--native-cpu-probe")
    write_json(path, receipt)
    with pytest.raises(ValueError):
        a.cohort_receipt(root, result, manifest, qualification, "cap960-l1", config)


def synthetic_full128_cohort(tmp_path):
    # Arbitrary CPU tokens and metadata, not model output or physical-drain evidence.
    manifest = dict(profile="mixed_readwrite",
                    families=[dict(name="cpu-family-" + str(i), tokens=[31 + i])
                              for i in range(5)])
    goldens = []
    rows = []
    for i in range(5):
        for kind in ("cold", "gpu_hot"):
            goldens.append(dict(family="cpu-family-" + str(i), kind=kind,
                                prompt_token_ids=[31 + i], output_tokens=[100 + i] * 128))
    for i in range(10):
        times = [100.0 * (i + 1) + j * .01 for j in range(128)]
        intervals = [b - a for a, b in zip(times, times[1:])]
        rows.append(dict(request_id="c2-" + str(i), family="cpu-family-" + str(i % 5),
                         num_cached_tokens=0, output_tokens=[100 + i % 5] * 128,
                         engine_token_timestamps=times, itl_seconds=intervals,
                         per_token_complete=True, ambiguous_events=[],
                         metrics=dict(is_corrupted=False, first_token_latency=.1),
                         itl_p95_seconds=.01))
    stages = {name: dict(accepted_ops=0, accepted_bytes=0,
                        inflight_ops=0, inflight_bytes=0)
              for name in ("ssd_read", "ssd_write", "d2h", "h2d")}
    control = dict(physical_drained=True,
        admission=dict(accepted_parents=0, peak_accepted_parents=0),
        stage_accounting=dict(valid=True, outstanding_records=0, stages=stages),
        controller=None)
    sampling = dict(mode="on", optional_sink_suppressed=False,
                    physical_proof_requires_optional_sampling=False)
    point = dict(optional_native_sampling=sampling)
    probe = dict(point, simple_native_control=[copy.deepcopy(control)],
                 flush_wait_calls=0, pending_flush_wait_calls=0,
                 pending_flush_wait_seconds=0., foreground_slot_unavailable=0)
    final = dict(simple_native_control=[copy.deepcopy(control)],
                 optional_native_sampling=dict(sampling, finish_restore_outside_cohort=True))
    report = dict(synthetic_cpu_metadata_not_gpu_evidence=True,
        status="PASSED_NATIVE_C2_DEVELOPMENT_REPLAY", rows=rows, profile="mixed_readwrite",
        engine_shutdown="completed", source_preservation=dict(checked_files=3048, changed=[]),
        policy_mode="off", native_observation="on", native_cpu_probe_requested=False,
        native_kv=copy.deepcopy(point), cohort_probe_start=copy.deepcopy(point),
        probe=probe, final_probe=final, cohort_seconds_including_drain=2.,
        cohort_read_bytes=1, cohort_write_bytes=0)
    result_path = write_json(tmp_path / "cpu-synthetic-cohort.json", report)
    reference_path = write_json(tmp_path / "cpu-synthetic-full128-reference.json",
        dict(synthetic_cpu_metadata_not_gpu_evidence=True, rows=goldens))
    return manifest, report, result_path, reference_path


def test_frozen_golden_audit_accepts_complete_cpu_vector_shape(tmp_path):
    manifest, _, result, reference = synthetic_full128_cohort(tmp_path)
    value = a.baseline_audit(result, reference, manifest)
    assert value["requests"] == 10 and value["output_tokens"] == 1280
    assert all(row["output_exact"] for row in value["comparisons"])
    assert value["formal_goodput"] is False


@pytest.mark.parametrize("mutation", ["point_one_token", "wrong_last_token", "wrong_prompt"])
def test_full128_golden_cannot_be_replaced_by_cost_point_reference(tmp_path, mutation):
    manifest, _, result, reference = synthetic_full128_cohort(tmp_path)
    golden = json.loads(reference.read_text())
    if mutation == "point_one_token":
        golden["rows"][0]["output_tokens"] = golden["rows"][0]["output_tokens"][:1]
    elif mutation == "wrong_last_token":
        golden["rows"][0]["output_tokens"][-1] += 1
    else:
        golden["rows"][0]["prompt_token_ids"][0] += 1
    write_json(reference, golden)
    with pytest.raises(ValueError, match="output mismatch|token family differs"):
        a.baseline_audit(result, reference, manifest)


@pytest.mark.parametrize("mutation", [
    "source_changed", "shutdown_missing", "physical_not_drained",
    "accounting_invalid", "accepted_disagrees", "cpu_invalid", "pressure_unknown"])
def test_frozen_audit_still_rejects_common_safety_and_observation_failures(tmp_path, mutation):
    manifest, report, result, reference = synthetic_full128_cohort(tmp_path)
    control = report["final_probe"]["simple_native_control"][0]
    if mutation == "source_changed":
        report["source_preservation"]["changed"] = ["synthetic-member.bin"]
    elif mutation == "shutdown_missing":
        report["engine_shutdown"] = "synthetic-not-completed"
    elif mutation == "physical_not_drained":
        control["physical_drained"] = False
    elif mutation == "accounting_invalid":
        control["stage_accounting"]["valid"] = False
    elif mutation == "accepted_disagrees":
        stages = control["stage_accounting"]["stages"]
        control["controller"] = dict(faulted=False, pending_attempt=None,
            uncertain_ops=0, completion_unknown_ops=0,
            observed_api_accepted={s: dict(ops=0, bytes=0) for s in stages})
        stages["ssd_read"]["accepted_ops"] = 1
    elif mutation == "cpu_invalid":
        report["native_cpu_probe_requested"] = True
        for name in ("native_kv", "cohort_probe_start", "probe", "final_probe"):
            report[name]["native_metadata_cpu"] = dict(valid=False, error="synthetic-clock-error")
    else:
        report["policy_mode"] = "pressure"
        report["probe"]["simple_native_control"][0]["staging_capacity"] = dict(
            valid=False, observation_valid=False, error="synthetic-owner-unknown",
            physical_release_credit=False, cache_slots=65, preload_cached_slots=0)
    write_json(result, report)
    with pytest.raises(ValueError):
        a.baseline_audit(result, reference, manifest)
