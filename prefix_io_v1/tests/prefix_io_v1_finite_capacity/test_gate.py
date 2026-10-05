"""Synthetic CPU metadata fixtures; no fixture is real GPU experiment evidence."""
import copy
import json
import sys
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "experiments/prefix_io_v1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import concurrent_capacity_contract_p316 as c
import validate_heldout_costs_capacity_p316 as v
import qualify_capacity_p316 as q

def declaration(domain_id="cap960-l1"):
    return dict(capacity_domain_id=domain_id, capacity_domain=c.capacity_domain(domain_id))

def reference_plan(domain_id="cap960-l1"):
    d = declaration(domain_id)
    d.update(engine_delta={"max_num_seqs":2}, connector_delta=c.connector_delta(domain_id),
             domain=16384, sizes=[16256], reps=3, kv_budget_bytes=2147483648,
             staging_budget_bytes=d["capacity_domain"]["staging_bytes"],
             cached_output_tolerance=0, selected_output_tolerance=0,
             external_groups=[{"label":"cpu-cached", "sizes":[16256]}])
    return d

def cost_plan(domain_id="cap960-l1"):
    d = declaration(domain_id)
    d.update(engine_delta={"max_num_seqs":2}, connector_delta=c.connector_delta(domain_id),
             maximum_absolute_relative_median_error=.25, used_for_fit=False,
             formal_evaluation=False, candidate_sha256=c.CANDIDATE_SHA256,
             jobs=[{"mode":m} for m in ("cold", "paired", "paired", "cold")])
    return d

@pytest.mark.parametrize("domain_id", c.DOMAIN_IDS)
def test_fixed_reference_and_cost_gate_standards(domain_id):
    assert v._domain_plan(reference_plan(domain_id)) == c.capacity_domain(domain_id)
    assert v._domain_plan(cost_plan(domain_id), cost=True) == c.capacity_domain(domain_id)

@pytest.mark.parametrize("key,value", [
    ("sizes",[8192]), ("reps",2), ("domain",8192), ("kv_budget_bytes",1073741824),
    ("cached_output_tolerance",1e-8), ("cached_output_tolerance",False),
    ("selected_output_tolerance",1), ("staging_budget_bytes",1073741824)])
def test_reference_cannot_relax_numerics_or_resource_domain(key, value):
    plan = reference_plan()
    plan[key] = value
    with pytest.raises(ValueError):
        v._domain_plan(plan)

@pytest.mark.parametrize("key,value", [
    ("maximum_absolute_relative_median_error",.26), ("used_for_fit",True),
    ("formal_evaluation",True), ("candidate_sha256","0"*64),
    ("engine_delta",{"max_num_seqs":1}),
    ("connector_delta",{"iodepth":8}),
    ("jobs",[{"mode":"cold"},{"mode":"paired"}])])
def test_cost_gate_cannot_refit_or_reuse_old_domain(key, value):
    plan = cost_plan()
    plan[key] = value
    with pytest.raises(ValueError):
        v._domain_plan(plan, cost=True)

def test_prediction_uses_original_candidate_denominator():
    check = v.prediction_check(2., [2.5]*4, .25)
    assert check["passed"] and check["relative_error"] == .25
    assert not v.prediction_check(2., [2.500001]*4, .25)["passed"]

def _handler(domain_id="cap960-l1"):
    d = c.capacity_domain(domain_id)
    return dict(staging_budget=d["staging_bytes"],
        capacity_geometry=dict(slot_count=d["slot_count"],
            cache_preload_ceiling_slots=d["cache_preload_ceiling_slots"],
            staging_cache_ceiling_slots=d["cache_preload_ceiling_slots"],
            io_depth=8, staging_budget_bytes=d["staging_bytes"], io_quantum_bytes=917504),
        owner_snapshot=dict(owner_capture=True, native_shutdown_read=False,
            physical_drain_inferred=False, gpu_release_credit=False,
            admission=dict(count_valid=True, accepted_parents=0, native_drain_unknown=False),
            staging_capacity=dict(valid=True, observation_valid=True, error=None,
                physical_release_credit=False, slot_count=d["slot_count"])))

def _report(handler):
    return dict(final_drain={"handlers":[copy.deepcopy(handler)]},
                rows=[{"drain":{"handlers":[copy.deepcopy(handler)]}}])

@pytest.mark.parametrize("domain_id", c.DOMAIN_IDS)
def test_actual_geometry_schema_requires_native_owner_snapshot(domain_id):
    v._cached_geometry(_report(_handler(domain_id)), c.capacity_domain(domain_id))

@pytest.mark.parametrize("change", [
    "wrong_slots", "wrong_ceiling", "wrong_budget", "no_owner", "shutdown_read",
    "future_only_credit", "count_unknown", "parents_live", "drain_unknown",
    "capacity_unknown", "sticky_observation_error", "fake_release_credit"])
def test_unknown_or_live_native_resources_reject_qualification(change):
    h = _handler()
    o = h["owner_snapshot"]
    if change == "wrong_slots":
        h["capacity_geometry"]["slot_count"] += 1
    elif change == "wrong_ceiling":
        h["capacity_geometry"]["cache_preload_ceiling_slots"] -= 1
    elif change == "wrong_budget":
        h["staging_budget"] = 1073741824
    elif change == "no_owner":
        o["owner_capture"] = False
    elif change == "shutdown_read":
        o["native_shutdown_read"] = True
    elif change == "future_only_credit":
        o["physical_drain_inferred"] = True
    elif change == "count_unknown":
        o["admission"]["accepted_parents"] = None
    elif change == "parents_live":
        o["admission"]["accepted_parents"] = 1
    elif change == "drain_unknown":
        o["admission"]["native_drain_unknown"] = True
    elif change == "capacity_unknown":
        o["staging_capacity"]["valid"] = False
    elif change == "sticky_observation_error":
        o["staging_capacity"]["observation_valid"] = False
        o["staging_capacity"]["error"] = "underflow"
    else:
        o["staging_capacity"]["physical_release_credit"] = True
    with pytest.raises(ValueError):
        v._cached_geometry(_report(h), c.capacity_domain("cap960-l1"))

@pytest.mark.parametrize("where", ["final_drain", "row"])
def test_missing_handler_cannot_vacuously_prove_staging_capacity(where):
    report = _report(_handler())
    if where == "final_drain":
        report["final_drain"]["handlers"] = []
    else:
        report["rows"][0]["drain"]["handlers"] = []
    with pytest.raises(ValueError):
        v._cached_geometry(report, c.capacity_domain("cap960-l1"))

def _mem_row():
    row = dict(kind="g_mem", prefix_tokens=16, rep=1, warmup=False,
        prompt_token_ids=list(range(17)), output_token_ids=[42], num_cached_tokens=16,
        metrics=dict(is_corrupted=False, first_token_latency=1.),
        trace=dict(foreground_logical_read_bytes=0, preload_actual_read_bytes=0,
            transfer_success=True, h2d_events=1, load_transfers=[{"num_bytes":917504}]),
        drain={"handlers":[{}]})
    return row, copy.deepcopy(row)

@pytest.mark.parametrize("field", ["foreground_logical_read_bytes", "preload_actual_read_bytes"])
def test_original_g_mem_zero_ssd_read_remains_required(field):
    row, reference = _mem_row()
    row["trace"][field] = 917504
    with pytest.raises(ValueError, match="wrong medium"):
        v.verify_measurement(row, reference, 16)

def test_original_exact_top5_has_no_new_tolerance():
    from analyze_cached_references import compare_rows
    left, right = _mem_row()
    top = {str(t):{"logprob":float(-t), "rank":i+1} for i,t in enumerate((42,43,44,45,46))}
    left["output_logprobs"] = [copy.deepcopy(top)]
    right["output_logprobs"] = [copy.deepcopy(top)]
    right["output_logprobs"][0]["42"]["logprob"] += 1e-10
    assert compare_rows(left, right, 16)["top5_exact_equal"] is False

@pytest.fixture
def receipt_parser_storage(monkeypatch):
    # Only receipt-schema unit tests bypass project authorization. Production
    # _phase_receipt still calls the original permission-gated details_path.
    def unit_details_path(project, label, plan):
        assert label == "cpu-only-receipt-schema-fixture"
        return Path(project) / "experiments/prefix_io_v1/runs" / label / "details"
    monkeypatch.setattr(v, "details_path", unit_details_path)

def _receipt_inputs(tmp_path):
    label = "cpu-only-receipt-schema-fixture"
    folder = tmp_path / "experiments/prefix_io_v1/runs" / label / "details"
    folder.mkdir(parents=True)
    receipt = dict(label=label, gpu_job_attempted=True, exit=0, child_exit=0,
        timed_out=False, error=None, interrupted_signal=None, session_drained=True,
        session_members_after_cleanup=[], gpu_uuid="GPU-CPU-SCHEMA-FIXTURE",
        command=["python","experiments/prefix_io_v1/scripts/"+v.ACQUIRER,
                 "--mode","paired","--capacity-domain","cap960-l1"])
    path = folder.parent / "result.json"
    config = dict(gpu_uuid=receipt["gpu_uuid"], engine={"kv_transfer_config":{}})
    report = dict(mode="paired", native_hot_diagnostic=False, cached_reference_logprobs=False)
    return label, path, receipt, config, report

def test_gpu_receipt_schema_binds_new_domain_command(tmp_path, receipt_parser_storage):
    label, path, receipt, config, report = _receipt_inputs(tmp_path)
    path.write_text(json.dumps(receipt))
    proof = v._phase_receipt(tmp_path, label, declaration(), config, report)
    assert proof["label"] == label and proof["sha256"] == c.digest(path)

@pytest.mark.parametrize("change", [
    "not_attempted", "child_failed", "timeout", "not_drained", "members_live",
    "wrong_gpu", "old_acquirer", "cross_domain", "duplicate_domain", "bool_exit"])
def test_unfinished_or_old_gpu_receipt_is_rejected(tmp_path, change, receipt_parser_storage):
    label, path, receipt, config, report = _receipt_inputs(tmp_path)
    if change == "not_attempted":
        receipt["gpu_job_attempted"] = False
    elif change == "child_failed":
        receipt["child_exit"] = 1
    elif change == "timeout":
        receipt["timed_out"] = True
    elif change == "not_drained":
        receipt["session_drained"] = False
    elif change == "members_live":
        receipt["session_members_after_cleanup"] = [123]
    elif change == "wrong_gpu":
        receipt["gpu_uuid"] = "GPU-wrong"
    elif change == "old_acquirer":
        receipt["command"][1] = "experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py"
    elif change == "cross_domain":
        receipt["command"][-1] = "cap1024-l2"
    elif change == "duplicate_domain":
        receipt["command"] += ["--capacity-domain","cap960-l1"]
    else:
        receipt["exit"] = False
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        v._phase_receipt(tmp_path, label, declaration(), config, report)

def test_absent_actual_gpu_receipt_cannot_authorize_capacity(tmp_path, receipt_parser_storage):
    label, path, receipt, config, report = _receipt_inputs(tmp_path)
    with pytest.raises(FileNotFoundError):
        v._phase_receipt(tmp_path, label, declaration(), config, report)

@pytest.mark.parametrize("domain_id", c.DOMAIN_IDS)
def test_old_point_permit_never_crosses_into_capacity_domain(tmp_path, domain_id):
    path = tmp_path / "old-permit.json"
    path.write_text(json.dumps(dict(status="PASSED_C2_UNLOADED_POINT_GATE", engine_delta={"max_num_seqs":2})))
    with pytest.raises(ValueError, match="old or unqualified"):
        q.verify_gate(path, capacity_domain_id=domain_id, root=tmp_path)

def test_new_status_alone_is_not_a_pass_permit(tmp_path):
    path = tmp_path / "unproven.json"
    path.write_text(json.dumps(dict(schema_version=2, status=q.STATUS, **declaration())))
    with pytest.raises(ValueError):
        q.verify_gate(path, capacity_domain_id="cap960-l1", root=tmp_path)

def test_requesting_other_domain_rejects_before_reading_cost_evidence(tmp_path):
    path = tmp_path / "cross.json"
    path.write_text(json.dumps(dict(schema_version=2, status=q.STATUS, **declaration())))
    with pytest.raises(ValueError, match="capacity domain mismatch"):
        q.verify_gate(path, capacity_domain_id="cap1024-l2", root=tmp_path)

def test_source_stamp_change_is_rejected_before_gpu_receipt_lookup(tmp_path, monkeypatch):
    expected = {"acquisition":"a"*64, "contract":"b"*64}
    monkeypatch.setattr(v, "_expected_source_hashes", lambda root: expected)
    config = dict(**declaration(), capacity_source_hashes=expected)
    report = dict(**declaration(), capacity_source_hashes={"acquisition":"a"*64,"contract":"c"*64})
    with pytest.raises(ValueError, match="source changed"):
        v._phase(tmp_path, "none", declaration(), config, report, c.capacity_domain("cap960-l1"))

def test_qualifier_never_writes_pass_when_cost_prediction_fails(tmp_path, monkeypatch):
    cp = tmp_path / "cost-plan.json"
    cp.write_text(json.dumps(cost_plan()))
    monkeypatch.setattr(q, "costs", lambda *args, **kwargs: {"status":"FAILED_HELDOUT_POINT_PREDICTION"})
    output = tmp_path / "permit.json"
    with pytest.raises(ValueError, match="PASS permit will not be issued"):
        q.issue_permit(tmp_path, cp, tmp_path/"result.json", tmp_path/"index.json", output,
                       capacity_domain_id="cap960-l1")
    assert not output.exists()


def test_production_receipt_lookup_requires_real_project_permissions(tmp_path):
    # No receipt_parser_storage fixture: a hand-written success declaration
    # outside an authorized project must not reach the parser.
    label, path, receipt, config, report = _receipt_inputs(tmp_path)
    path.write_text(json.dumps(receipt))
    with pytest.raises(FileNotFoundError, match="permissions.yaml"):
        v._phase_receipt(tmp_path, label, declaration(), config, report)
