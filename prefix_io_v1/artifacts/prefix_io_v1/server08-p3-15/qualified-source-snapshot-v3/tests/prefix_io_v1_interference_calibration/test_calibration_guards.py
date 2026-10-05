"""CPU-only guards for the new synthetic calibration boundary."""
import copy, importlib.util, sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/"experiments/prefix_io_v1/scripts"
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))
import run_decode_interference_calibration as cal
from decode_interference_worker_probe import is_pure_decode,expected_stage_bytes,TimedAccounting,bind_native_request

def frozen(partition="calibration"):
    s=cal.candidate_spec("session-a","GPU-test",partition)
    s["source_sha256"]={p:"0"*64 for p in s["source_sha256"]}
    if partition=="validation":
        s["reference_results"]=[dict(path="experiments/prefix_io_v1/runs/ref"+str(i)+"/result.json",
            sha256=str(i)*64) for i in (1,2)]
    return s

def test_single_active_request_fits_real_context_and_combined_budget():
    s=cal.validate_spec(frozen())
    assert s["active_decode_requests"]==1
    assert len(s["prompt_token_ids"])+s["sampling"]["max_tokens"]==16385
    assert s["engine"]["max_num_seqs"]==2
    assert s["engine"]["kv_cache_memory_bytes"]+s["owned"]["owned_bytes"]==2*1024**3
    assert s["owned"]["tensor_count"]*s["owned"]["page_bytes"]==917504
    assert s["table_domain"]["production_state_fallback"] is None
    assert s["table_domain"]["cost_permit_reused"] is False
    assert s["table_domain"]["p4_connected"] is False

@pytest.mark.parametrize("mutate",[
 lambda s:s.update(active_decode_requests=2),
 lambda s:s.update(active_decode_requests=True),
 lambda s:s["engine"].update(kv_cache_memory_bytes=2*1024**3),
 lambda s:s["engine"].update(kv_transfer_config={"kv_connector":"OffloadingConnector"}),
 lambda s:s["engine"].update(enforce_eager=False),
 lambda s:s["sampling"].update(max_tokens=256),
 lambda s:s["owned"].update(source_units=32),
 lambda s:s["owned"].update(page_bytes=229376),
 lambda s:s["io"].update(iodepth=16),
 lambda s:s["io"].update(load_planner="on"),
 lambda s:s["io"].update(staging_mem=2.0),
 lambda s:s["pulse"].update(maximum_parent_units=17),
 lambda s:s["pulse"].update(interval_decode_steps=1),
 lambda s:s["table_domain"].update(production_resource_witness=True),
 lambda s:s["table_domain"].update(production_state_fallback=0),
 lambda s:s["table_domain"].update(cost_permit_reused=True),
 lambda s:s.update(sequence=["A","B"]),
 lambda s:s["cells"].append({"anchor":"joint","units":16}),
 lambda s:s.update(prompt_token_ids=s["prompt_token_ids"]+[13]*129),
 lambda s:s["prompt_token_ids"].__setitem__(0,32000),
 lambda s:s.update(prompt_family="owned-validation"),
 lambda s:s.update(gpu_uuid="0"),
 lambda s:s.update(native_worktree="third_party/unregistered"),
 lambda s:s["source_sha256"].pop(next(iter(s["source_sha256"]))),
 lambda s:s["source_sha256"].update(unlocked="0"*64),
 lambda s:s.update(unfrozen_mode="prod"),
])
def test_scope_and_resource_mutations_fail_before_runtime(mutate):
    s=frozen();mutate(s)
    with pytest.raises(ValueError):cal.validate_spec(s)

def test_validation_requires_two_frozen_independent_results():
    assert cal.validate_spec(frozen("validation"))["partition"]=="validation"
    s=frozen("validation");s["reference_results"].pop()
    with pytest.raises(ValueError):cal.validate_spec(s)
    s=frozen("validation");s["reference_results"][1]=dict(s["reference_results"][0])
    with pytest.raises(ValueError):cal.validate_spec(s)

@pytest.mark.parametrize("path",[
 "/tmp/result.json","experiments/prefix_io_v1/runs/../other/result.json",
 "experiments\\prefix_io_v1\\runs\\result.json",
 "experiments/prefix_io_v1/runs/ref/result.json\0",
 "experiments/prefix_io_v1/runs/ref/other.json"])
def test_reference_cannot_escape_or_select_other_payload(path):
    s=frozen("validation");s["reference_results"][0]["path"]=path
    with pytest.raises(ValueError):cal.validate_spec(s)

def test_duplicate_spec_keys_fail_without_runtime(tmp_path):
    p=tmp_path/"spec.json";p.write_text('{"scope":"a","scope":"b"}')
    with pytest.raises(ValueError,match="duplicate"):cal.json_load(p)

@pytest.mark.parametrize("counts,new,expected",[
 ({"active":1},[],True),({"active":128},[],False),
 ({"active":True},[],False),({"other":1},[],False),
 ({"active":1,"other":1},[],False),({"active":1},[object()],False),
 ({},[],False),({"active":1},None,False)])
def test_only_original_single_token_decode_is_a_pulse_boundary(counts,new,expected):
    output=SimpleNamespace(num_scheduled_tokens=counts,scheduled_new_reqs=new)
    assert is_pure_decode(output,{"active"}) is expected

@pytest.mark.parametrize("anchor,stages",[
 ("none",()),("warm_h2d",("h2d",)),("d2h_write",("d2h","ssd_write")),
 ("cold_ssd_h2d",("ssd_read","h2d")),("joint",("ssd_read","ssd_write","h2d","d2h"))])
def test_observed_native_chain_bytes_include_each_stage(anchor,stages):
    expected=expected_stage_bytes(anchor,8,12)
    assert all(expected[s]==(8*12*917504 if s in stages else 0) for s in expected)

def test_acceptance_is_not_completion_or_resource_release():
    a=TimedAccounting();a.bind();a.accepted("ssd_read",1,917504)
    before=a.export(include_spans=True)
    assert before["accounting"]["stages"]["ssd_read"]["inflight_bytes"]==917504
    assert before["stage_spans"]==[]
    assert before["accounting"]["resource_release_inferred"] is False
    a.completed("ssd_read",1,917504)
    after=a.export(include_spans=True)
    assert after["accounting"]["outstanding_records"]==0
    assert after["stage_spans"][0]["end_ns"]>=after["stage_spans"][0]["start_ns"]

def test_short_native_completion_remains_a_failed_operation():
    a=TimedAccounting();a.bind();a.accepted("ssd_write",9,917504);a.completed("ssd_write",9,4096)
    s=a.export(include_spans=True)["accounting"]["stages"]["ssd_write"]
    assert s["failed_ops"]==1 and s["transferred_bytes"]==4096
    assert s["completed_requested_bytes"]==917504

def test_provenance_sources_cover_owned_probe_and_actual_native_backend():
    names=cal.source_names(cal.NATIVE_ROOTS[-1])
    assert "experiments/prefix_io_v1/scripts/decode_interference_worker_probe.py" in names
    assert cal.NATIVE_ROOTS[-1]+"/py_kvcache/linux_aio.py" in names
    assert cal.NATIVE_ROOTS[-1]+"/py_kvcache/reactor.py" in names

@pytest.mark.parametrize("suffix,prompt,counts,valid",[
 ("1234abcd",[1,2],{"frontend-1234abcd":16},True),
 ("1234abc",[1,2],{"frontend-1234abc":16},False),
 ("1234abcZ",[1,2],{"frontend-1234abcZ":16},False),
 ("1234abcd",[1,3],{"frontend-1234abcd":16},False),
 ("1234abcd",[1,2],{"other-1234abcd":16},False)])
def test_prearmed_frontend_binds_only_real_frozen_native_request(suffix,prompt,counts,valid):
    row=SimpleNamespace(req_id="frontend-"+suffix,prompt_token_ids=prompt)
    output=SimpleNamespace(scheduled_new_reqs=[row],num_scheduled_tokens=counts)
    if valid:assert bind_native_request(output,"frontend",[1,2])=="frontend-1234abcd"
    else:
        with pytest.raises(RuntimeError):bind_native_request(output,"frontend",[1,2])

def measurement_row():
    return dict(num_cached_tokens=16256,per_token_complete=True,output_tokens=list(range(128)),
        ambiguous_events=[],engine_token_timestamps=[1+i*.01 for i in range(128)],itl_seconds=[.01]*127)

def test_complete_hot_prefix_window_is_eligible_for_calibration():
    cal.validate_measurement_row(measurement_row(),False)

@pytest.mark.parametrize("mutate",[
 lambda r:r.update(num_cached_tokens=0),
 lambda r:r.update(num_cached_tokens=16240),
 lambda r:r.update(num_cached_tokens=True),
 lambda r:r.update(per_token_complete=False),
 lambda r:r["output_tokens"].pop(),
 lambda r:r["ambiguous_events"].append({"event":"coalesced"}),
 lambda r:r["engine_token_timestamps"].__setitem__(1,r["engine_token_timestamps"][0]),
 lambda r:r["itl_seconds"].__setitem__(0,0.0),
 lambda r:r["itl_seconds"].__setitem__(0,float("nan")),
 lambda r:r["engine_token_timestamps"].pop(),
])
def test_different_hot_state_or_incomplete_time_evidence_cannot_be_a_table_cell(mutate):
    row=measurement_row();mutate(row)
    with pytest.raises(ValueError):cal.validate_measurement_row(row,False)

def test_native_corruption_flag_prevents_table_cell():
    with pytest.raises(ValueError):cal.validate_measurement_row(measurement_row(),True)
