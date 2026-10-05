"""Explicit CPU fixture roundtrip; synthetic nanoseconds never prove GPU work."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import importlib.util,json,sys
import pytest
from prefix_io_control.p4_raw_pair_recorder import (
    prepare_raw_pair,write_raw_pair,assemble_raw_pair,ROLES
)
from prefix_io_control.p4_paired_measurement_verifier import load_verification_plan
from prefix_io_control.p4_production_table_contract import TableContractError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"prefix_io_v1_p4_measurements"))
from test_semantic_verifier import Fixture,io

@pytest.fixture
def fixture(tmp_path):
    return Fixture(tmp_path)

def args(f):
    return dict(context=f.context,cell_id="cell-1",load=deepcopy(f.load_state),stage="h2d",units=2,
        action_operations=2,existing_io=io(1,8,"ssd_read"),
        baseline_runs=deepcopy(f.docs["baseline_wrapper"]["runs"]),
        action_runs=deepcopy(f.docs["action_wrapper"]["runs"]),
        baseline_windows=deepcopy(f.docs["baseline_observations"]["windows"]),
        action_windows=deepcopy(f.docs["action_observations"]["windows"]),
        origin=f.docs["baseline_wrapper"]["origin"])

def geometry(f,physical_bytes=16):
    return dict(cell_id="cell-1",load=f.load_state,existing_io=io(1,8,"ssd_read"),
                stage="h2d",physical_bytes=physical_bytes)

@pytest.mark.parametrize("basis",["existing_io_plus_delta","no_io_plus_joint"])
@pytest.mark.parametrize("origin",["cpu_fixture","native_gpu_recording"])
def test_raw_prepare_to_only_estimator_to_final_loader_is_always_cpu(tmp_path,basis,origin):
    f=Fixture(tmp_path,basis=basis,origin=origin);prepared=prepare_raw_pair(**args(f))
    plan=load_verification_plan(f.root,"plan.json",expected_plan_ref=f.plan_ref)
    table,receipt=assemble_raw_pair(f.root,prepared,"recorded",cell_geometry=geometry(f),
        expected_plan=plan,expected_verifier_ref=f.verifier_ref)
    assert tuple(receipt["raw_refs"])==ROLES
    assert table.verification.cells[0].cost.total_ns==119
    assert not table.production_qualified and not table.gpu_verified
    assert not prepared.production_qualified and not prepared.gpu_verified
    assert table.cell_count==1 and receipt["GPU_runs_performed"]==0
    cell=table.verification.cells[0].cost
    assert table.lookup(cell.load_signature,cell.existing_io,"h2d",16) is None

@pytest.mark.parametrize("units",[1,2,4,8])
def test_single_stage_native_finite_unit_geometry(fixture,units):
    f=fixture;data=args(f);ops=min(2,units);data["units"]=units;data["action_operations"]=ops
    for row in data["action_windows"]:row["new_io"]=io(ops,units*8)
    for row in data["action_runs"]:row["completed_new_io"]=io(ops*3,units*8*3)
    p=prepare_raw_pair(**data);assert p.measured_windows==6 and p.paired_runs==3

def test_source_observations_freeze_before_caller_mutation(fixture):
    data=args(fixture);p=prepare_raw_pair(**data)
    data["action_windows"][0]["end_ns"]=0
    frozen=json.loads(dict(p.role_json)["action_observations"])
    assert frozen["windows"][0]["end_ns"]>0

@pytest.mark.parametrize("change",["missing_active","scheduled_domain","growing_context","missing_drain",
                                  "multi_stage","missing_new_IO","aggregate_P3","host_scope",
                                  "bool_counter","bool_units","illegal_units","extra_ops","token_work",
                                  "timing","pair_overlap","empty_wrapper"])
def test_unknown_or_incomplete_observations_fail_closed(fixture,change):
    data=args(fixture)
    if change=="missing_active":del data["load"]["active_decode"]
    elif change=="scheduled_domain":data["load"]["scheduled_decode"]=data["load"].pop("active_decode")
    elif change=="growing_context":data["action_windows"][1]["load"]["context_length"]+=1
    elif change=="missing_drain":data["action_runs"][0]["accepted_io_drained"]=False
    elif change=="multi_stage":data["action_windows"][0]["new_io"][3]=dict(ops=1,bytes=8)
    elif change=="missing_new_IO":del data["action_windows"][0]["new_io"]
    elif change=="aggregate_P3":data["action_runs"][0]["stage_delta"]=io(6,48)
    elif change=="host_scope":data["action_windows"][0]["host_execute_model_ns"]=100
    elif change=="bool_counter":data["action_windows"][0]["new_io"][2]["ops"]=True
    elif change=="bool_units":data["units"]=True
    elif change=="illegal_units":data["units"]=3
    elif change=="extra_ops":data["action_operations"]=3
    elif change=="token_work":data["action_windows"][1]["output_tokens"]=2
    elif change=="timing":data["baseline_windows"][0]["end_ns"]=0
    elif change=="pair_overlap":data["action_runs"][0]["start_ns"]=0
    elif change=="empty_wrapper":data["baseline_runs"]=[]
    with pytest.raises(ValueError):prepare_raw_pair(**data)

def test_raw_output_is_append_only(fixture):
    p=prepare_raw_pair(**args(fixture))
    refs=write_raw_pair(fixture.root,p,"raw")
    assert all(ref.verify(fixture.root).is_file() for ref in refs.values())
    with pytest.raises(ValueError):write_raw_pair(fixture.root,p,"raw")

@pytest.mark.parametrize("path",["../escape","/tmp/escape","a/../b","a\\b"])
def test_raw_output_paths_cannot_escape(fixture,path):
    with pytest.raises(ValueError):write_raw_pair(fixture.root,prepare_raw_pair(**args(fixture)),path)

def test_raw_symlink_rejected(fixture):
    (fixture.root/"alias").symlink_to(fixture.root,target_is_directory=True)
    with pytest.raises(ValueError):write_raw_pair(fixture.root,prepare_raw_pair(**args(fixture)),"alias/raw")

def test_assembly_verifier_digest_rebound_and_never_executed(fixture):
    p=prepare_raw_pair(**args(fixture));plan=load_verification_plan(fixture.root,"plan.json",
        expected_plan_ref=fixture.plan_ref)
    (fixture.root/fixture.verifier_name).write_text("CHANGED")
    with pytest.raises(TableContractError):assemble_raw_pair(fixture.root,p,"raw",cell_geometry=geometry(fixture),
        expected_plan=plan,expected_verifier_ref=fixture.verifier_ref)
    assert not (fixture.root/"raw").exists()

def test_builder_rejects_independence_leak_after_raw_preparation(fixture):
    f=fixture;data=args(f)
    for role in ("baseline_runs","action_runs"):
        data[role][2]["workload_sha256"]=data[role][0]["workload_sha256"]
    f.entries[2]["workload_sha256"]=f.entries[0]["workload_sha256"];f.refresh()
    p=prepare_raw_pair(**data)
    with pytest.raises(TableContractError):
        load_verification_plan(f.root,"plan.json",expected_plan_ref=f.plan_ref)

def test_incomplete_forged_prepared_roles_are_not_written(fixture):
    p=prepare_raw_pair(**args(fixture))
    with pytest.raises(ValueError):write_raw_pair(fixture.root,replace(p,role_json=p.role_json[:1]),"raw")
