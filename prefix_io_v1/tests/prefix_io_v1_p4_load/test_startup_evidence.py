"""Startup evidence integration uses actual CPU raw fixtures, never GPU grants."""
from dataclasses import asdict
import copy
import pytest
from prefix_io_control.p4_options import parse_p4_options,build_p4_kwargs,P4_KEY
from prefix_io_control.p4_startup_evidence import parse_prepared_cost_request
from tests.prefix_io_v1_p4_02_bridge.test_options import options,fixed_raw
from tests.prefix_io_v1_p4_measurements.test_semantic_verifier import Fixture

def raw_request(f):
    return dict(kind="paired_semantic_candidate",candidate_path="candidate.json",
        expected_context=asdict(f.context),qualification_ref=asdict(f.qualification_ref),
        expected_verifier_ref=asdict(f.verifier_ref),plan_path="plan.json",
        expected_plan_ref=asdict(f.plan_ref))

@pytest.mark.parametrize("mode",["shadow","interference","joint"])
@pytest.mark.parametrize("origin",["cpu_fixture","native_gpu_recording"])
def test_real_semantic_prepared_fixture_never_installs_native_mock_table(tmp_path,mode,origin):
    f=Fixture(tmp_path,origin=origin)
    raw=options(mode)
    raw["prefix_io_parent_admission"]["run_id"]=f.context.run_id
    raw[P4_KEY]["run_id"]=f.context.run_id
    caps=fixed_raw();caps["run_id"]=f.context.run_id
    raw[P4_KEY]["fixed_stage_policy"]=caps
    raw[P4_KEY]["cost_table"]=raw_request(f)
    parsed=parse_p4_options(raw)
    with pytest.raises(ValueError):build_p4_kwargs(parsed)
    kwargs=build_p4_kwargs(parsed,evidence_root=tmp_path)
    bridge=kwargs["p4_bridge"]
    assert bridge.policy.table is None
    assert not bridge.policy.production_interference_qualified
    assert ("dispatch_controller" in kwargs)==(mode=="shadow")
    summary=dict(bridge.prepared_cost_summary)
    assert summary["cell_count"]==1 and not summary["gpu_verified"]

@pytest.mark.parametrize("change",[
    lambda x:x.update(gpu_verified=True),
    lambda x:x.update(candidate_path="../candidate.json"),
    lambda x:x.update(candidate_path="/candidate.json"),
    lambda x:x.update(kind="qualified"),
    lambda x:x["expected_plan_ref"].update(path="other.json"),
])
def test_strict_startup_request_cannot_grant_or_escape(tmp_path,change):
    f=Fixture(tmp_path);r=raw_request(f);change(r)
    with pytest.raises(ValueError):parse_prepared_cost_request(r)

def test_changed_raw_bytes_are_rejected_at_startup(tmp_path):
    f=Fixture(tmp_path);request=parse_prepared_cost_request(raw_request(f))
    (tmp_path/"baseline_observations.json").write_text("{}")
    with pytest.raises(ValueError):request.load(tmp_path)

def test_dependency_arm_cannot_consume_an_interference_request(tmp_path):
    f=Fixture(tmp_path);raw=options()
    raw[P4_KEY]["cost_table"]=raw_request(f)
    with pytest.raises(ValueError):parse_p4_options(raw)
