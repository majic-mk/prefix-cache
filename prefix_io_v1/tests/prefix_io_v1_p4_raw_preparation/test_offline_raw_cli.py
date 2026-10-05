"""Offline executable CLI against byte-pinned CPU fixtures, never GPU samples."""
import importlib.util,json,sys
from dataclasses import asdict
from pathlib import Path
import pytest
from prefix_io_control.p4_production_table_contract import EvidenceRef,TableContractError
SCRIPTS=Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("p4_raw_cli_test",SCRIPTS/"prepare_p4_raw_pair.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import prepare_p4_gpu_next_day as base
from tests.prefix_io_v1_p4_raw_preparation.test_explicit_raw_preparation import Fixture,args

@pytest.fixture
def raw_project(tmp_path):
    f=Fixture(tmp_path)
    p=tmp_path/base.PERMISSION;p.parent.mkdir(parents=True)
    p.write_text("allow_gpu_runs: true\n")
    a=dict(schema_version=1,allow_project_local_edits=True,allow_cpu_tests=True,
       allow_gpu_initialization=False,allow_gpu_runs=False,allow_model_downloads=False,
       allow_driver_or_system_changes=False,base_permissions=base.ref(tmp_path,base.PERMISSION))
    dest=tmp_path/base.AUTH_CHOICES[0];dest.parent.mkdir(parents=True);dest.write_text(json.dumps(a))
    bundle=args(f);bundle["context"]=asdict(bundle["context"])
    bundle.update(schema_version=1,scope="p4_explicit_pair_observations")
    bref=f.dump("bundle.json",bundle)
    f.dump("bundle-ref.json",asdict(bref));f.dump("plan-ref.json",asdict(f.plan_ref))
    f.dump("verifier-ref.json",asdict(f.verifier_ref))
    return f

def argv(f,output="assembled"):
    return ["--project",str(f.root),"--bundle","bundle.json","--bundle-ref","bundle-ref.json",
        "--plan","plan.json","--plan-ref","plan-ref.json","--verifier-ref","verifier-ref.json",
        "--output-dir",base.OUT+"/"+output]

def test_real_cli_roundtrip_assembles_all_five_roles_and_stays_blocked(raw_project):
    f=raw_project;assert m.main(argv(f))==0
    receipt=json.loads((f.root/base.OUT/"assembled/receipt.json").read_text())
    assert receipt["cpu_mock_cells"]==1
    assert not receipt["production_qualified"] and not receipt["gpu_verified"]
    assert receipt["GPU_runs_performed"]==0 and not receipt["GPU_collector_implemented"]
    assert receipt["P5_SLO"] is None and not receipt["P5_allowed"]
    assert set(receipt["raw_refs"])=={"baseline_wrapper","action_wrapper","baseline_observations","action_observations"}
    with pytest.raises(ValueError):m.main(argv(f))

def test_raw_check_launch_before_input_or_backend_import(raw_project,monkeypatch):
    f=raw_project;(f.root/"bundle.json").unlink()
    before=set(sys.modules)
    assert m.main(argv(f)+["--check-launch"])==78
    assert not (set(sys.modules)-before)&{"torch","vllm","py_kvcache"}
    assert not (f.root/base.OUT).exists()

def test_changed_raw_bytes_cannot_rebind_pinned_input(raw_project):
    f=raw_project;(f.root/"bundle.json").write_text("{}")
    with pytest.raises(TableContractError):m.main(argv(f))
    assert not (f.root/base.OUT).exists()

def test_missing_exact9_cannot_be_assembled(raw_project):
    f=raw_project;bundle=json.loads((f.root/"bundle.json").read_text());del bundle["load"]["active_decode"]
    ref=f.dump("bundle.json",bundle);f.dump("bundle-ref.json",asdict(ref))
    with pytest.raises(ValueError):m.main(argv(f))
    assert not (f.root/base.OUT).exists()

def test_ref_path_substitution_rejected(raw_project):
    f=raw_project;ref=f.dump("different.json",json.loads((f.root/"bundle.json").read_text()))
    f.dump("bundle-ref.json",asdict(ref))
    with pytest.raises(ValueError):m.main(argv(f))
