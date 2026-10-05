import ast, importlib.util, json, hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest

SCRIPT = Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts/prepare_p4_gpu_next_day.py"
spec=importlib.util.spec_from_file_location("prepare_next_day",SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
UUID="GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8"

@pytest.fixture
def project(tmp_path):
    root=tmp_path
    runs=root/"experiments/prefix_io_v1/runs";runs.mkdir(parents=True)
    p=root/m.PERMISSION;p.parent.mkdir(parents=True)
    p.write_text("allow_gpu_runs: true\nmax_gpu_hours: 8\napproved_gpu_ids:\n- "+UUID+"\napproved_experiment_root: "+str(runs)+"\n")
    a={"schema_version":1,"allow_project_local_edits":True,"allow_cpu_tests":True,
       "allow_gpu_initialization":False,"allow_gpu_runs":False,
       "allow_model_downloads":False,"allow_driver_or_system_changes":False,
       "base_permissions":m.ref(root,m.PERMISSION)}
    dest=root/m.AUTH_CHOICES[0];dest.parent.mkdir(parents=True);dest.write_text(json.dumps(a))
    (root/m.LEDGER).write_text(json.dumps({"gpu_wall_seconds":16520.5,"active_reservation":None,"events":[]}))
    for script in ["run_gpu_stage.py","qualify_author_copy.py","native_gpu_prefix_smoke.py",*m.LEGACY]:
        path=root/m.SCRIPTS/script;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("import argparse\np=argparse.ArgumentParser()\np.add_argument('--output')\n")
    return root

def plan(root,free=11*1024**3):
    return m.plan(root,disk_usage=lambda _:SimpleNamespace(free=free))

def test_general_gpu_permission_is_not_current_scope(project):
    p=plan(project)
    assert p["current_scope"].startswith("CPU_ONLY")
    assert p["new_gpu_runs"]==p["budget_reserved_seconds"]==0
    assert p["source_freeze_pending"] and p["final_source_lock"] is None
    assert p["gpu_identity_revalidated"] is False
    assert not p["effect_verified"] and not p["full_P4_complete"]

def test_authorization_strict_type_and_changed_permissions(project):
    path=project/m.AUTH_CHOICES[0];a=json.loads(path.read_text())
    a["allow_gpu_initialization"]=0;path.write_text(json.dumps(a))
    with pytest.raises(ValueError):plan(project)
    a["allow_gpu_initialization"]=False;path.write_text(json.dumps(a))
    (project/m.PERMISSION).write_text((project/m.PERMISSION).read_text()+"#changed\n")
    with pytest.raises(ValueError):plan(project)

def test_unresolved_reservation_blocks_budget_not_launches(project):
    ledger=project/m.LEDGER;b=json.loads(ledger.read_text());b["active_reservation"]={"id":"pending"};ledger.write_text(json.dumps(b))
    p=plan(project)
    assert all(not c["budget_fit"] for c in p["command_gates"])
    assert p["budget_reserved_seconds"]==0 and json.loads(ledger.read_text())==b

def test_per_case_storage_gate_preserves_128mib_fit(project):
    p=plan(project,free=m.FLOOR+200*1024**2)
    assert p["command_gates"][0]["PRIMARY_storage_fit"]
    assert not p["command_gates"][1]["PRIMARY_storage_fit"]
    assert not p["storage"]["standard_model_round_fit"]
    assert p["storage"]["standard_model_round_reserve_bytes"]==3*1024**3

def test_gpu_budget_counts_guard_termination_reserve(project):
    ledger=project/m.LEDGER;ledger.write_text(json.dumps({"gpu_wall_seconds":28800-75,"active_reservation":None}))
    p=plan(project)
    assert p["command_previews"][0]["planned_reserved_seconds"]==80
    assert not p["command_gates"][0]["budget_fit"]

def test_legacy_cli_never_claims_p4(project):
    p=plan(project)
    assert all(not row["direct_P4_launch_allowed"] and not row["qualifies_P4"]
               for row in p["legacy_cli_compatibility"].values())
    assert m.NATIVE in p["command_previews"][0]["environment_preview"]["PYTHONPATH"]
    assert "vllm-author-p4-02-cpu" in p["command_previews"][0]["environment_preview"]["PYTHONPATH"]

def test_factorial_exact_labels_no_slo_or_p5(project):
    f=plan(project)["factorial"]
    assert f["C01"]==dict(dependency=False,interference=True)
    assert f["C10"]==dict(dependency=True,interference=False)
    assert f["independent_reference"]=="U" and f["SLO"] is None and not f["P5_allowed"]

def test_source_lock_rejects_old_workspace_and_changed_byte(project):
    lock=project/m.ART/"source.json";lock.parent.mkdir(exist_ok=True)
    file=project/"old.py";file.write_text("old")
    lock.write_text(json.dumps({"files":[m.ref(project,"old.py")]}))
    with pytest.raises(ValueError):m.validate_source_lock(project,str(lock.relative_to(project)))
    file.write_text("changed")
    with pytest.raises(ValueError):m.validate_source_lock(project,str(lock.relative_to(project)))

@pytest.mark.parametrize("relative",["../escape","/root/out","a/../b","a\\b"])
def test_evidence_paths_are_bounded(project,relative):
    with pytest.raises(ValueError):m.safe_path(project,relative)

def test_symlink_evidence_rejected(project):
    link=project/"alias";link.symlink_to(project/m.ART,target_is_directory=True)
    with pytest.raises(ValueError):m.safe_path(project,"alias/x.json")

def test_no_gpu_or_launch_import_at_module_boundary():
    tree=ast.parse(SCRIPT.read_text())
    imports={n.name.split(".")[0] for v in ast.walk(tree) if isinstance(v,ast.Import) for n in v.names}
    imports|={v.module.split(".")[0] for v in ast.walk(tree) if isinstance(v,ast.ImportFrom) and v.module}
    assert not imports&{"torch","vllm","py_kvcache","subprocess","ctypes","cupy"}

def test_check_launch_denies_before_gpu_budget_and_append_only(project,monkeypatch):
    ledger=(project/m.LEDGER).read_bytes()
    p=project/m.OUT/"launch-denied.json"
    monkeypatch.setattr(m.shutil,"disk_usage",lambda _:SimpleNamespace(free=11*1024**3))
    assert m.main(["--project",str(project),"--output",str(p),"--check-launch"])==78
    receipt=json.loads(p.read_text())
    assert receipt["status"]=="BLOCKED_CPU_ONLY_NO_GPU_LAUNCH"
    assert receipt["gpu_initialized"] is False and receipt["commands_executed"]==0
    assert (project/m.LEDGER).read_bytes()==ledger
    with pytest.raises(ValueError):m.main(["--project",str(project),"--output",str(p)])
