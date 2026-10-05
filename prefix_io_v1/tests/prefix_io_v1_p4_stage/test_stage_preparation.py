import hashlib, importlib.util, json, sys
from pathlib import Path
import pytest
SCRIPT = Path(__file__).resolve().parents[2] / "experiments/prefix_io_v1/scripts/prepare_p4_gpu_stage.py"
spec = importlib.util.spec_from_file_location("p4_gpu_preparation", SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@pytest.fixture
def project(tmp_path):
    root = tmp_path
    base = root / "experiments/prefix_io_v1/configs/permissions.yaml"
    base.parent.mkdir(parents=True); base.write_text("allow_gpu_runs: true\n")
    auth = {"schema_version": 1, "status": "USER_AUTHORIZED_P4_CPU_ONLY",
            "allow_project_local_edits": True, "allow_cpu_tests": True,
            "allow_gpu_initialization": False, "allow_gpu_runs": False,
            "allow_model_downloads": False, "allow_driver_or_system_changes": False,
            "base_permissions": m.file_ref(root, str(base.relative_to(root)))}
    out = root / m.P4_ARTIFACT; out.mkdir(parents=True)
    (root / m.AUTH).write_text(json.dumps(auth))
    (root / m.LEDGER).write_text('{"consumed_seconds":123,"active":null}')
    p = root / m.PRODUCTION; p.parent.mkdir(parents=True); p.write_text('{"production_value":null}')
    return root

def test_general_gpu_true_cannot_override_current_cpu_scope(project):
    before = (project / m.LEDGER).read_bytes()
    out = project / m.P4_ARTIFACT / "denied.json"
    assert m.main(["--project", str(project), "--output", str(out), "--check-launch"]) == 78
    assert (project / m.LEDGER).read_bytes() == before
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "BLOCKED_CPU_ONLY_NO_GPU_LAUNCH"
    assert receipt["gpu_initialized"] is False and receipt["new_gpu_runs"] == 0
    assert receipt["budget_reserved_seconds"] == 0
    assert receipt["factorial"]["independent_main_reference"] == "U (strongest P3 baseline)"

@pytest.mark.parametrize("key,value", [("allow_gpu_runs", True), ("allow_gpu_runs", 0),
    ("allow_gpu_initialization", 0), ("allow_cpu_tests", 1), ("schema_version", True)])
def test_explicit_authorization_types_are_required(project, key, value):
    p=project/m.AUTH; a=json.loads(p.read_text());a[key]=value;p.write_text(json.dumps(a))
    with pytest.raises(ValueError):m.plan(project)

def test_changed_permissions_rejected_before_launch(project):
    p=project/"experiments/prefix_io_v1/configs/permissions.yaml";p.write_text("changed")
    with pytest.raises(ValueError):m.plan(project)

def test_null_cost_and_slo_do_not_become_zero_qualification(project):
    p=m.plan(project)
    assert p["slo"] is None
    assert p["P3_qualified_table_is_P4_production_table"] is False
    assert p["positive_effect_established"] is False
    assert all(g["status"].startswith("BLOCKED") for g in p["next_gpu_gates"])

def test_output_cannot_escape_artifact_scope(project):
    with pytest.raises(ValueError):
        m.main(["--project",str(project),"--output",str(project/"unsafe.json")])
    assert not (project/"unsafe.json").exists()

def test_receipts_are_append_only(project):
    out=project/m.P4_ARTIFACT/"result.json"
    assert m.main(["--project",str(project),"--output",str(out)]) == 0
    raw=out.read_bytes()
    with pytest.raises(ValueError):m.main(["--project",str(project),"--output",str(out)])
    assert out.read_bytes()==raw

def test_preparation_has_no_gpu_or_process_dependency():
    import ast
    tree=ast.parse(SCRIPT.read_text())
    imports={n.name.split(".")[0] for v in ast.walk(tree) if isinstance(v,ast.Import) for n in v.names}
    imports |= {v.module.split(".")[0] for v in ast.walk(tree) if isinstance(v,ast.ImportFrom) and v.module}
    assert not imports & {"torch","py_kvcache","vllm","cupy","subprocess"}

def test_factorial_labels_match_original_03_contract(project):
    f=m.plan(project)["factorial"]
    assert f["C01"] == "same caps; dependency off; interference on"
    assert f["C10"] == "same caps; dependency on; interference off"
