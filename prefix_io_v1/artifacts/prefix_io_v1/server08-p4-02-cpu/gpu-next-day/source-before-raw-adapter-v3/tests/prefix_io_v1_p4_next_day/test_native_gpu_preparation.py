import ast, importlib.util, json, sys, hashlib, os
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py"
spec=importlib.util.spec_from_file_location("p4_native_preparation",SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize("mode",["off","shadow"])
def test_dry_run_bound_real_guard_cli_and_workspaces(tmp_path,mode):
    p=m.preview(tmp_path,mode,"server08-p4-02-"+mode)
    assert p["guard_argv"][:6]==[".venv/bin/python","experiments/prefix_io_v1/scripts/run_gpu_stage.py",
            "--label","server08-p4-02-"+mode,"--seconds","180"]
    assert p["new_storage_reserve_bytes"]==128*1024**2
    assert "py-kvcache-p4-02-cpu" in p["environment_preview"]["PYTHONPATH"]
    assert "prefix-io-p4-02-cpu/src" in p["environment_preview"]["PYTHONPATH"]
    assert "vllm-author-p4-02-cpu" in p["environment_preview"]["PYTHONPATH"]
    assert p["source_lock_pending"] and p["binary_compatibility_GPU_unverified"]
    assert p["actual_GPU_cases_run"]==0 and not p["effect_verified"]

@pytest.mark.parametrize("name",["../escape","","a b","a/b","--flag","a;rm"])
def test_run_name_rejects_escape_and_commands(tmp_path,name):
    with pytest.raises(ValueError):m.preview(tmp_path,"off",name)

@pytest.mark.parametrize("mode",["fixed","dependency_only","interference","joint"])
def test_g1_scope_does_not_smuggle_advanced_modes(tmp_path,mode):
    with pytest.raises(ValueError):m.preview(tmp_path,mode,"safe")

def test_check_launch_never_calls_execution_or_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"execution_gates",lambda *a:(_ for _ in ()).throw(AssertionError("scope read")))
    monkeypatch.setattr(m,"qualify",lambda *a:(_ for _ in ()).throw(AssertionError("GPU invoked")))
    before=set(sys.modules)
    receipt=tmp_path/m.ART/"gpu-next-day/denial.json"
    assert m.main(["--project",str(tmp_path),"--mode","off","--name","safe",
                   "--check-launch","--receipt",str(receipt)])==78
    r=json.loads(receipt.read_text())
    assert not r["gpu_initialized"] and r["new_gpu_runs"]==r["budget_reserved_seconds"]==0
    assert not (set(sys.modules)-before)&{"torch","vllm","py_kvcache"}

def test_execute_missing_explicit_scope_fails_before_backend(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"qualify",lambda *a:(_ for _ in ()).throw(AssertionError("GPU invoked")))
    assert m.main(["--project",str(tmp_path),"--mode","shadow","--name","safe","--execute"])==78
    assert not (tmp_path/"experiments/prefix_io_v1/runs").exists()

def test_default_does_not_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"execution_gates",lambda *a:(_ for _ in ()).throw(AssertionError("actual execution")))
    assert m.main(["--project",str(tmp_path),"--mode","off","--name","safe"])==78

def test_cpu_receipts_append_new_and_bounded(tmp_path):
    p=tmp_path/m.ART/"gpu-next-day/test.json"
    args=["--project",str(tmp_path),"--mode","off","--name","safe","--dry-run","--receipt",str(p)]
    assert m.main(args)==0
    with pytest.raises(FileExistsError):m.main(args)
    with pytest.raises(ValueError):
        m.main(["--project",str(tmp_path),"--mode","off","--name","safe","--dry-run",
                "--receipt",str(tmp_path/"escape.json")])

def test_top_level_imports_do_not_initialize_gpu():
    tree=ast.parse(SCRIPT.read_text())
    imports={n.name.split(".")[0] for v in tree.body if isinstance(v,ast.Import) for n in v.names}
    imports|={v.module.split(".")[0] for v in tree.body if isinstance(v,ast.ImportFrom) and v.module}
    assert not imports&{"torch","vllm","py_kvcache","prefix_io_control","experiment_storage","subprocess"}

def test_source_refs_require_new_author_and_locked_binary(tmp_path):
    lock=tmp_path/"lock.json";lock.write_text('{"files":[]}')
    with pytest.raises(ValueError):m.source_refs(tmp_path,"lock.json")

def test_uuid_normalization():
    import uuid
    s="83ac724c-6d18-6f08-ab96-fb871fb1f4a8"
    assert m.normalize_uuid("GPU-"+s)==m.normalize_uuid(uuid.UUID(s).bytes)==s
