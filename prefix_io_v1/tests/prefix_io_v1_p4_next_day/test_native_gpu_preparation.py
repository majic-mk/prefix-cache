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

def test_author_finder_rejects_unlocked_python_and_old_python(tmp_path):
    path=tmp_path/m.AUTHOR/"vllm";path.mkdir(parents=True)
    src=path/"__init__.py";src.write_text("VALUE=1")
    finder=m.BoundAuthorFinder(tmp_path,{})
    with pytest.raises(ValueError):finder.find_spec("vllm")
    rel=src.relative_to(tmp_path).as_posix()
    refs={rel:dict(path=rel,bytes=src.stat().st_size,sha256=m.sha_file(src))}
    spec=finder.__class__(tmp_path,refs).find_spec("vllm")
    assert isinstance(spec.loader,m.LockedSourceLoader)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.VALUE==1
    old=tmp_path/m.BINARY/"old.py";old.parent.mkdir(parents=True);old.write_text("VALUE=2")
    with pytest.raises(ValueError):finder.__class__(tmp_path,refs).find_spec("vllm.old",[str(old.parent)])
    src.write_text("VALUE=99")
    with pytest.raises(ValueError):spec.loader.exec_module(module)

def test_author_finder_missing_module_cannot_fall_through_to_other_finder(tmp_path):
    with pytest.raises(ModuleNotFoundError):m.BoundAuthorFinder(tmp_path,{}).find_spec("vllm.missing",[])

def test_launch_missing_scope_rejects_before_guard_or_preflight_or_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"invoke_existing_guard",lambda *a:(_ for _ in ()).throw(AssertionError("guard invoked")))
    monkeypatch.setattr(m,"qualify",lambda *a:(_ for _ in ()).throw(AssertionError("GPU invoked")))
    before=set(sys.modules)
    assert m.main(["--project",str(tmp_path),"--mode","off","--name","safe","--launch",
                   "--source-lock","not-yet-frozen.json"])==78
    assert not (set(sys.modules)-before)&{"torch","vllm","py_kvcache"}
    assert not (tmp_path/"experiments/prefix_io_v1/gpu-budget-ledger.json").exists()

@pytest.mark.parametrize("reason",["invalid scope","invalid source","invalid context","storage","budget"])
def test_failed_pure_launch_gate_never_invokes_existing_guard(tmp_path,monkeypatch,reason):
    monkeypatch.setattr(m,"launch_gates",lambda *a:(_ for _ in ()).throw(ValueError(reason)))
    monkeypatch.setattr(m,"invoke_existing_guard",lambda *a:(_ for _ in ()).throw(AssertionError("guard invoked")))
    assert m.main(["--project",str(tmp_path),"--mode","shadow","--name","safe","--launch"])==78

def test_preview_shell_uses_launch_gate_and_custom_scope(tmp_path):
    p=m.preview(tmp_path,"off","safe",scope_record="custom-scope.json")
    assert "--launch" in p["shell_preview"] and "run_gpu_stage.py" not in p["shell_preview"]
    assert p["launch_argv"][p["launch_argv"].index("--scope-record")+1]=="custom-scope.json"
    assert p["internal_guard_argv_requires_prior_launch_gates"]

@pytest.fixture
def scope_project(tmp_path):
    from prepare_p4_gpu_next_day import required_source_paths,ref
    root=tmp_path;primary=root/"experiments/prefix_io_v1/runs";primary.mkdir(parents=True)
    perm=root/"experiments/prefix_io_v1/configs/permissions.yaml";perm.parent.mkdir(parents=True)
    uuid="GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8"
    perm.write_text("allow_gpu_runs: true\nmax_gpu_hours: 8\napproved_gpu_ids:\n- "+uuid+
        "\napproved_experiment_root: "+str(primary)+"\n")
    ledger=root/"experiments/prefix_io_v1/gpu-budget-ledger.json"
    ledger.write_text(json.dumps(dict(gpu_wall_seconds=100,active_reservation=None)))
    control=root/m.CONTROL/"prefix_io_control/p4_fixture.py"
    control.parent.mkdir(parents=True);control.write_text("# CPU source fixture\n")
    names=required_source_paths(root)
    for rel in names:
        p=root/rel
        if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_text("# locked CPU fixture\n")
    lock=root/"lock.json";lock.write_text(json.dumps(dict(files=[ref(root,rel) for rel in sorted(names)])))
    scope=dict(schema_version=1,status="USER_AUTHORIZED_P4_GPU_NATIVE_QUALIFICATION",
        allow_gpu_initialization=True,allow_gpu_runs=True,allowed_modes=["off","shadow"],
        new_executor=False,allow_model_downloads=False,gpu_uuid=uuid,
        base_permissions=ref(root,"experiments/prefix_io_v1/configs/permissions.yaml"),
        source_lock="lock.json",source_lock_sha256=m.sha_file(lock),qualification_context=m.qualification_context())
    scopefile=root/m.GPU_SCOPE;scopefile.parent.mkdir(parents=True);scopefile.write_text(json.dumps(scope))
    return root,scopefile,ledger,perm,control

def scope_args():
    from types import SimpleNamespace
    return SimpleNamespace(mode="off",name="safe",scope_record=m.GPU_SCOPE,source_lock="lock.json")

def test_pure_scope_source_context_storage_gate_with_no_actual_guard(scope_project):
    from types import SimpleNamespace
    root,*_=scope_project
    gates,reason=m.launch_gates(root,scope_args(),lambda _:SimpleNamespace(free=m.FLOOR+m.STORAGE_RESERVE))
    assert reason=="READY_FOR_AUTHORIZED_OLD_GUARD" and gates["scope"]["qualification_context"]==m.qualification_context()
    assert not (gates["out"]).exists()

@pytest.mark.parametrize("field,value",[("allow_gpu_runs",False),("allow_gpu_initialization",0),
    ("gpu_uuid","GPU-00000000-0000-0000-0000-000000000000"),("qualification_context",{}),
    ("allowed_modes",["joint"]),("source_lock_sha256","0"*64)])
def test_scope_source_context_counterexamples_rejected_before_guard(scope_project,field,value):
    root,scopefile,*_=scope_project
    obj=json.loads(scopefile.read_text());obj[field]=value;scopefile.write_text(json.dumps(obj))
    with pytest.raises(ValueError):m.launch_gates(root,scope_args())

@pytest.mark.parametrize("change",["source_bytes","new_source","permissions","budget","active","storage","existing_output"])
def test_pure_actual_budget_storage_and_source_counterexamples(scope_project,change):
    from types import SimpleNamespace
    root,scopefile,ledger,perm,control=scope_project
    free=m.FLOOR+m.STORAGE_RESERVE
    if change=="source_bytes":control.write_text("changed")
    elif change=="new_source":(control.parent/"p4_added_after_freeze.py").write_text("changed")
    elif change=="permissions":perm.write_text(perm.read_text()+"# changed\n")
    elif change=="budget":ledger.write_text(json.dumps(dict(gpu_wall_seconds=28800-199,active_reservation=None)))
    elif change=="active":ledger.write_text(json.dumps(dict(gpu_wall_seconds=100,active_reservation={"busy":True})))
    elif change=="storage":free-=1
    elif change=="existing_output":(root/"experiments/prefix_io_v1/runs/safe/details").mkdir(parents=True)
    with pytest.raises(ValueError):m.launch_gates(root,scope_args(),lambda _:SimpleNamespace(free=free))

def fake_locked_binary(root,name="vllm.vllm_flash_attn._vllm_fa2_C",suffix=".abi3.so"):
    rel=m.BINARY+"/"+"/".join(name.split(".")[1:])+suffix
    p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"CPU fake extension; never execute")
    return p,{rel:dict(path=rel,bytes=p.stat().st_size,sha256=m.sha_file(p))}

def test_exact_nested_extension_fallback_constructs_spec_without_binary_execution(tmp_path):
    name="vllm.vllm_flash_attn._vllm_fa2_C";old,refs=fake_locked_binary(tmp_path,name)
    nested=tmp_path/m.AUTHOR/"vllm/vllm_flash_attn";nested.mkdir(parents=True)
    spec=m.BoundAuthorFinder(tmp_path,refs).find_spec(name,[str(nested)])
    assert Path(spec.origin)==old
    assert isinstance(spec.loader,m.importlib.machinery.ExtensionFileLoader)
    # Do not call module_from_spec or exec_module: this is CPU path eligibility only.

def test_unknown_nested_binary_never_scans_old_python_or_unlocked_so(tmp_path,monkeypatch):
    name="vllm.vllm_flash_attn._vllm_fa2_C";old,_=fake_locked_binary(tmp_path,name)
    nested=tmp_path/m.AUTHOR/"vllm/vllm_flash_attn";nested.mkdir(parents=True)
    (old.parent/"_vllm_fa2_C.py").write_text("raise AssertionError('old source executed')")
    monkeypatch.setattr(m,"sha_file",lambda *a:(_ for _ in ()).throw(AssertionError("unknown binary read")))
    with pytest.raises(ModuleNotFoundError):m.BoundAuthorFinder(tmp_path,{}).find_spec(name,[str(nested)])

def test_ambiguous_locked_extension_suffixes_are_rejected(tmp_path):
    name="vllm.vllm_flash_attn._vllm_fa2_C"
    suffixes=m.importlib.machinery.EXTENSION_SUFFIXES
    assert len(suffixes)>=2
    _,one=fake_locked_binary(tmp_path,name,suffixes[0]);_,two=fake_locked_binary(tmp_path,name,suffixes[1])
    with pytest.raises(ValueError,match="ambiguous"):m.BoundAuthorFinder(tmp_path,{**one,**two}).find_spec(name,[])

@pytest.mark.parametrize("change",["bytes","hash","symlink","module_name"])
def test_exact_binary_fallback_rejects_metadata_and_path_drift(tmp_path,change):
    name="vllm.vllm_flash_attn._vllm_fa2_C";old,refs=fake_locked_binary(tmp_path,name)
    row=next(iter(refs.values()))
    if change=="bytes":row["bytes"]+=1
    elif change=="hash":old.write_bytes(b"CPU fake extension; never executE")
    elif change=="symlink":
        other=tmp_path/"other.so";other.write_bytes(old.read_bytes());old.unlink();old.symlink_to(other)
    else:name="vllm.vllm_flash_attn.bad-name"
    with pytest.raises(ValueError):m.BoundAuthorFinder(tmp_path,refs).find_spec(name,[])

def test_source_selected_before_exact_locked_binary_and_old_python_never_used(tmp_path):
    name="vllm.vllm_flash_attn._vllm_fa2_C";old,refs=fake_locked_binary(tmp_path,name)
    nested=tmp_path/m.AUTHOR/"vllm/vllm_flash_attn";nested.mkdir(parents=True)
    src=nested/"_vllm_fa2_C.py";src.write_text("VALUE='new-source'")
    rel=src.relative_to(tmp_path).as_posix()
    refs[rel]=dict(path=rel,bytes=src.stat().st_size,sha256=m.sha_file(src))
    spec=m.BoundAuthorFinder(tmp_path,refs).find_spec(name,[str(nested)])
    assert Path(spec.origin)==src and isinstance(spec.loader,m.LockedSourceLoader)
