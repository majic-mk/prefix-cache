"""Pure CPU guards for isolated patch reconstruction; no native/GPU imports."""
import copy
import datetime as dt
import json
from pathlib import Path
import pytest
import qualify_patch_roundtrip_p316 as q

@pytest.mark.parametrize("name",["../a.py","a/../b.py","/a.py","a//b.py",
    "a/./b.py",".git/config.py","a/__pycache__/b.py","a\\b.py","a\x00b.py","a\nb.py"])
def test_source_names_cannot_escape_or_use_generated_files(name):
    with pytest.raises(ValueError):q.relative_name(name)

def test_regular_source_rejects_symlink_and_payload_type(tmp_path):
    root=tmp_path/"root";root.mkdir()
    outside=tmp_path/"outside.py";outside.write_text("value=1\n")
    (root/"link.py").symlink_to(outside)
    with pytest.raises(ValueError,match="symlink/path"):q.source_path(root,"link.py")
    (root/"payload.bin").write_text("CPU fixture text")
    with pytest.raises(ValueError,match="text source"):q.source_path(root,"payload.bin")

def test_output_is_new_subdirectory_and_never_reuses_existing_tree(tmp_path):
    root=tmp_path/"root";allowed=root/q.OUT_REL;allowed.mkdir(parents=True)
    path=allowed/"roundtrip-01"
    assert q.output_path(root,path)==path
    path.mkdir()
    with pytest.raises(ValueError,match="already exists"):q.output_path(root,path)
    with pytest.raises(ValueError,match="new exact subdirectory"):q.output_path(root,tmp_path/"other")
    with pytest.raises(ValueError,match="new exact subdirectory"):q.output_path(root,allowed)

def lock_map(prefix):
    return {prefix+"/py_kvcache/source"+str(i)+".py":"a"*64 for i in range(31)}

def test_native_seed_requires_exact_31_matching_locked_sources():
    lock=lock_map(q.P313_NATIVE)
    rows=q.expected_native(lock,q.P313_NATIVE)
    seed=dict(author_head="3abba7a502d553f6e7e2e58b92086487e3395d7e",
              files={k:dict(sha256=v,bytes=1) for k,v in rows.items()})
    assert q.seed_manifest(seed,rows)==rows
    bad=copy.deepcopy(seed);bad["files"][next(iter(rows))]["sha256"]="b"*64
    with pytest.raises(ValueError,match="exact locked P313"):q.seed_manifest(bad,rows)
    bad=copy.deepcopy(seed);bad["files"][next(iter(rows))]["bytes"]=True
    with pytest.raises(ValueError,match="invalid seed"):q.seed_manifest(bad,rows)
    with pytest.raises(ValueError,match="exact 31"):q.expected_native(dict(list(lock.items())[:30]),q.P313_NATIVE)
    bad=copy.deepcopy(lock);bad[next(iter(bad))]="invalid"
    with pytest.raises(ValueError,match="invalid native"):q.expected_native(bad,q.P313_NATIVE)

@pytest.mark.parametrize("label",["common315","increment316","research315"])
def test_patch_target_parser_matches_declared_roots(label):
    parts=[]
    for name in sorted(q.approved_targets(label)):
        old="/dev/null" if name in q.NEW_POLICY else "a/"+name
        parts.append("--- "+old+"\n+++ b/"+name+"\n@@ -0,0 +1 @@\n+value=1\n")
    assert q.patch_targets("".join(parts))==q.approved_targets(label)

@pytest.mark.parametrize("text",[
    "--- a/../escape.py\n+++ b/../escape.py\n",
    "--- a/.git/config.py\n+++ b/.git/config.py\n",
    "--- a/file.py\n+++ b/other.py\n",
    "--- a/file.py\n",
    "+++ b/file.py\n",
    "--- /dev/null\n+++ /dev/null\n",
    "--- a/file.py\n+++ b/file.py\n--- a/file.py\n+++ b/file.py\n"])
def test_patch_headers_reject_rename_escape_and_ambiguous_pairs(text):
    with pytest.raises(ValueError):q.patch_targets(text)

def test_git_commands_use_isolated_worktree_and_explicit_increment_directory(tmp_path):
    scratch=tmp_path/"scratch";patch=tmp_path/"increment.patch"
    argv=q.patch_command(scratch,patch,directory=q.NATIVE,check=True,reverse=True)
    assert argv[argv.index("-C")+1]==str(scratch)
    assert "--directory="+q.NATIVE in argv and "--check" in argv and "--reverse" in argv
    assert "--unsafe-paths" not in argv and argv[-1]==str(patch.resolve())

@pytest.mark.parametrize("change",["active","scope","extra","stale","future"])
def test_idle_receipt_must_be_fresh_root_declaration_bound_to_lock(tmp_path,change):
    root=tmp_path/"root";out=root/q.OUT_REL;out.mkdir(parents=True)
    value=dict(scope="P316_CPU_PATCH_ROUNDTRIP_GPU_IDLE",
               created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               active_gpu_operation=None,execution_lock_sha256="a"*64)
    if change=="active":value["active_gpu_operation"]="model-active"
    if change=="scope":value["scope"]="different"
    if change=="extra":value["permission"]=True
    if change in ("stale","future"):
        seconds=-3600 if change=="stale" else 3600
        value["created_utc"]=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=seconds)).isoformat()
    path=out/"idle.json";path.write_text(json.dumps(value))
    with pytest.raises(ValueError):q.idle_receipt(root,path,"a"*64)

def test_idle_receipt_accepts_only_matching_execution_lock(tmp_path):
    root=tmp_path/"root";out=root/q.OUT_REL;out.mkdir(parents=True)
    value=dict(scope="P316_CPU_PATCH_ROUNDTRIP_GPU_IDLE",
               created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               active_gpu_operation=None,execution_lock_sha256="a"*64)
    path=out/"idle.json";path.write_text(json.dumps(value))
    assert q.idle_receipt(root,path,"a"*64)==value
    with pytest.raises(ValueError,match="lock evidence"):q.idle_receipt(root,path,"b"*64)

def complete_component_record(mode):
    return dict(status="PASS_CPU_COMPOSITION_ONLY",mode=mode,
        cuda_initialized=False,cuda_initialization_attempts=[],research_import_attempts=[],
        real_gpu_qualification=False,native_constructor_resources_allocated=False,
        native_backend_or_model_executed=False,physical_completion_proved=False,
        default_off=dict(
            constructor_defaults={name:dict(dispatch_controller=None,max_accepted_parents=None)
                                  for name in ("IoReactor","TransferCoordinator")},
            actual_hook_controller_none=True,policy_clock_calls=0,
            research_off_factory_controller_none=True if mode=="research" else None))

def complete_record():
    target=q.expected_native(lock_map(q.P316_NATIVE),q.P316_NATIVE)
    stages=["seed-p313","common315","common316","full316-research315",
            "reverse-research-common316","reverse-increment-common315","reverse-common-seed-p313"]
    report=dict(status="PASS_CPU_PATCH_ROUNDTRIP_COMPOSITIONS",
        commands=[dict(label=label+"-"+direction+"-"+kind,returncode=0)
                  for label in ("common315","increment316","research315")
                  for direction in ("forward","reverse") for kind in ("check","apply")],
        stages=[dict(name=s,native=dict(sha256_by_relative_path=dict(target),files=31)) for s in stages],
        common_cpu_smoke=complete_component_record("common"),
        research_cpu_smoke=complete_component_record("research"),
        source_preservation=dict(changed=[],unverified=[]))
    return report,target

def test_final_machine_pass_is_derived_from_all_completed_evidence():
    report,target=complete_record()
    q.final_machine_fields(report,target)
    assert report["status"]=="PASS_PATCH_ROUNDTRIP"
    assert report["check_exit"]==report["apply_exit"]==0 and report["exact_content"] is True
    assert len(report["files"])==31
    assert all(set(r)=={"path","expected_sha256","actual_sha256"} and
               r["expected_sha256"]==r["actual_sha256"] for r in report["files"])
    assert set(report["individual_checks"])=={"common_only","full","default_off","reverse"}
    assert all(row["passed"] for row in report["individual_checks"].values())

@pytest.mark.parametrize("failure",[
    "check_nonzero","missing_apply","timeout","missing_stage","full_digest",
    "source_changed","source_unverified","common_failure","research_failure",
    "off_clock","off_factory","cuda_attempt","physical_claim","storage","earlier_error"])
def test_partial_or_failed_proof_never_creates_pass_receipt(failure):
    report,target=complete_record()
    if failure=="check_nonzero":report["commands"][0]["returncode"]=1
    elif failure=="missing_apply":report["commands"].pop()
    elif failure=="timeout":
        report["commands"][1]["returncode"]=None;report["commands"][1]["error"]="TimeoutExpired"
    elif failure=="missing_stage":report["stages"].pop()
    elif failure=="full_digest":
        full=next(r for r in report["stages"] if r["name"]=="full316-research315")
        full["native"]["sha256_by_relative_path"][next(iter(target))]="b"*64
    elif failure=="source_changed":report["source_preservation"]["changed"]=["source.py"]
    elif failure=="source_unverified":report["source_preservation"]["unverified"]=[dict(path="source.py",error="missing")]
    elif failure=="common_failure":report["common_cpu_smoke"]["status"]="FAILED"
    elif failure=="research_failure":report["research_cpu_smoke"]["status"]="FAILED"
    elif failure=="off_clock":report["common_cpu_smoke"]["default_off"]["policy_clock_calls"]=1
    elif failure=="off_factory":report["research_cpu_smoke"]["default_off"]["research_off_factory_controller_none"]=False
    elif failure=="cuda_attempt":report["common_cpu_smoke"]["cuda_initialization_attempts"]=["forbidden"]
    elif failure=="physical_claim":report["research_cpu_smoke"]["physical_completion_proved"]=True
    elif failure=="storage":report["storage_error"]="below free floor"
    else:report["status"]="FAILED_CPU_PATCH_ROUNDTRIP"
    q.final_machine_fields(report,target)
    assert report["status"]=="FAILED_CPU_PATCH_ROUNDTRIP"
