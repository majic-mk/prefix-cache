"""CPU-only, append-only P4-02 source patches with byte-exact reverse proof."""
from pathlib import Path
import sys, os, json, hashlib, subprocess, difflib, io, tarfile
root=Path.cwd()
sys.path.insert(0,str(root/"experiments/prefix_io_v1/scripts"))
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-02-cpu/src"))
sys.path.extend(str(p) for p in (root/".venv/lib").glob("python*/site-packages"))
from experiment_storage import preflight
out=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-integration/patch-roundtrip-04"
scratch_root=root/"experiments/prefix_io_v1/runs/server08-p4-02-patch-roundtrip-04/cpu-evidence"
preflight(scratch_root,128*1024*1024)
out.mkdir(exist_ok=False);scratch_root.mkdir(parents=True,exist_ok=False)
head="817a7e3124f817cd6e549581d3e5483207a753a4"
build=root/"third_party/work/vllm-author-build"
assert subprocess.check_output(["git","-C",str(build),"rev-parse","HEAD"],text=True).strip()==head
ledger=root/"experiments/prefix_io_v1/gpu-budget-ledger.json";ledger_before=ledger.read_bytes()
assert json.loads(ledger_before)["active_reservation"] is None
def digest(raw):return hashlib.sha256(raw).hexdigest()
def collect(directory,prefix,package=None):
    base=directory if package is None else directory/package
    values={}
    for path in base.rglob("*.py"):
        if any(x in path.parts for x in (".git","__pycache__",".pytest_cache")):continue
        assert not path.is_symlink(),"source links require independent lock"
        key=prefix+"/"+path.relative_to(directory).as_posix()
        values[key]=path.read_bytes()
    return values
dirs={
"control_01":root/"third_party/work/prefix-io-p4-01-cpu/src/prefix_io_control",
"control_02":root/"third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control",
"control_p3":root/"src/prefix_io_control",
"native_01":root/"third_party/work/py-kvcache-p4-01-cpu",
"native_02":root/"third_party/work/py-kvcache-p4-02-cpu",
"native_p3":root/"third_party/work/py-kvcache-p3-16-cpu",
"author_old":build,
"author_02":root/"third_party/work/vllm-author-p4-02-cpu"}
trees={}
for key,directory in dirs.items():
    trees[key]=collect(directory,"control" if key.startswith("control") else "native" if key.startswith("native") else "author",
                       package="vllm" if key.startswith("author") else None)
raw=subprocess.check_output(["git","-C",str(build),"archive",head,"vllm/"])
tracked={}
with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as archive:
    for member in archive.getmembers():
        if member.name.endswith(".py"):
            assert member.isfile(),"tracked Python symlinks unsupported"
            tracked["author/"+member.name]=archive.extractfile(member).read()
assert set(tracked)<=set(trees["author_old"])
assert set(tracked)<=set(trees["author_02"])
supplement=set(trees["author_old"])-set(tracked)
assert len(supplement)==92
assert all(trees["author_old"][p]==trees["author_02"][p] for p in supplement)
author_inherited={p for p in tracked if tracked[p]!=trees["author_old"][p]}
assert author_inherited=={"author/vllm/platforms/cuda.py","author/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"}
author_metadata={p for p in trees["author_old"] if trees["author_old"][p]!=trees["author_02"][p]}
assert author_metadata=={"author/vllm/distributed/kv_transfer/kv_connector/v1/offloading/"+n+".py" for n in ("common","scheduler","worker")}
final={**trees["control_02"],**trees["native_02"],**trees["author_02"]}
inc={**trees["control_01"],**trees["native_01"],**trees["author_old"]}
full={**trees["control_p3"],**trees["native_p3"],**tracked}
research_names={"control/"+n+".py" for n in ("p4_bridge","p4_policy","p4_cost_table","p4_types","p4_eta")}
evidence_names={"control/"+n+".py" for n in ("p4_production_table_contract",
 "p4_paired_measurement_verifier","p4_verified_cost_loader","p4_raw_pair_recorder")}
changed_inc={p for p in final if inc.get(p)!=final[p]}
changed_full={p for p in final if full.get(p)!=final[p]}
assert set(inc)<=set(final) and set(full)<=set(final)
def write_tree(directory,values):
    for name,raw in values.items():
        p=directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
def read_tree(directory):
    return {p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob("*.py")}
def refs(values):
    return [{"path":k,"bytes":len(v),"sha256":digest(v)} for k,v in sorted(values.items())]
def patch(state,target,names):
    output=[]
    for name in sorted(names):
        if name not in state and target[name] == b"":
            output.append(chr(10).join(["diff --git a/"+name+" b/"+name,"new file mode 100644","index 0000000..e69de29",""]))
            continue
        output.append(chr(10).join(["diff --git a/"+name+" b/"+name,""]))
        if name not in state:output.append("new file mode 100644"+chr(10))
        a=state.get(name,b"").decode().splitlines(keepends=True)
        b=target[name].decode().splitlines(keepends=True)
        output.extend(difflib.unified_diff(a,b,fromfile="a/"+name if name in state else "/dev/null",tofile="b/"+name))
    return "".join(output)
commands=[]
def git_apply(directory,path,reverse=False,check=False):
    cmd=["git","apply"]+(["--reverse"] if reverse else [])+(["--check"] if check else [])+[str(path)]
    result=subprocess.run(cmd,cwd=directory,capture_output=True,text=True)
    commands.append(dict(command=cmd,cwd=str(directory),exit=result.returncode,stdout=result.stdout,stderr=result.stderr))
    (out/"commands.json").write_text(json.dumps(commands,indent=2))
    assert result.returncode==0,commands[-1]
def off_probe(directory,name):
    alias=scratch_root/(name+"-common-off")/"prefix_io_control";alias.mkdir(parents=True)
    for path in (directory/"control").glob("*.py"):(alias/path.name).write_bytes(path.read_bytes())
    code="""
import sys,importlib.abc
sys.path.insert(0,sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,n,path=None,target=None):
  if n.split('.')[0] in ('torch','vllm','py_kvcache','numpy','cupy','cuda') or n in (
  'prefix_io_control.p4_policy','prefix_io_control.p4_bridge','prefix_io_control.p4_eta',
  'prefix_io_control.p4_verified_cost_loader','prefix_io_control.p4_startup_evidence'):
   raise AssertionError('off imported optional strategy/backend '+n)
sys.meta_path.insert(0,Guard())
from prefix_io_control.p4_options import parse_p4_options,build_p4_kwargs
assert 'p4_bridge' not in build_p4_kwargs(parse_p4_options({'prefix_io_p4_policy':{'mode':'off'}}))
print('PASS glue-only off without strategy, evidence loader or backend')
"""
    cmd=[str(root/".venv/bin/python"),"-I","-S","-c",code,str(alias.parent)]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
    commands.append(dict(command=cmd,cwd=str(root),exit=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,commands[-1]
    return result.stdout

results=[]
for label,base in [("p4-01-increment",inc),("p3-author-head-full",full)]:
    scratch=scratch_root/label;scratch.mkdir()
    directory=out/label;directory.mkdir()
    write_tree(scratch,base);state=dict(base);groups=[]
    if label=="p3-author-head-full":
        groups.extend([
          ("0001-inherited-common-cuda-uuid",{"author/vllm/platforms/cuda.py"},trees["author_old"],"inherited common compatibility fix; no GPU test here"),
          ("0002-inherited-flush-observation",{"author/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"},trees["author_old"],"inherited opt-in diagnostic flush probe; no scheduling change"),
        ])
    changed={p for p in final if state.get(p)!=final[p]}
    # Names depend on pre-existing first two steps, so explicitly exclude supplement.
    glue=changed-research_names-evidence_names-supplement
    if label=="p3-author-head-full":glue.discard("author/vllm/platforms/cuda.py")
    # The inherited scheduler file is advanced to old author bytes first, then
    # metadata glue advances it to new fork bytes.
    groups.extend([
      ("0003-p4-observation-native-glue",glue,final,"inert off adapters and scheduled work values; no actual GPU load capability"),
      ("0004-bounded-research-values",changed & research_names,final,"bounded policy/value contracts and causal diagnostic ETA; production gates closed"),
      ("0005-measurement-preparation",changed & evidence_names,final,"CPU raw/semantic evidence preparation, not real GPU recorder/qualification"),
    ])
    if label=="p3-author-head-full":
        groups.append(("0006-locked-runtime-python-supplement",supplement,final,"92 existing build Python files copied with actual bytes provenance; neither research nor driver/system change"))
    reports=[]
    off=None
    for group,names,target,scope in groups:
        names={p for p in names if state.get(p)!=target[p]}
        if not names:continue
        p=directory/(group+".patch")
        p.write_text(patch(state,target,names),encoding="utf-8")
        git_apply(scratch,p,check=True);git_apply(scratch,p)
        for name in names:state[name]=target[name]
        assert read_tree(scratch)==state,"each forward step must match actual source bytes"
        reports.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p.read_bytes()),inputs=sorted(names),scope=scope))
        if group=="0003-p4-observation-native-glue":
            off=off_probe(scratch,label)
    assert state==final and read_tree(scratch)==final
    for record in reversed(reports):
        p=root/record["path"]
        git_apply(scratch,p,reverse=True,check=True);git_apply(scratch,p,reverse=True)
    assert read_tree(scratch)==base,"reverse apply must match every original byte and path"
    result=dict(base_name=label,status="PASS_FORWARD_REVERSE_BYTE_ROUNDTRIP",
                base_files=len(base),final_files=len(final),changed_files=sum(base.get(p)!=v for p,v in final.items()),
                patches=reports,base_refs=refs(base),final_refs=refs(final),glue_only_off=off,
                no_source_paths_removed=True)
    (directory/"result.json").write_text(json.dumps(result,indent=2))
    results.append({k:v for k,v in result.items() if k not in ("base_refs","final_refs")})

# Re-read every live project input; nothing was applied to original worktrees.
source_unchanged=all(collect(dirs[k],"control" if k.startswith("control") else "native" if k.startswith("native") else "author",
                           package="vllm" if k.startswith("author") else None)==trees[k] for k in dirs)
assert source_unchanged and ledger.read_bytes()==ledger_before
manifest=dict(status="PASS_P4_02_INCREMENT_AND_P3_AUTHOR_HEAD_FULL_SOURCE_PATCHES",
              author_locked_head=head,results=results,
              changed_control_refs=refs({p:final[p] for p in changed_full if p.startswith("control/")}),
              changed_native_refs=refs({p:final[p] for p in changed_full if p.startswith("native/")}),
              author_head_uncommitted_source_paths=sorted(author_inherited | author_metadata),
              author_head_changed_refs=refs({p:final[p] for p in author_inherited | author_metadata}),
              inherited_common_source_refs=refs({p:trees["author_old"][p] for p in author_inherited}),
              runtime_supplement_refs=refs({p:final[p] for p in supplement}),
              common_py_inherited_uncommitted_fix=False,
              common_py_explanation="author-build common.py equals locked author HEAD; only new P4 metadata field differs in fork",
              source_tree_scope="All .py under control/native and author vllm package; unchanged repository assets and compiled binaries are not included in source patch proof.",
              binary_fallback_required=True,binary_rebuilt=False,
              original_sources_modified=False,source_inputs_unchanged=True,
              ledger_unchanged=True,ledger_sha256=digest(ledger_before),gpu_workloads_run=0,
              storage_after=preflight(scratch_root,0))
(out/"commands.json").write_text(json.dumps(commands,indent=2))
(out/"result.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:manifest[k] for k in ("status","author_locked_head","author_head_uncommitted_source_paths","original_sources_modified","source_inputs_unchanged","gpu_workloads_run")}))
print(json.dumps([dict(base=x["base_name"],files=x["base_files"],final=x["final_files"],changed=x["changed_files"],patches=len(x["patches"])) for x in results]))
