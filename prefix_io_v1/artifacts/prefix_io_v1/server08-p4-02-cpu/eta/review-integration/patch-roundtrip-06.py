"""CPU-only final source delta and exact composed patch roundtrip."""
from pathlib import Path
import sys,json,hashlib,difflib,subprocess
root=Path.cwd()
sys.path.insert(0,str(root/"experiments/prefix_io_v1/scripts"))
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-02-cpu/src"))
sys.path.extend(str(p) for p in (root/".venv/lib").glob("python*/site-packages"))
from experiment_storage import preflight
out=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-integration/patch-roundtrip-06"
scratch_root=root/"experiments/prefix_io_v1/runs/server08-p4-02-patch-roundtrip-06/cpu-evidence"
preflight(scratch_root,128*1024**2)
out.mkdir(exist_ok=False);scratch_root.mkdir(parents=True,exist_ok=False)
ledger=root/"experiments/prefix_io_v1/gpu-budget-ledger.json";ledger_before=ledger.read_bytes()
assert json.loads(ledger_before)["active_reservation"] is None
def digest(b):return hashlib.sha256(b).hexdigest()
def read_tree(directory):
 return {p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob("*.py")
         if not any(x in p.parts for x in (".git","__pycache__",".pytest_cache"))}
def collect(directory,prefix,package=None):
 base=directory if package is None else directory/package
 result={}
 for p in base.rglob("*.py"):
  if any(x in p.parts for x in (".git","__pycache__",".pytest_cache")):continue
  assert not p.is_symlink()
  result[prefix+"/"+p.relative_to(directory).as_posix()]=p.read_bytes()
 return result
def write_tree(directory,tree):
 for name,raw in tree.items():
  p=directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
def refs(tree):return [{"path":p,"bytes":len(v),"sha256":digest(v)} for p,v in sorted(tree.items())]
def require_refs(tree,expected):
 assert refs(tree)==expected,"every actual path/length/source byte SHA must match"
baseline_root=root/"experiments/prefix_io_v1/runs/server08-p4-02-reviewed-baseline-01/cpu-evidence/source"
baseline=read_tree(baseline_root)
assert len(baseline)==1908 and sum(map(len,baseline.values()))==27207528
receipt=json.loads((root/"artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-integration/REVIEWED02_BASELINE_RECONSTRUCTION.json").read_text())
old_out=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-integration/patch-roundtrip-05"
old_inc=json.loads((old_out/"p4-01-increment/result.json").read_text())
require_refs(baseline,old_inc["final_refs"])
dirs={"control":root/"third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control",
      "native":root/"third_party/work/py-kvcache-p4-02-cpu",
      "author":root/"third_party/work/vllm-author-p4-02-cpu"}
final={**collect(dirs["control"],"control"),**collect(dirs["native"],"native"),
       **collect(dirs["author"],"author","vllm")}
assert set(baseline)<=set(final)
changed={p for p,v in final.items() if baseline.get(p)!=v}
assert changed=={"control/p4_gpu_step_observation.py","control/p4_native_window_journal.py",
                 "control/p4_paired_measurement_verifier.py"},sorted(changed)
# The final semantic verifier SHA is supplied only after owner freeze.
assert digest(final["control/p4_paired_measurement_verifier.py"])=="3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac"
assert digest(final["control/p4_gpu_step_observation.py"])=="2dbd58eb095413520c5818cf5bb982ae6ca584933fc9b6db567eef41b29ca690"
assert digest(final["control/p4_native_window_journal.py"])=="3a9ded39846cccf88ee9aceed2e16b86d8a7bbe953adfea3ba1a302b2c5aa8c3"
def patch(state,target,names):
 output=[]
 for name in sorted(names):
  if name not in state and target[name]==b"":
   output.append(chr(10).join(["diff --git a/"+name+" b/"+name,"new file mode 100644","index 0000000..e69de29",""]));continue
  output.append(chr(10).join(["diff --git a/"+name+" b/"+name,""]))
  if name not in state:output.append("new file mode 100644"+chr(10))
  output.extend(difflib.unified_diff(state.get(name,b"").decode().splitlines(keepends=True),
    target[name].decode().splitlines(keepends=True),fromfile="a/"+name if name in state else "/dev/null",tofile="b/"+name))
 return "".join(output)
patches=[]
for name,names,scope in [
 ("0007-forward-native-diagnostic-seams",{"control/p4_gpu_step_observation.py","control/p4_native_window_journal.py"},
  "Two optional bounded observers, not installed in the real worker; full decode/output/drain collector remains incomplete. Original accounting executes first; boundary/invalid/host-backwards observations fail closed. No scheduler/executor/queue."),
 ("0008-v2-semantic-measurement-selection",{"control/p4_paired_measurement_verifier.py"},
  "Finite preregistered actual native ordinal/exact-cell selection, complete sample output reconstruction, GPU elapsed scalar and full-run additional IO consistency; CPU preparation only, no real GPU/production qualification.")]:
 p=out/(name+".patch");p.write_text(patch(baseline,final,names),encoding="utf-8")
 patches.append({"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":digest(p.read_bytes()),"inputs":sorted(names),"scope":scope})
commands=[]
def git_apply(directory,path,reverse=False,check=False):
 cmd=["git","apply"]+(["--reverse"] if reverse else [])+(["--check"] if check else [])+[str(path)]
 p=subprocess.run(cmd,cwd=directory,capture_output=True,text=True)
 commands.append(dict(command=cmd,cwd=str(directory),exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
 (out/"commands.json").write_text(json.dumps(commands,indent=2))
 assert p.returncode==0,commands[-1]

def off_probe(tree,label):
 alias=scratch_root/(label+"-off-control")/"prefix_io_control";alias.mkdir(parents=True)
 for name,raw in tree.items():
  if name.startswith("control/"):
   p=alias/name.split("/",1)[1];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 code="""import sys,importlib.abc
sys.path.insert(0,sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,n,path=None,target=None):
  if n.split('.')[0] in ('torch','vllm','py_kvcache','numpy','cupy','cuda') or n in (
   'prefix_io_control.p4_policy','prefix_io_control.p4_bridge','prefix_io_control.p4_eta',
   'prefix_io_control.p4_gpu_step_observation','prefix_io_control.p4_native_window_journal',
   'prefix_io_control.p4_verified_cost_loader','prefix_io_control.p4_startup_evidence'):
   raise AssertionError('off imported optional strategy/backend '+n)
sys.meta_path.insert(0,Guard())
from prefix_io_control.p4_options import parse_p4_options,build_p4_kwargs
assert 'p4_bridge' not in build_p4_kwargs(parse_p4_options({'prefix_io_p4_policy':{'mode':'off'}}))
print('PASS final off without optional policy, observers, evidence or backend')
"""
 cmd=[str(root/".venv/bin/python"),"-I","-S","-c",code,str(alias.parent)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
 commands.append(dict(command=cmd,cwd=str(root),exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
 assert p.returncode==0,commands[-1]
 return p.stdout

results=[]
for label in ("p4-01-increment","p3-author-head-full"):
 old=json.loads((old_out/label/"result.json").read_text())
 base_root=root/"experiments/prefix_io_v1/runs/server08-p4-02-patch-roundtrip-05/cpu-evidence"/label
 base=read_tree(base_root);require_refs(base,old["base_refs"])
 scratch=scratch_root/label;scratch.mkdir();write_tree(scratch,base)
 subprocess.run(["git","init","--quiet",str(scratch)],capture_output=True,check=True)
 for record in old["patches"]:
  p=root/record["path"];assert p.stat().st_size==record["bytes"] and digest(p.read_bytes())==record["sha256"]
  git_apply(scratch,p,check=True);git_apply(scratch,p)
 require_refs(read_tree(scratch),old["final_refs"])
 assert read_tree(scratch)==baseline
 for record in patches:
  p=root/record["path"];git_apply(scratch,p,check=True);git_apply(scratch,p)
 assert read_tree(scratch)==final
 off_result=off_probe(final,label)
 for record in reversed(old["patches"]+patches):
  p=root/record["path"];git_apply(scratch,p,reverse=True,check=True);git_apply(scratch,p,reverse=True)
 assert read_tree(scratch)==base
 result={"base_name":label,"status":"PASS_COMPOSED_FORWARD_REVERSE_BYTE_ROUNDTRIP",
         "base_files":len(base),"reviewed02_baseline_files":len(baseline),"final_files":len(final),
         "changed_files":sum(base.get(p)!=v for p,v in final.items()),"patches":old["patches"]+patches,
         "base_refs":refs(base),"final_refs":refs(final),"source_paths_removed":False,"final_off_guard":off_result}
 directory=out/label;directory.mkdir();(directory/"result.json").write_text(json.dumps(result,indent=2))
 results.append({k:v for k,v in result.items() if k not in ("base_refs","final_refs")})
# The separate offline CLI has an independently retained old source byte base.
cli_relative="experiments/prefix_io_v1/scripts/prepare_p4_raw_pair.py"
cli_before=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/cost-verifier/run-13-v2-cli-red/cli-old-source.py"
cli_old=cli_before.read_bytes();cli_final=(root/cli_relative).read_bytes()
assert digest(cli_final)=="dae3b4f78d174233ae63eb9f2ca2cd1aefdfd5160d2e09ccf6a872317202c6a4"
cli_patch=out/"0009-offline-v2-cli-adapter.patch"
cli_patch.write_text(patch({cli_relative:cli_old},{cli_relative:cli_final},{cli_relative}),encoding="utf-8")
cli_scratch=scratch_root/"offline-cli-increment";cli_scratch.mkdir()
write_tree(cli_scratch,{cli_relative:cli_old})
subprocess.run(["git","init","--quiet",str(cli_scratch)],capture_output=True,check=True)
git_apply(cli_scratch,cli_patch,check=True);git_apply(cli_scratch,cli_patch)
assert read_tree(cli_scratch)=={cli_relative:cli_final}
git_apply(cli_scratch,cli_patch,reverse=True,check=True);git_apply(cli_scratch,cli_patch,reverse=True)
assert read_tree(cli_scratch)=={cli_relative:cli_old}
assert cli_before.read_bytes()==cli_old and (root/cli_relative).read_bytes()==cli_final
cli_proof={"status":"PASS_OFFLINE_CLI_FORWARD_REVERSE_BYTE_ROUNDTRIP","base_source":cli_before.relative_to(root).as_posix(),
 "base_bytes":len(cli_old),"base_sha256":digest(cli_old),"final_source":cli_relative,"final_bytes":len(cli_final),
 "final_sha256":digest(cli_final),"patch":{"path":cli_patch.relative_to(root).as_posix(),"bytes":cli_patch.stat().st_size,"sha256":digest(cli_patch.read_bytes())},
 "scope":"Offline V2 byte-ref ingestion adapter only, no GPU collector or production qualification."}
# The baseline is itself frozen; production files and accounting ledger are read-only.
assert read_tree(baseline_root)==baseline and ledger.read_bytes()==ledger_before
again={**collect(dirs["control"],"control"),**collect(dirs["native"],"native"),**collect(dirs["author"],"author","vllm")}
assert again==final
extras=["experiments/prefix_io_v1/scripts/prepare_p4_gpu_next_day.py",
        "experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py",
        "experiments/prefix_io_v1/scripts/prepare_p4_raw_pair.py",
        "tests/prefix_io_v1_p4_observation/test_forward_native_journal.py",
        "tests/prefix_io_v1_p4_measurements/test_semantic_v2.py",
        "tests/prefix_io_v1_p4_measurements/test_offline_v2_cli.py",
        "tests/prefix_io_v1_p4_measurements/test_semantic_verifier.py"]
extra_refs=refs({p:(root/p).read_bytes() for p in extras if (root/p).is_file()})
source_map={"control":"third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control",
            "native":"third_party/work/py-kvcache-p4-02-cpu","author":"third_party/work/vllm-author-p4-02-cpu"}
def live_path(alias):
 head,tail=alias.split("/",1);return source_map[head]+"/"+tail
manifest={"status":"PASS_FINAL_P4_02_DIAGNOSTIC_AND_V2_DELTA_WITH_COMPOSED_BASES",
          "reviewed02_baseline_source":baseline_root.relative_to(root).as_posix(),
          "reviewed02_baseline_files":len(baseline),"final_source_files":len(final),"delta_files":len(changed),
          "delta_changed_source_refs":[{"path":live_path(p),"bytes":len(final[p]),"sha256":digest(final[p])} for p in sorted(changed)],
          "delta_patches":patches,"results":results,"offline_cli_byte_proof":cli_proof,"auxiliary_script_test_refs":extra_refs,
          "auxiliary_scope":"Final source refs; tests/preparation/qualification scripts are outside the package patch byte proof. Offline CLI has a separate exact old-source forward/reverse proof.",
          "author_locked_head":"817a7e3124f817cd6e549581d3e5483207a753a4",
          "inherited_author_common_fix":"platforms/cuda.py GPU UUID lookup; unchanged from reviewed02. Offloading common.py was equal HEAD before P4 metadata.",
          "inherited_author_flush_observation":"scheduler.py old opt-in flush probe is retained in both composed chains.",
          "runtime_supplement":"92 exact old-build pure Python files remain separately proven in patch05; no binary copies/builds.",
          "source_tree_scope":"All control/native Python and author vllm package Python; unchanged repo assets and binary ABI are outside this proof.",
          "original_source_inputs_unchanged":True,"reviewed_baseline_unchanged":True,"ledger_unchanged":True,
          "ledger_sha256":digest(ledger_before),"gpu_operations":0,"binary_execution_calls":0,
          "gpu_verified":False,"production_qualified":False,"observers_installed_real_worker":False,
          "full_decode_output_drain_collector_complete":False,"storage_after":preflight(scratch_root,0)}
(out/"result.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:manifest[k] for k in ("status","delta_changed_source_refs","gpu_operations","original_source_inputs_unchanged")}))
(out/"commands.json").write_text(json.dumps(commands,indent=2))
print(json.dumps([dict(base=r["base_name"],base_files=r["base_files"],final_files=r["final_files"],changed=r["changed_files"],patches=len(r["patches"])) for r in results]))
