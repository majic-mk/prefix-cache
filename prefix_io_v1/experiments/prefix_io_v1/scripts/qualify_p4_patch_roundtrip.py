"""Build and verify separate P4 observer/glue and research patches on fresh CPU scratch."""
import argparse, difflib, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"artifacts/prefix_io_v1/server08-p4-01-cpu"
def digest(raw):return hashlib.sha256(raw).hexdigest()
def collect():
    base={};current={}
    for prefix,old,new in (
        ("control",ROOT/"src/prefix_io_control",ROOT/"third_party/work/prefix-io-p4-01-cpu/src/prefix_io_control"),
        ("native",ROOT/"third_party/work/py-kvcache-p3-16-cpu",ROOT/"third_party/work/py-kvcache-p4-01-cpu")):
        for target,directory in ((base,old),(current,new)):
            for p in directory.rglob("*.py"):
                if any(s in p.parts for s in (".git","__pycache__",".pytest_cache")):continue
                assert not p.is_symlink(),"source symlink requires independent freeze"
                target[prefix+"/"+str(p.relative_to(directory))]=p.read_bytes()
    assert set(base)<=set(current),"P4 must preserve original source paths"
    return base,current
def patch(base,current,names):
    parts=[]
    for name in sorted(names):
        before=base.get(name,b"").decode().splitlines(keepends=True)
        after=current[name].decode().splitlines(keepends=True)
        parts.extend(difflib.unified_diff(before,after,
            fromfile="a/"+name if name in base else "/dev/null",tofile="b/"+name))
    return "".join(parts)
def read_tree(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*.py")}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--name",required=True);a=ap.parse_args()
    assert a.name and all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in a.name)
    out=OUT/a.name;out.mkdir(parents=True,exist_ok=False)
    base,current=collect()
    changed={k for k,v in current.items() if base.get(k)!=v}
    research={k for k in changed if k in {
        "control/p4_policy.py","control/p4_cost_table.py","control/p4_production_table_contract.py"}}
    observer=changed-research
    assert research and observer,"both boundaries must be concrete"
    scratch=out/"scratch";scratch.mkdir()
    for name,raw in base.items():
        p=scratch/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    reports=[]
    for label,names in (("0001-observer-native-glue",observer),("0002-bounded-research-policy",research)):
        p=out/(label+".patch");p.write_text(patch(base,current,names))
        subprocess.run(["git","apply","--check",str(p)],cwd=scratch,check=True,capture_output=True)
        subprocess.run(["git","apply",str(p)],cwd=scratch,check=True,capture_output=True)
        reports.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p.read_bytes()),bytes=p.stat().st_size,
                            inputs=sorted(names)))
        if label.startswith("0001"):
            # Off must work with only observer/glue; missing strategy cannot be imported or scanned.
            code="""import importlib.abc,sys
sys.path.insert(0,sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','vllm','py_kvcache','cupy'} or name in {'prefix_io_control.p4_policy','prefix_io_control.p4_bridge'}:
   raise AssertionError('off imported strategy/backend '+name)
sys.meta_path.insert(0,Guard())
from prefix_io_control.p4_options import parse_p4_options,build_p4_kwargs
o=parse_p4_options({'prefix_io_p4_policy':{'mode':'off'}})
assert 'p4_bridge' not in build_p4_kwargs(o)
print('PASS common-only off: no strategy/backend imports')
"""
            # Adapter folder becomes the actual package at a safe alias in this new scratch.
            alias=out/"common-off-import"/"prefix_io_control";alias.mkdir(parents=True)
            for f in (scratch/"control").glob("*.py"):(alias/f.name).write_bytes(f.read_bytes())
            result=subprocess.run([str(ROOT/".venv/bin/python"),"-I","-S","-c",code,str(alias.parent)],check=True,capture_output=True,text=True)
            (out/"common-only-off.log").write_text(result.stdout)
    assert read_tree(scratch)==current,"forward apply must match every P4 byte"
    for report in reversed(reports):
        p=ROOT/report["path"]
        subprocess.run(["git","apply","--reverse","--check",str(p)],cwd=scratch,check=True,capture_output=True)
        subprocess.run(["git","apply","--reverse",str(p)],cwd=scratch,check=True,capture_output=True)
    assert read_tree(scratch)==base,"reverse apply must match every P3 byte"
    result=dict(status="PASS_P4_PATCH_BYTE_ROUNDTRIP_AND_COMMON_ONLY_OFF",base_files=len(base),
        final_files=len(current),changed_files=len(changed),patches=reports,
        baseline_sha256={k:digest(v) for k,v in base.items()},
        p4_sha256={k:digest(v) for k,v in current.items()},
        original_sources_modified=False,GPU_initialized=False,new_GPU_runs=0,
        grouping="No new engine correctness fix. Common P3 bytes preserved; observer/native-glue hook is inert with off; research logic in separate patch.")
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","base_files","final_files","changed_files","new_GPU_runs")}))
if __name__=="__main__":main()
