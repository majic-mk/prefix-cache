"""Freeze CPU verification of a real gated GPU entry revision. No GPU authority."""
from __future__ import annotations
import argparse,datetime,hashlib,json,pathlib,subprocess,sys
NAMES={"candidate":"server11-c5-gpu-entry-revision-20261004","review":"server11-c5-gpu-entry-review-20261004","delivery":"server11-c5-gpu-entry-delivery-20261004"}
PARENT="artifacts/prefix_io_v1/server11-c5-combined-runtime-delivery-cpu-20261004/SOURCE_LOCK_COMBINED_RUNTIME_CPU.json"
PARENT_SHA="bb7522102bddee35e5878551e98d23ae475897e24663d4c3aa9ae235329b176d"
FLAGS=("gpu_launch_allowed","native_execution_verified","native_cost_qualified","full_runtime_cost_qualified","on_observation_cost_measured")
def require(ok,reason):
    if not ok:raise ValueError(reason)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def unique(items):
    value={}
    for k,v in items:
        require(k not in value,"duplicate JSON key")
        value[k]=v
    return value
def read(path):return json.loads(path.read_bytes(),object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError("nonfinite JSON")))
def safe(root,name):
    require(type(name) is str and name and "\\" not in name and ":" not in name and not name.startswith("/") and all(p not in ("",".","..") for p in name.split("/")),"relative bounded path")
    path=root/name
    require(path.resolve().is_relative_to(root) and not any(p.is_symlink() for p in (path,)+tuple(path.parents)),"source contained no symlink")
    return path
def reference(root,path):
    require(path.is_file() and path.stat().st_size<=10*1024**2,"source cap")
    b=path.read_bytes()
    return {"path":path.relative_to(root).as_posix(),"bytes":len(b),"sha256":sha(b)}
def verify(root,row):
    require(type(row) is dict and set(row)=={"path","bytes","sha256"} and type(row["bytes"]) is int and 0<=row["bytes"]<=70*1024**2,"exact bounded prior/source ref")
    path=safe(root,row["path"])
    require(path.is_file() and path.stat().st_size==row["bytes"],"source size drift "+row["path"])
    digest=hashlib.sha256();count=0
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):
            count+=len(block);require(count<=row["bytes"],"source grew");digest.update(block)
    require(count==row["bytes"] and digest.hexdigest()==row["sha256"],"source drift "+row["path"])
    return path
def put(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write("\n")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=pathlib.Path,required=True)
    a=parser.parse_args()
    root=a.root.resolve(strict=True)
    dirs={k:root/"artifacts/prefix_io_v1"/n for k,n in NAMES.items()}
    require(pathlib.Path(__file__).resolve()==dirs["delivery"]/"freeze_gpu_entry.py","fixed server freezer")
    for d in dirs.values():require(d.is_dir() and not d.is_symlink(),"new bounded scope exists")
    baselinepath=dirs["delivery"]/"SESSION_BEFORE_GPU_ENTRY_PREPARATION.json"
    before=read(baselinepath)
    for row in before["refs"]:verify(root,row)
    parentpath=safe(root,PARENT);parent=read(parentpath)
    require(sha(parentpath.read_bytes())==PARENT_SHA and len(parent["files"])==202 and parent["gpu_uuid"] is None and all(parent[k] is False for k in FLAGS),"closed parent CPU source")
    selected={}
    def add(path,expected=None):
        row=reference(root,path)
        if expected is not None:require(row==expected,"ancestry drift")
        require(row["path"] not in selected or selected[row["path"]]==row,"no aliases")
        selected[row["path"]]=row
    for row in parent["files"]:add(safe(root,row["path"]),row)
    add(parentpath);add(baselinepath)
    for scope,d in dirs.items():
        for path in sorted(d.rglob("*")):
            require(not path.is_symlink(),"new source symlink")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py",".md",".txt"):
                add(path)
        for path in sorted(d.glob("*INHERITANCE*.json")):add(path)
    assetpath=dirs["delivery"]/"SERVER_PREPARATION_RESOURCE_ASSETS.json"
    assets=read(assetpath)
    add(assetpath)
    add(safe(root,"experiments/prefix_io_v1/scripts/run_gpu_stage.py"))
    add(safe(root,assets["input_manifest_ref"]["path"]),assets["input_manifest_ref"])
    for row in assets["actual_input_and_provenance_refs_verified"]:add(safe(root,row["path"]),row)
    for name in ("control_native_cost_job.py","native_conditional_cost.py","prepare_and_verify_native_cost.py","run_native_cost_experiment.py"):
        add(safe(root,"artifacts/prefix_io_v1/server11-native-cost-v6-20261003/"+name))
    require(len(selected)<=512,"CPU proof closure bound")
    lock={"schema":"c5_gpu_entry_cpu_source_lock_v1","source_only":True,"gpu_uuid":None,"gpu_entry_code_prepared":True,
          "gpu_launch_allowed":False,"native_execution_verified":False,"native_cost_qualified":False,
          "full_runtime_cost_qualified":False,"on_observation_cost_measured":False,"valid_native_receipt":None,
          "effective_cost_upper_ns":None,"effective_step_budget_ns":None,"parent_ref":reference(root,parentpath),
          "baseline_ref":reference(root,baselinepath),"source_files":len(selected),"files":[selected[k] for k in sorted(selected)]}
    lockpath=dirs["delivery"]/"SOURCE_LOCK_GPU_ENTRY_CPU.json";put(lockpath,lock)
    receipt={"status":"PASS_CPU_SOURCE_FREEZE_NO_GPU_GRANT","source_lock_ref":reference(root,lockpath),"source_rows":len(selected),
             "baseline_ref":reference(root,baselinepath),"prior_references_verified":len(before["refs"]),
             "gpu_runs":0,"gpu_launch_allowed":False,"gpu_uuid":None,"actual_command":[sys.executable]+sys.argv}
    put(dirs["delivery"]/"SOURCE_FREEZE_RECEIPT_CPU.json",receipt)
    print(json.dumps(receipt,sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())

