"""Explicitly granted auxiliary storage; no mounts, deletion, or implicit fallback."""
import json,os,shutil
from pathlib import Path
from prefix_io_control.config import read_yaml,validate_permissions

ROOT=Path(__file__).resolve().parents[3]
def require(ok,message):
    if not ok:raise ValueError(message)

def permission(project=ROOT):
    p=validate_permissions(read_yaml(project/"experiments/prefix_io_v1/configs/permissions.yaml"))
    aux=p.get("approved_auxiliary_storage")
    if aux:
        grant_path=(project/aux["authorization_record"]).resolve()
        require(grant_path.is_relative_to(project.resolve()),"grant outside project")
        grant=json.loads(grant_path.read_text())
        require(all(grant[k]==aux[k] for k in ("root","max_bytes","minimum_free_bytes")),"authorization mismatch")
        require(grant["gpu_uuid"] in p["approved_gpu_ids"] and grant["max_gpu_hours_unchanged"]==p["max_gpu_hours"],
                "authorization environment/budget mismatch")
    return p

def authorized_path(path,project=ROOT):
    p=permission(project);path=Path(path).resolve()
    primary=Path(p["approved_experiment_root"]).resolve()
    if path!=primary and path.is_relative_to(primary):return path,None
    aux=p.get("approved_auxiliary_storage")
    if aux:
        root=Path(aux["root"])
        require(root.resolve()==root and not root.is_symlink(),"auxiliary root may not be redirected")
        if path!=root and path.is_relative_to(root):return path,aux
    raise ValueError("path outside explicitly approved experiment storage")

def usage(root):
    # Unique inodes count hardlinks once; do not follow the model alias symlink.
    seen=set();used=0;files=0
    for folder,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
        for name in names:
            p=Path(folder)/name
            if p.is_symlink():continue
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);used+=max(s.st_size,s.st_blocks*512);files+=1
    return dict(used_bytes=used,unique_files=files)

def budget_values(used,free,new_bytes,max_bytes,minimum_free_bytes):
    require(all(type(x) is int and x>=0 for x in (used,free,new_bytes,max_bytes,minimum_free_bytes)),"nonnegative byte integers required")
    require(used+new_bytes<=max_bytes,"auxiliary 20 GiB cap exceeded")
    require(free-new_bytes>=minimum_free_bytes,"auxiliary minimum free space would be crossed")
    return dict(used_bytes=used,free_bytes=free,reserved_new_bytes=new_bytes,
                max_bytes=max_bytes,minimum_free_bytes=minimum_free_bytes)

def preflight(path,new_bytes=0,project=ROOT):
    path,aux=authorized_path(path,project)
    if aux:
        root=Path(aux["root"]);require(root.is_dir(),"authorized root not provisioned")
        consumed=usage(root)
        result=budget_values(consumed["used_bytes"],shutil.disk_usage(root).free,new_bytes,
                             aux["max_bytes"],aux["minimum_free_bytes"])
        return dict(result,root=str(root),device=root.stat().st_dev,path=str(path),unique_files=consumed["unique_files"])
    ancestor=path
    while not ancestor.exists():ancestor=ancestor.parent
    require(shutil.disk_usage(ancestor).free-new_bytes>=8*1024**3,"primary disk floor would be crossed")
    return dict(path=str(path),root=str(project/"experiments/prefix_io_v1/runs"),
                free_bytes=shutil.disk_usage(ancestor).free,reserved_new_bytes=new_bytes)

def details_path(project,label,plan):
    if "run_details" in plan:
        require(label in plan["run_details"],"missing frozen run path")
        path=Path(plan["run_details"][label])
    else:path=project/"experiments/prefix_io_v1/runs"/label/"details"
    path,_=authorized_path(path,project);return path
