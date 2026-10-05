"""Read-only deduplication opportunity audit; never alters cache data."""
import argparse,collections,hashlib,json,os,time
from pathlib import Path
from experiment_storage import preflight,ROOT

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b=f.read(1024**2)
            if not b:break
            h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    ledger=json.loads((ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json").read_text())
    if ledger.get("active_reservation") is not None:raise RuntimeError("audit must not contend with a GPU experiment")
    base=Path("/root/prefix-io-v1-validation").resolve();seen=set();groups=collections.defaultdict(list);unique_bytes=0
    # Only completed project run caches are candidates. Published source is read-only canonical evidence.
    roots=[]
    for run in sorted((base/"runs").iterdir()):
        storage=run/"details/storage";result=run/"details/result.json"
        if not storage.is_dir() or not result.is_file():continue
        state=json.loads(result.read_text())
        if not str(state.get("status","")).startswith("PASSED_") or state.get("engine_shutdown")!="completed":continue
        roots.append(storage)
    source=base/"storage/heldout-long"
    if source.is_dir():roots.insert(0,source)
    for root in roots:
        for p in sorted(root.rglob("*.bin")):
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):continue
            s=p.stat();ident=(s.st_dev,s.st_ino)
            if ident in seen:continue
            seen.add(ident)
            key=(s.st_size,digest(p));after=p.stat()
            if (s.st_size,s.st_mtime_ns,s.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError("cache changed during read-only audit")
            unique_bytes+=s.st_size
            groups[key].append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,device=s.st_dev,
                inode=s.st_ino,links=s.st_nlink,mtime_ns=s.st_mtime_ns,
                private_completed_run=root!=source and s.st_nlink==1))
    proposals=[];savings=0;count=0
    for (size,sha),items in groups.items():
        if len(items)<2:continue
        # Prefer an existing published/shared source as canonical; it would never be modified.
        canonical=next((x for x in items if not x["private_completed_run"]),items[0])
        targets=[x for x in items if x is not canonical and x["private_completed_run"] and x["device"]==canonical["device"]]
        if not targets:continue
        reclaim=sum(x["allocated_bytes"] for x in targets);savings+=reclaim;count+=len(targets)
        proposals.append(dict(sha256=sha,bytes=size,canonical=canonical,targets=targets,reclaimable_allocated_bytes=reclaim))
    result=dict(unix=time.time(),read_only=True,data_changed=False,completed_cache_roots=len(roots)-int(source in roots),
        unique_files_hashed=len(seen),unique_bytes_hashed=unique_bytes,candidate_files=count,candidate_groups=len(proposals),
        possible_reclaim_bytes=savings,proposals=proposals,
        caution="Proposal only. Replacing private copies with hard links retains bytes/paths but creates shared inodes; requires explicit scoped approval and revalidation before any change.",
        storage=preflight(base/"runs/next",0))
    with a.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="proposals"},indent=2))
if __name__=="__main__":main()
