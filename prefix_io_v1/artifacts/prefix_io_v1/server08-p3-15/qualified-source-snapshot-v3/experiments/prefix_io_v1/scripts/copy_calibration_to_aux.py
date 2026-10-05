"""Preserve original published KV; copy and verify into the approved filesystem."""
import argparse,hashlib,json,shutil
from pathlib import Path
from experiment_storage import ROOT,permission,preflight
p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
aux=Path(permission()["approved_auxiliary_storage"]["root"])
groups=[("long",1152),("8k",1536),("16k",3072)]
result=[]
for name,count in groups:
    source=ROOT/"experiments/prefix_io_v1/runs"/("server07-p3-"+name+"-storage-01")
    target=aux/"storage"/("calibration-"+name)
    files=sorted(source.rglob("*.bin"))
    assert len(files)==count and not target.exists()
    budget=preflight(target,int(count*917504*1.03)+64*1024**2)
    target.mkdir(parents=True);manifest=[]
    for src in files:
        assert not src.is_symlink() and src.stat().st_size==917504 and src.resolve().is_relative_to(source.resolve())
        dst=target/src.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copyfile(src,dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
        manifest.append(dict(path=str(src.relative_to(source)),bytes=917504,sha256=digest))
    record=dict(source=str(source),target=str(target),files=count,bytes=count*917504,
        manifest=manifest,budget_before=budget,budget_after=preflight(target,0),
        copied_not_moved=True,no_source_deletion=True)
    result.append(record)
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!="manifest"}),flush=True)
