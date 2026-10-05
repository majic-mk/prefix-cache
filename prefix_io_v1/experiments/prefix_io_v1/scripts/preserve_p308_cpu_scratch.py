"""Preserve this turn's own CPU scratch data while restoring the disk floor."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"artifacts/prefix_io_v1/server07-p3-08"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments/prefix_io_v1/scripts")]
from experiment_storage import preflight
TARGET=Path("/root/prefix-io-v1-validation/cpu-evidence/server07-p3-08")
names=("targeted-tmp","combined-tmp","final-tmp")
def inventory(root):
    result={}
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=Path(directory)/name
            rel=str(p.relative_to(root))
            if p.is_symlink():
                result[rel]=dict(kind="symlink",target=os.readlink(p))
            elif p.is_file():
                result[rel]=dict(kind="file",bytes=p.stat().st_size,
                                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return result
assert not TARGET.exists()
assert OUT.resolve()==OUT and OUT.is_relative_to(ROOT/"artifacts/prefix_io_v1")
sources=[]
total=0
for name in names:
    src=OUT/name
    assert src.exists() and src.is_dir() and not src.is_symlink()
    assert src.resolve().parent==OUT and src.name in names
    data=inventory(src)
    total+=sum(x.get("bytes",0) for x in data.values())
    sources.append((src,data))
pre=preflight(TARGET,total+16*1024**2)
record=dict(reason="CPU scratch exceeded primary free-space floor; preserve all test bytes in authorized auxiliary storage",
            primary_before=shutil.disk_usage(ROOT)._asdict(),aux_preflight=pre,
            planned_payload_bytes=total,old_or_shared_data_affected=False,rows=[],unix=time.time())
(OUT/"scratch-relocation-plan.json").write_text(json.dumps(record,indent=2))
TARGET.mkdir(parents=True)
for src,original in sources:
    dst=TARGET/src.name
    shutil.copytree(src,dst,symlinks=True)
    copied=inventory(dst)
    assert copied==original,("verification failed; original retained",src)
    # The source is exclusively this turn's finished pytest scratch tree.
    # Check both paths again immediately before retiring the verified duplicate.
    assert src.resolve().parent==OUT and dst.resolve().parent==TARGET
    assert src.name in names and not src.is_symlink() and not dst.is_symlink()
    shutil.rmtree(src)
    row=dict(source=str(src),destination=str(dst),files=original,
             manifest_equal=True,old_source_retired_after_verified_copy=True,
             symlinks="literal targets preserved as evidence; pytest current-links are not replay entry points")
    record["rows"].append(row)
    (OUT/"scratch-relocation-progress.json").write_text(json.dumps(record,indent=2))
record["primary_after"]=shutil.disk_usage(ROOT)._asdict()
record["aux_after"]=preflight(TARGET,0)
record["primary_floor_restored"]=record["primary_after"]["free"]>=8*1024**3
assert record["primary_floor_restored"]
(OUT/"scratch-relocation.json").write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!="rows"},indent=2))
