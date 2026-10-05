"""Bounded byte-exact PRIMARY copy of the frozen published source. No deletion."""
from pathlib import Path
import hashlib,json,sys,time,shutil
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"experiments/prefix_io_v1/scripts"),str(ROOT/"src")]
from experiment_storage import preflight,permission
ORIGIN=Path("/root/prefix-io-v1-validation/storage/heldout-long")
MANIFEST=Path("/root/prefix-io-v1-validation/runs/server07-p3-12-mixed-order-01-off/details/storage-source-manifest.json")
FROZEN="c03381abb29e21b54a62a4815892553b58e4ef39cc8087761cc4d173cf75d274"
OUT=ROOT/"experiments/prefix_io_v1/runs/server08-p3-14-source"
SOURCE=OUT/"heldout-long"
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();raw=MANIFEST.read_bytes();assert hashlib.sha256(raw).hexdigest()==FROZEN
 rows=json.loads(raw);assert len(rows)==3048 and sum(r["bytes"] for r in rows)==2796552192
 assert permission(ROOT)["approved_gpu_ids"]==["GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8"]
 before=preflight(SOURCE,3*1024**3);OUT.mkdir(parents=True,exist_ok=False);SOURCE.mkdir()
 seen=set()
 for row in rows:
  rel=Path(row["path"]);assert not rel.is_absolute() and ".." not in rel.parts and rel.suffix==".bin"
  assert row["path"] not in seen;seen.add(row["path"])
  src=ORIGIN/rel;dst=SOURCE/rel
  assert src.is_file() and not any(p.is_symlink() for p in [src,*src.parents])
  assert src.stat().st_size==row["bytes"]
  dst.parent.mkdir(parents=True,exist_ok=True)
  h=hashlib.sha256();n=0
  with src.open("rb") as fi,dst.open("xb") as fo:
   for b in iter(lambda:fi.read(1024*1024),b""):h.update(b);n+=len(b);fo.write(b)
  assert n==row["bytes"] and h.hexdigest()==row["sha256"]
  assert dst.stat().st_size==row["bytes"] and digest(dst)==row["sha256"]
 assert set(p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*") if p.is_file())==seen
 assert all(digest(ORIGIN/r["path"])==r["sha256"] for r in rows)
 assert MANIFEST.read_bytes()==raw and SOURCE.stat().st_dev!=(ORIGIN.stat().st_dev)
 source_manifest=OUT/"source-manifest.json";source_manifest.write_bytes(raw)
 reg=dict(schema_version=1,gpu_uuid="GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8",
  source_root=str(SOURCE),origin_root=str(ORIGIN),origin_manifest=str(MANIFEST),origin_manifest_sha256=FROZEN,
  source_manifest=str(source_manifest),source_manifest_sha256=FROZEN,file_count=3048,total_bytes=2796552192)
 with (OUT/"registration.json").open("x") as f:json.dump(reg,f,indent=2)
 proof=dict(status="PASS_BYTE_EXACT_PRIMARY_COPY",files=len(rows),total_bytes=reg["total_bytes"],
  original_unchanged=True,source_device=SOURCE.stat().st_dev,origin_device=ORIGIN.stat().st_dev,
  copied_not_cross_device_hardlinked=True,manifest_sha256=FROZEN,
  registration_sha256=digest(OUT/"registration.json"),before=before,after=preflight(SOURCE,0),
  seconds=time.monotonic()-start)
 with (OUT/"copy-proof.json").open("x") as f:json.dump(proof,f,indent=2)
 print(json.dumps(proof))
if __name__=="__main__":main()
