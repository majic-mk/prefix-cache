from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,io,stat,time,subprocess,shutil,datetime
root=Path(".").resolve();out=Path("artifacts/prefix_io_v1/server08-p3-16")
def rd(p):return json.loads(Path(p).read_text())
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024**2),b""):h.update(b)
 return h.hexdigest()
def names(z,rows,extras):
 values=z.infolist();seen=[i.filename for i in values]
 assert len(seen)==len(set(seen))
 for i in values:
  p=PurePosixPath(i.filename);assert not p.is_absolute() and ".." not in p.parts and "\\" not in i.filename and not stat.S_ISLNK(i.external_attr>>16)
 assert set(seen)=={r["entry"] for r in rows}|set(extras)
def rows(z,entries):
 for r in entries:
  h=hashlib.sha256();size=0
  with z.open(r["entry"]) as f:
   for b in iter(lambda:f.read(1024**2),b""):h.update(b);size+=len(b)
  assert size==r["bytes"] and h.hexdigest()==r["sha256"],r["entry"]
start=time.monotonic()
m=rd(out/"server08-p3-16-complete-evidence-v1-manifest.json");archive=Path(m["archive"]["path"])
assert archive.stat().st_size==m["archive"]["bytes"]==62590070
assert sha(archive)==m["archive"]["sha256"]=="70cde45fb9389b8acb16fe236765d02d73b575bebf3ee564ae712ef0ae98b699"
with zipfile.ZipFile(archive) as z:
 inner=json.loads(z.read("final-evidence-file-manifest.json"))
 assert all(inner[k]==m[k] for k in inner)
 assert inner["P3_complete"] is True and inner["P4_enabled"] is False
 names(z,m["delta_files"],[m["historical_base_entry"],"final-evidence-file-manifest.json"]);rows(z,m["delta_files"])
 oldbytes=z.read(m["historical_base_entry"])
 assert len(oldbytes)==m["historical_base"]["bytes"] and digest(oldbytes)==m["historical_base"]["sha256"]
 with zipfile.ZipFile(io.BytesIO(oldbytes)) as bz:
  bm=json.loads(bz.read("evidence-file-manifest.json"));names(bz,bm["files"],["evidence-file-manifest.json"]);rows(bz,bm["files"])
  source_bytes=bz.read(bm["source_snapshot"])
  sm=json.loads(bz.read("project/artifacts/prefix_io_v1/server08-p3-16/frozen-input-snapshot-at-storage-block-v2.json"))
  assert len(source_bytes)==sm["archive"]["bytes"] and digest(source_bytes)==sm["archive"]["sha256"]
  with zipfile.ZipFile(io.BytesIO(source_bytes)) as sz:
   source=json.loads(sz.read("snapshot-entries.json"));names(sz,source["files"],["snapshot-entries.json"]);rows(sz,source["files"])
 assert len(bm["files"])==1927 and len(source["files"])==2599
covered={r["source_path"]:r["sha256"] for r in [*bm["files"],*source["files"],*m["delta_files"]]}
lock=rd(out/"execution-lock-12-final-p3.json")
assert len(lock)==2724
assert all(covered[str((Path(p) if Path(p).is_absolute() else root/p).resolve())]==h for p,h in lock.items())
assert all(sha(p)==h for p,h in lock.items())
review=rd(out/"p3-final-root-review.json");assert review["p3_phase_closed"] and not review["P4_enabled"]
budget=rd("experiments/prefix_io_v1/gpu-budget-ledger.json")
assert budget==rd(out/"gpu-budget-final-p3.json") and budget["active_reservation"] is None and budget["gpu_wall_seconds"]<28800
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True).strip()
free=shutil.disk_usage(root/"experiments/prefix_io_v1/runs").free;assert free>=8*1024**3
v=dict(schema_version=1,status="PASS_FINAL_OUTER_AND_HISTORICAL_ARCHIVE_EVERY_FILE_AND_ALL2724_REQUIRED_IDENTITIES",P3_complete=True,P4_enabled=False,archive=m["archive"],archive_manifest=dict(path=str(out/"server08-p3-16-complete-evidence-v1-manifest.json"),sha256=sha(out/"server08-p3-16-complete-evidence-v1-manifest.json")),final_delta_files_verified=len(m["delta_files"]),historical_metadata_files_verified=len(bm["files"]),historical_frozen_input_files_verified=len(source["files"]),all2724_required_input_keys_covered=True,all_live_locked_bytes_unchanged=True,GPU_active_reservation=None,GPU_compute_processes=[],GPU_successful_stages=37,GPU_failed_attempts=1,CPU_unique_passed=1613,CPU_skipped=16,new_GPU_workloads_run=0,model_payload_read=False,local_full_archive_download_verified=False,verification_location="server08",primary_free_bytes=free,seconds=time.monotonic()-start,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (out/"final-evidence-archive-server-verification.json").open("x",encoding="utf8") as f:json.dump(v,f,indent=2,ensure_ascii=False)
print(json.dumps(v))
