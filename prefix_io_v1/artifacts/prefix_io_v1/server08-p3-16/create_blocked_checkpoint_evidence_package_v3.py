from pathlib import Path
import json,hashlib,zipfile,shutil,datetime,time,subprocess
root=Path(".").resolve();out=root/"artifacts/prefix_io_v1/server08-p3-16";primary=root/"experiments/prefix_io_v1/runs"
ledger=json.loads((root/"experiments/prefix_io_v1/gpu-budget-ledger.json").read_text());assert ledger["active_reservation"] is None
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True).strip()
locked=json.loads((out/"execution-lock-09-at-storage-block.json").read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in locked.items())
free=shutil.disk_usage(primary).free;reserve=512*1024*1024
assert free-reserve>=8*1024**3 and out.stat().st_dev==primary.stat().st_dev
suffixes={".json",".jsonl",".xml",".log",".txt",".md",".yaml",".yml",".csv",".tsv",".patch",".py"}
bases=[out,root/"artifacts/prefix_io_v1/server08-p3-15"]+[p for p in primary.iterdir() if p.is_dir() and p.name.startswith(("server08-p3-16-","server08-p3-15-"))]
selected={}
# Explicit metadata-only inventory; source/model/cache binary payloads are never traversed as entries.
for base in bases:
 for p in base.rglob("*"):
  if not p.is_file() or p.suffix not in suffixes:continue
  rel=p.relative_to(root)
  if any(part.endswith("-tmp") for part in rel.parts):continue
  assert not p.is_symlink() and p.resolve().is_relative_to(root)
  selected[rel.as_posix()]=p
extras=[out/"frozen-input-snapshot-at-storage-block-v2.zip",root/"experiments/prefix_io_v1/execution_state.json",root/"experiments/prefix_io_v1/configs/permissions.yaml",root/"experiments/prefix_io_v1/gpu-budget-ledger.json",root/"docs/prefix_io_v1/SERVER08_P3_P316_STORAGE_BLOCK_REPORT.md",root/"experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316_v2.py",root/"experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316_v3.py",root/"tests/prefix_io_v1_p3_closeout_v2/test_closeout_schema.py",root/"tests/prefix_io_v1_p3_closeout_v3/test_closeout_schema.py"]
for p in extras:
 assert p.is_file();selected[p.relative_to(root).as_posix()]=p
archive=out/"server08-p3-16-blocked-checkpoint-evidence-v3.zip";manifest=out/"server08-p3-16-blocked-checkpoint-evidence-v3-manifest.json";assert not archive.exists() and not manifest.exists()
rows=[];started=time.monotonic()
with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for rel,p in sorted(selected.items()):
  entry="project/"+rel;before=p.stat();h=hashlib.sha256();size=0
  with p.open("rb") as f,z.open(entry,"w",force_zip64=True) as w:
   for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk);w.write(chunk);size+=len(chunk)
  after=p.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  assert size==before.st_size
  rows.append(dict(entry=entry,source_path=str(p),bytes=size,sha256=h.hexdigest()))
  assert archive.stat().st_size<=reserve and shutil.disk_usage(primary).free>=8*1024**3
 inner=dict(schema_version=1,status="VERIFIED_P3_BLOCKED_CHECKPOINT_NOT_PHASE_COMPLETION",p3_phase_closed=False,p4_enabled=False,files=rows,entry_files=len(rows),model_and_private_cache_payload_included=False,binary_runtime_libraries_included=False,source_snapshot="project/artifacts/prefix_io_v1/server08-p3-16/frozen-input-snapshot-at-storage-block-v2.zip",CPU_unique_passed=1613,CPU_skipped=16,GPU_P316_runs=29,remaining_GPU_runs=8,source_lock09_sha256=hashlib.sha256((out/"execution-lock-09-at-storage-block.json").read_bytes()).hexdigest())
 z.writestr("evidence-file-manifest.json",json.dumps(inner,indent=2,ensure_ascii=False).encode())
h=hashlib.sha256()
with archive.open("rb") as f:
 for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
result=dict(**inner,archive=dict(path=str(archive.relative_to(root)),bytes=archive.stat().st_size,sha256=h.hexdigest()),seconds=time.monotonic()-started,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_before=free,free_after=shutil.disk_usage(primary).free,new_GPU_workloads_run=0)
with manifest.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(status=result["status"],archive=result["archive"],manifest=dict(path=str(manifest.relative_to(root)),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),metadata_entries=len(rows),metadata_uncompressed_bytes=sum(x["bytes"] for x in rows),free_after=result["free_after"],seconds=result["seconds"])))
