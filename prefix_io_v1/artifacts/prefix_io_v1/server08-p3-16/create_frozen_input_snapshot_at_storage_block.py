from pathlib import Path
import json,hashlib,zipfile,shutil,datetime,time
root=Path(".").resolve();out=root/"artifacts/prefix_io_v1/server08-p3-16";lp=out/"execution-lock-08.json"
lock=json.loads(lp.read_text());assert len(lock)==2599
free=shutil.disk_usage(root/"experiments/prefix_io_v1/runs").free
assert free-256*1024*1024>=8*1024**3
archive=out/"frozen-input-snapshot-at-storage-block.zip";manifest=out/"frozen-input-snapshot-at-storage-block.json"
assert not archive.exists() and not manifest.exists()
canon={}
for key,expected in lock.items():
 p=Path(key).resolve();p.relative_to(root)
 d=canon.setdefault(str(p),dict(path=p,sha256=expected,aliases=[]))
 assert d["sha256"]==expected;d["aliases"].append(key)
rows=[];start=time.monotonic()
with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for item in sorted(canon.values(),key=lambda x:str(x["path"])):
  p=item["path"];before=p.stat();entry="project_inputs/"+p.relative_to(root).as_posix();h=hashlib.sha256();size=0
  with p.open("rb") as f,z.open(entry,"w",force_zip64=True) as w:
   for chunk in iter(lambda:f.read(1024*1024),b""):
    h.update(chunk);w.write(chunk);size+=len(chunk)
  after=p.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  assert h.hexdigest()==item["sha256"] and size==before.st_size
  rows.append(dict(entry=entry,source_path=str(p),source_aliases=item["aliases"],bytes=size,sha256=h.hexdigest()))
 inner=dict(schema_version=1,status="EXACT_FROZEN_INPUT_SNAPSHOT_AT_STORAGE_BLOCK_NOT_P3_ACCEPTANCE",input_keys=len(lock),unique_source_paths=len(rows),files=rows,execution_lock_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(),p3_phase_closed=False,model_payload_and_private_cache_included=False)
 z.writestr("snapshot-entries.json",json.dumps(inner,indent=2,ensure_ascii=False).encode())
h=hashlib.sha256()
with archive.open("rb") as f:
 for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
d=dict(**inner,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=dict(path=str(archive.relative_to(root)),bytes=archive.stat().st_size,sha256=h.hexdigest()),all_source_files_stat_and_bytes_stable=True,seconds=time.monotonic()-start,free_bytes_before=free,free_bytes_after=shutil.disk_usage(root/"experiments/prefix_io_v1/runs").free,new_gpu_workloads_run=0)
with manifest.open("x",encoding="utf-8") as f:json.dump(d,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(status=d["status"],input_keys=len(lock),unique_source_paths=len(rows),archive=d["archive"],snapshot_manifest=dict(path=str(manifest.relative_to(root)),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),free_after=d["free_bytes_after"],seconds=d["seconds"])))
