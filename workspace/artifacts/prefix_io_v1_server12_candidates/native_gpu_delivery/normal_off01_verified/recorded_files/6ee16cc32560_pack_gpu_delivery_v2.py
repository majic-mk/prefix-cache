"""Archive completed GPU evidence without model/SDK/cache payload copies or deletion."""
import argparse,gzip,hashlib,io,json,pathlib,shutil,tarfile
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project",required=True);root=pathlib.Path(ap.parse_args().project).resolve(strict=True)
 d=root/"artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004"
 out=root/"artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004"
 run=root/"experiments/prefix_io_v1/runs/server12-c5-native-common-cost-gpu01"
 g=root/"artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004"
 assert shutil.disk_usage(root).free>=8*1024**3
 selected={}
 def add(p):
  assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
  assert not any(x.is_symlink() for x in p.parents)
  rel=p.relative_to(root).as_posix();b=p.read_bytes();assert len(b)==p.stat().st_size and len(b)<=32*1024**2
  selected[rel]={"path":rel,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
 excluded_parts={"__pycache__","runtime-cache","private-storage"}
 excluded_names={"GPU_DELIVERY_MANIFEST.json","GPU_ARCHIVE_RESULT.json","GPU_EVIDENCE.tar.gz"}
 for base in (d,out,run,g/"common_candidate"):
  for p in sorted(base.rglob("*")):
   if p.is_file() and p.suffix in (".json",".log",".py",".md",".txt",".yaml") and not excluded_parts.intersection(p.parts) and p.name not in excluded_names:
    add(p)
 for p in sorted(g.glob("*.py")):add(p)
 for rel in (
  "docs/prefix_io_v1/04_CODEX_EXECUTION.md",
  "experiments/prefix_io_v1/configs/permissions.yaml",
  "experiments/prefix_io_v1/scripts/run_gpu_stage.py",
  "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py",
  "artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/CUDA13_SOURCE_INVENTORY.json",
  "artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/CPU_COMPILE_LINK_RESULT.json",
  "artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/SDK_REBIND_RESULT.json",
  "artifacts/prefix_io_v1/server09-cuda13-toolchain-v2-20261001/cuda13_sdk_overlay.py"):
  add(root/rel)
 rows=[selected[k] for k in sorted(selected)];total=sum(r["bytes"] for r in rows);assert len(rows)<1200 and total<=100*1024**2
 manifest={"schema":"server12_gpu_evidence_archive_manifest_v1","files":rows,"count":len(rows),"payload_bytes":total,
  "coverage":"completed_gpu_raw_json_logs_current_source_locks_permissions_grant_full_step_and_public_cost_evidence",
  "excluded_payloads":["model_weights","SDK_binaries","runtime-cache","private-storage"],
  "excluded_payloads_preserved_on_server":True,"GPU_operations_this_action":0,"data_deletions_this_action":0}
 mp=out/"GPU_DELIVERY_MANIFEST.json";mb=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
 with mp.open("xb") as f:f.write(mb)
 archive=out/"GPU_EVIDENCE.tar.gz"
 def info(rel,b):
  t=tarfile.TarInfo(rel);t.size=len(b);t.mode=0o644;t.mtime=0;return t
 with archive.open("xb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0,compresslevel=6) as gz:
   with tarfile.open(fileobj=gz,mode="w|",format=tarfile.PAX_FORMAT) as tar:
    for row in rows:
     b=(root/row["path"]).read_bytes();assert len(b)==row["bytes"] and hashlib.sha256(b).hexdigest()==row["sha256"]
     tar.addfile(info(row["path"],b),io.BytesIO(b))
    tar.addfile(info(mp.relative_to(root).as_posix(),mb),io.BytesIO(mb))
 for row in rows:
  b=(root/row["path"]).read_bytes();assert len(b)==row["bytes"] and hashlib.sha256(b).hexdigest()==row["sha256"]
 ab=archive.read_bytes()
 result={"status":"PASS_COMPLETED_GPU_EVIDENCE_ARCHIVE","archive":{"path":archive.relative_to(root).as_posix(),"bytes":len(ab),"sha256":hashlib.sha256(ab).hexdigest()},
  "manifest":{"path":mp.relative_to(root).as_posix(),"bytes":len(mb),"sha256":hashlib.sha256(mb).hexdigest()},
  "source_files":len(rows),"archive_members":len(rows)+1,"payload_bytes":total,"GPU_operations_this_action":0,"data_deletions_this_action":0}
 with (out/"GPU_ARCHIVE_RESULT.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(result))
if __name__=="__main__":main()
