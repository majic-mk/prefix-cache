from pathlib import Path
import json,hashlib,time,datetime,subprocess,os
out=Path("artifacts/prefix_io_v1/server08-p3-16");regp=Path("experiments/prefix_io_v1/runs/server08-p3-14-source/registration.json");reg=json.loads(regp.read_text())
b=json.loads(Path("experiments/prefix_io_v1/gpu-budget-ledger.json").read_text());assert b["active_reservation"] is None
gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True);assert not gpu.strip()
root=Path(reg["source_root"]).resolve();mp=Path(reg["source_manifest"])
assert hashlib.sha256(mp.read_bytes()).hexdigest()==reg["source_manifest_sha256"]=="c03381abb29e21b54a62a4815892553b58e4ef39cc8087761cc4d173cf75d274"
manifest=json.loads(mp.read_text());assert len(manifest)==reg["file_count"]==3048
started=time.monotonic();total=0;seen=set()
for row in manifest:
 p=(root/row["path"]).resolve();p.relative_to(root);assert p.is_file() and p not in seen;seen.add(p)
 s=p.stat();assert s.st_size==row["bytes"]
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
 assert h.hexdigest()==row["sha256"],str(p);total+=s.st_size
assert total==reg["total_bytes"]==2796552192
actual={p.resolve() for p in root.rglob("*.bin")};assert actual==seen
selected={e["session_id"] for e in b["events"] if e["label"].startswith("server08-p3-16-")}
live=[]
for p in Path("/proc").iterdir():
 if not p.name.isdigit():continue
 try:
  sid=os.getsid(int(p.name))
 except ProcessLookupError:continue
 if sid in selected:live.append(dict(pid=int(p.name),sid=sid))
assert not live
lockp=out/"execution-lock-11-selected-hot-plans.json";lock=json.loads(lockp.read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in lock.items())
ginfo=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader"],text=True)
assert ginfo.split(",")[0].strip()=="GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8"
assert int(ginfo.split(",")[-1].strip().replace("%",""))==0
events=[e for e in b["events"] if e["label"].startswith("server08-p3-16-")]
success=[];failed=[]
for e in events:
 w=json.loads((Path(e["evidence"])/"result.json").read_text())
 assert w["session_drained"] and not w["timed_out"]
 row=dict(label=e["label"],wrapper=dict(path=str(Path(e["evidence"])/"result.json"),sha256=hashlib.sha256((Path(e["evidence"])/"result.json").read_bytes()).hexdigest()),gpu_seconds=e["elapsed_seconds"],session_id=e["session_id"],exit=w["exit"],child_exit=w["child_exit"])
 (success if w["exit"]==0 and w["child_exit"]==0 else failed).append(row)
assert len(events)==38 and len(success)==37 and len(failed)==1
assert failed[0]["label"]=="server08-p3-16-cap960-l1-mixed-U-01"
from shutil import disk_usage
assert disk_usage("experiments/prefix_io_v1/runs").free>=8*1024**3
assert b["gpu_wall_seconds"]<8*3600
d=dict(schema_version=1,status="PASS_REGISTERED_SOURCE_BYTES_AND_GPU_IDLE_FINAL_P3",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),file_count=len(manifest),bytes_hashed=total,seconds=time.monotonic()-started,source_root=str(root),source_registration=dict(path=str(regp),sha256=hashlib.sha256(regp.read_bytes()).hexdigest()),source_manifest=dict(path=str(mp),sha256=hashlib.sha256(mp.read_bytes()).hexdigest()),all_file_bytes_match=True,no_extra_bin_files=True,gpu_active_reservation=None,gpu_processes=gpu,selected_p316_sessions_checked=len(selected),selected_p316_sessions_live=live,frozen_lock=dict(path=str(lockp),sha256=hashlib.sha256(lockp.read_bytes()).hexdigest()),locked_inputs=len(lock),locked_input_bytes_match=True,final_gpu_status=ginfo,primary_free_bytes=disk_usage("experiments/prefix_io_v1/runs").free,p316_successful_stages=success,p316_failed_attempts=failed,p316_attempt_count=len(events),p316_success_count=len(success),p316_gpu_seconds=sum(e["elapsed_seconds"] for e in events),cumulative_gpu_seconds=b["gpu_wall_seconds"],remaining_gpu_seconds=8*3600-b["gpu_wall_seconds"],new_gpu_workloads_run=0,model_payload_read=False,private_cache_merge_applied=True,system_changes=False)
with (out/"source-preservation-final-p3.json").open("x") as f:json.dump(d,f,indent=2)
bp=out/"gpu-budget-final-p3.json"
with bp.open("xb") as f:f.write(Path("experiments/prefix_io_v1/gpu-budget-ledger.json").read_bytes())
print(json.dumps({k:v for k,v in d.items() if k not in ["p316_successful_stages","p316_failed_attempts"]}))
