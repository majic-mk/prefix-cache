"""CPU-only read-only final audit; append evidence and an immutable ledger snapshot."""
import argparse, hashlib, json, pathlib, shutil, subprocess, time
def ref(p, root):
    b=p.read_bytes()
    return {"path":str(p.relative_to(root)),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
def put(p, data):
    with p.open("xb") as f: f.write(data)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project",required=True); args=ap.parse_args()
    root=pathlib.Path(args.project).resolve(strict=True)
    d=root/"artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004"
    out=root/"artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004"
    label="server12-c5-native-common-cost-gpu01"
    guard_path=root/"experiments/prefix_io_v1/runs"/label/"result.json"
    guard=json.loads(guard_path.read_text())
    assert guard["exit"]==0 and guard["child_exit"]==0 and guard["session_drained"] is True
    ledger_path=root/"experiments/prefix_io_v1/gpu-budget-ledger.json"; ledger_bytes=ledger_path.read_bytes()
    ledger=json.loads(ledger_bytes); assert ledger["active_reservation"] is None
    events=[e for e in ledger["events"] if e.get("label")==label]
    assert len(events)==1 and events[0]["elapsed_seconds"]==guard["elapsed_seconds"]
    assert events[0]["reservation_id"]=="bbe0d5e7cc084147985d4d413301e2ce"
    site=json.loads((d/"SITE_SOURCE_LOCK.json").read_text())
    rows=[row for row in site["files"] if str(row["path"]).startswith(str(d.relative_to(root))+"/")]
    for row in rows: assert ref(root/row["path"],root)==row
    session=guard["session_id"]; members=[]
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try:
            stat=(p/"stat").read_text(); tail=stat.rsplit(")",1)[1].split()
            if int(tail[3])==session: members.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    assert not members
    query=["nvidia-smi","--query-gpu=uuid,name,driver_version,memory.total,memory.free,utilization.gpu","--format=csv,noheader,nounits"]
    gpu=subprocess.run(query,capture_output=True,text=True,timeout=20)
    processes=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=20)
    assert gpu.returncode==processes.returncode==0
    status=subprocess.run(["git","status","--porcelain=v1","--untracked-files=no"],cwd=root,capture_output=True,text=True,timeout=20)
    assert status.returncode==0 and not status.stdout.strip()
    snapshot=out/"FINAL_GPU_BUDGET_LEDGER_SNAPSHOT.json"; put(snapshot,ledger_bytes)
    total=8*3600; primary=shutil.disk_usage(root); assert primary.free>=8*1024**3
    summary={"schema":"server12_completed_gpu_final_state_v1","status":"PASS_COMPLETED_GUARD_SOURCE_LEDGER_AUDIT",
        "utc_unix":time.time(),"job":label,"guard_ref":ref(guard_path,root),
        "GPU_operations_this_action":0,"nvidia_smi_queries_are_resource_observation":True,
        "gpu_query_argv":query,"gpu_query_stdout":gpu.stdout,"compute_processes_stdout":processes.stdout,
        "session_id":session,"current_session_members":members,
        "site_frozen_direct_refs_verified":len(rows),"site_lock_ref":ref(d/"SITE_SOURCE_LOCK.json",root),
        "ledger_ref":ref(ledger_path,root),"ledger_snapshot_ref":ref(snapshot,root),
        "completed_job_elapsed_seconds":guard["elapsed_seconds"],"completed_job_events":len(events),
        "gpu_wall_seconds":ledger["gpu_wall_seconds"],"budget_total_seconds":total,
        "remaining_gpu_budget_seconds":total-ledger["gpu_wall_seconds"],"active_reservation":None,
        "ledger_unchanged_since_public_receipt":ref(ledger_path,root)==json.loads((d/"REAL_CANONICAL_RECEIPT_SUMMARY.json").read_text())["ledger_after_ref"],
        "primary_free_bytes":primary.free,"primary_total_bytes":primary.total,"server_tracked_git_status":status.stdout,
        "downloads_this_stage":0,"data_deletions_this_stage":0,"system_driver_package_modifications_this_stage":0}
    assert summary["ledger_unchanged_since_public_receipt"]
    assert ledger_path.read_bytes()==ledger_bytes
    p=out/"FINAL_SERVER_GPU_STATE.json";put(p,(json.dumps(summary,indent=2,sort_keys=True)+"\n").encode())
    print(json.dumps(summary,sort_keys=True))
if __name__=="__main__":main()
