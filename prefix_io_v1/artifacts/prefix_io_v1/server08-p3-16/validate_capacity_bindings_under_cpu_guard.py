from pathlib import Path
import sys,json,builtins,hashlib,datetime,copy
root=Path(".").resolve();out=root/"artifacts/prefix_io_v1/server08-p3-16";sys.path[:0]=[str(root/"third_party/work/py-kvcache-p3-16-cpu"),str(root/"experiments/prefix_io_v1/scripts"),str(root/"src"),str(root/".venv/lib/python3.12/site-packages")]
from qualify_capacity_p316 import verify_gate
records=[]
for domain,run in [("cap960-l1",2),("cap1024-l2",1)]:
 oldpath=out/domain/"mixed-U-01-v2-plan.json";p=out/domain/f"mixed-U-{run:02d}-v3-plan.json";assert not p.exists()
 old=json.loads(oldpath.read_text());new=copy.deepcopy(old);args=new["command"]
 i=args.index("--qualification")+1;old_qualification=args[i];permit=out/domain/"capacity-permit.json";args[i]=str(permit.relative_to(root))
 q=verify_gate(permit,capacity_domain_id=domain,manifest_path=args[args.index("--manifest")+1])
 label=f"server08-p3-16-{domain}-mixed-U-{run:02d}"
 new["label"]=label;new["output"]=str(root/"experiments/prefix_io_v1/runs"/label/"details")
 args[args.index("--label")+1]=label;args[args.index("--output")+1]=new["output"]
 new["qualification_pointer_correction_only"]=dict(original_plan=str(oldpath.relative_to(root)),original_plan_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),old_qualification=old_qualification,qualified_file=str(permit.relative_to(root)),qualification_sha256=hashlib.sha256(permit.read_bytes()).hexdigest(),failed_original_run_retained=domain=="cap960-l1")
 assert not Path(new["output"]).exists()
 with p.open("x") as f:json.dump(new,f,indent=2)
 assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==new["qualification_pointer_correction_only"]["original_plan_sha256"]
 records.append(dict(domain=domain,label=label,status=q["status"],plan=str(p.relative_to(root)),plan_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),qualification_sha256=hashlib.sha256(permit.read_bytes()).hexdigest()))
receipt=out/"actual-capacity-qualification-binding-preflight.json";assert not receipt.exists()
d=dict(status="PASS_ACTUAL_QUALIFIED_CAPACITY_BINDINGS_CPU_ONLY",records=records,source_and_gate_thresholds_unchanged=True,experimental_parameters_unchanged=True,prior_failed_attempt_retained=True,GPU_workloads_run=0,GPU_initialization_forbidden=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with receipt.open("x") as f:json.dump(d,f,indent=2)
lock=json.loads((out/"execution-lock-08.json").read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in lock.items())
for p in [receipt]+[root/x["plan"] for x in records]:lock[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
lockpath=out/"execution-lock-10-qualified-capacity-plans.json";assert not lockpath.exists()
with lockpath.open("x") as f:json.dump(lock,f,indent=2)
print(json.dumps(dict(**d,lock_path=str(lockpath.relative_to(root)),lock_sha256=hashlib.sha256(lockpath.read_bytes()).hexdigest(),locked_inputs=len(lock))))
