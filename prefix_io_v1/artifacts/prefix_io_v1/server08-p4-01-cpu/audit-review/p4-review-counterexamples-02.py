"""Independent CPU-only restore/issue-preview contract examples."""
import os,sys,pathlib,json,hashlib,datetime
from dataclasses import replace
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root=pathlib.Path.cwd(); overlay=root/"third_party/work/prefix-io-p4-01-cpu/src"
class NoGPUImports:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","vllm","py_kvcache","cupy"):raise RuntimeError("runtime import prohibited")
sys.meta_path.insert(0,NoGPUImports());sys.path.insert(0,str(overlay))
from prefix_io_control.p4_types import *
from prefix_io_control.p4_policy import P4Policy
from prefix_io_control.dependencies import ResourceId,Resource,Parent
identity=ResourceId("audit-run","pool",0,1,1)
parents=(Parent("audit-run",1,1,0,0,False,False,1,150),Parent("audit-run",2,1,0,0,False,False,1,150))
caps=frozenset(("native_ready_work","owner_release_protocol","restore_parent_completion","gpu_owner_generation","gpu_active_refs","gpu_protectors"))
target=WaitingTarget("target",0,1,restore_parent_id=2)
snapshot=SystemSnapshot("audit-run",1,100,caps,parents=parents,targets=(target,),generations=((identity,1),))
work=WorkDescriptor("audit-run",1,1,1,"ssd_write",16,0,1,False,generation=1,resource_identity=identity)
policy=P4Policy("audit-run",P4Config("dependency_only",1000,1000))
witness=ReleaseWitness("target",Resource(identity,64,0,frozenset((1,)),"observed_blocking"),
 (parents[0],),"restore",(work.work_id,),"fixture-owner",150,completed_restore_parent_id=2)
choice=policy.choose(snapshot,(work,),(witness,),now_ns=100,expected_epoch=1)
result={"schema_version":1,"scope":"CPU_ONLY_NEGATIVE_CONTRACT_REVIEW_NOT_GPU","GPU_runs":0,
 "restore_parent_binding":{"work_parent":1,"work_stage":"ssd_write","claimed_restore_parent":2,
 "choice_action":choice.action,"expected":"native_fallback for unbound unrelated restore"}}
samples={"unaccepted":(replace(work,accepted=False),snapshot),
 "future_created":(replace(work,created_ns=101),snapshot),
 "stale_generation":(work,replace(snapshot,generations=((identity,2),)))}
result["issue_preview"]={}
for name,(w,s) in samples.items():
 p=policy.issue_preview(w,s,now_ns=100,expected_epoch=1)
 result["issue_preview"][name]={"action":p.action,"reason":p.reason,"expected":"native_fallback"}
result["source_inputs"]=[{"path":str((overlay/"prefix_io_control"/n).relative_to(root)),
 "sha256":hashlib.sha256((overlay/"prefix_io_control"/n).read_bytes()).hexdigest()}for n in("p4_types.py","p4_policy.py","p4_cost_table.py")]
result["guard"]={"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
 "forbidden_runtime_imports_present":any(n.split(".")[0]in("torch","vllm","py_kvcache","cupy")for n in sys.modules)}
result["created_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
out=root/"artifacts/prefix_io_v1/server08-p4-01-cpu/audit-review/p4-review-counterexamples-02.json"
with out.open("x",encoding="utf-8")as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps(result,ensure_ascii=False))
