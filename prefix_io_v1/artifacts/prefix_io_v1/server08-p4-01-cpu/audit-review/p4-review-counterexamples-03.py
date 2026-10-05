"""Independent CPU-only negative contract examples. No native/GPU execution."""
import os,sys,pathlib,json,hashlib,datetime
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root=pathlib.Path.cwd()
overlay=root/"third_party/work/prefix-io-p4-01-cpu/src"
class NoGPUImports:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in ("torch","vllm","py_kvcache","cupy"):
            raise RuntimeError("GPU/runtime library import prohibited by independent audit")
sys.meta_path.insert(0,NoGPUImports())
sys.path.insert(0,str(overlay))
from prefix_io_control.p4_types import *
from prefix_io_control.p4_policy import P4Policy
from prefix_io_control.dependencies import ResourceId,Resource,Parent
own=ResourceId("audit-run","pool",0,1,1)
foreign=ResourceId("foreign-run","pool",0,2,1)
parent=Parent("audit-run",1,1,0,0,False,False,1,150)
caps=frozenset(("native_ready_work","owner_release_protocol","gpu_owner_generation","gpu_active_refs","gpu_protectors"))
target=WaitingTarget("target",0,1,need_gpu_bytes=64)
result={"schema_version":1,"scope":"CPU_ONLY_NEGATIVE_CONTRACT_REVIEW_NOT_GPU","GPU_runs":0}
try:
    snapshot=SystemSnapshot("audit-run",1,100,caps,parents=(parent,),targets=(target,),
                            generations=((own,1),(foreign,1)))
    work=WorkDescriptor("audit-run",1,1,1,"ssd_write",16,0,1,False,generation=1,resource_identity=foreign)
    witness=ReleaseWitness("target",Resource(own,64,0,frozenset((1,)),"observed_blocking"),
        (parent,),"gpu",(work.work_id,),"fixture-owner",150)
    choice=P4Policy("audit-run",P4Config("dependency_only",1000,1000)).choose(
        snapshot,(work,),(witness,),now_ns=100,expected_epoch=1)
    result["cross_run_physical_work"]={"rejected":False,"choice_action":choice.action,
        "expected":"reject or native_fallback; never selected"}
except (ValueError,TypeError) as error:
    result["cross_run_physical_work"]={"rejected":True,"error_type":type(error).__name__}
parents=(parent,Parent("audit-run",2,1,0,0,False,False,1,150))
ids=(own,ResourceId("audit-run","pool",0,2,1))
snapshot=SystemSnapshot("audit-run",2,100,caps,parents=parents,
    targets=(WaitingTarget("target",0,1,need_gpu_bytes=128),),
    generations=tuple((identity,1) for identity in ids))
works=tuple(WorkDescriptor("audit-run",2,i+1,i+1,("h2d","d2h")[i],16,i,1,False)
            for i in range(2))
witnesses=tuple(ReleaseWitness("target",Resource(ids[i],64,0,frozenset((i+1,)),"observed_blocking"),
    (parents[i],),"gpu",(works[i].work_id,),"fixture-owner",150,interference_ns=(11,17)[i])
    for i in range(2))
policy=P4Policy("audit-run",P4Config("dependency_only",1000,1000))
closures=tuple(policy._witness(w,snapshot,works,snapshot.targets[0],100)[0] for w in witnesses)
combined=policy._combine(closures,works)
result["unqualified_joint_interference"]={"actual_ns":combined.interference_ns,
    "expected":None,"supported_joint_cost_source":False,
    "contract_holds":combined.interference_ns is None}
result["source_inputs"]=[{"path":str((overlay/"prefix_io_control"/n).relative_to(root)),
    "sha256":hashlib.sha256((overlay/"prefix_io_control"/n).read_bytes()).hexdigest()}
    for n in ("p4_types.py","p4_policy.py","p4_cost_table.py")]
result["guard"]={"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
    "forbidden_runtime_imports_present":any(n.split(".")[0] in ("torch","vllm","py_kvcache","cupy") for n in sys.modules)}
assert result["cross_run_physical_work"]["rejected"] and result["unqualified_joint_interference"]["contract_holds"]
result["all_counterexamples_closed"]=True
result["created_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
out=root/"artifacts/prefix_io_v1/server08-p4-01-cpu/audit-review/p4-review-counterexamples-03.json"
with out.open("x",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps(result,ensure_ascii=False))
