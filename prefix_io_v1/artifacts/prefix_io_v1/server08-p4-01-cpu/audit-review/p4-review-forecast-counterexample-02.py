"""Independent CPU publication-forecast boundary counterexample. No backend imports."""
import os,sys,pathlib,json,hashlib,datetime
from dataclasses import replace
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
root=pathlib.Path.cwd()
class NoGPUImports:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0]in("torch","vllm","py_kvcache","cupy"):raise RuntimeError("runtime import prohibited")
sys.meta_path.insert(0,NoGPUImports());sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-01-cpu/src"))
from prefix_io_control.p4_types import P4Config,SystemSnapshot,WorkDescriptor,ReleaseWitness,WaitingTarget
from prefix_io_control.p4_policy import P4Policy
from prefix_io_control.p4_bridge import NativeP4Bridge,NativePublication
from prefix_io_control.dependencies import Parent,Resource,ResourceId
caps=frozenset(("native_ready_work","owner_release_protocol","restore_parent_completion"))
p1=Parent("audit-forecast",1,1,0,0,False,False,1,None)
p2=Parent("audit-forecast",2,1,0,0,False,False,1,None)
identity=ResourceId("audit-forecast","restore-parent",0,1,1)
w1=WorkDescriptor("audit-forecast",1,1,0,"ssd_read",16,1,99,False)
w2=WorkDescriptor("audit-forecast",1,2,0,"ssd_read",16,0,99,False)
target=WaitingTarget("load1",0,99,restore_parent_id=1)
s=SystemSnapshot("audit-forecast",1,100,caps,parents=(p1,p2),targets=(target,),generations=((identity,1),))
wit=ReleaseWitness("load1",Resource(identity,16,0,frozenset((1,)),"observed_blocking"),(p1,),"restore",(w1.work_id,),"original-native-parent-restore-fence",None,completed_restore_parent_id=1)
current=NativePublication(s,(w2,w1),(wit,))
pub_p=replace(p1,completion_estimate_ns=150)
publication=NativePublication(replace(s,parents=(pub_p,p2)),current.works,(replace(wit,parents=(pub_p,),estimated_unblock_ns=150),))
b=NativeP4Bridge(P4Policy("audit-forecast",P4Config("dependency_only",1000,1000)))
b.bind();b.epoch(("bounded-fixture",),100)
result={"schema_version":1,"scope":"CPU_ONLY_PUBLICATION_BOUNDARY_REVIEW_NOT_GPU","GPU_runs":0,"native_forecast":None,"publication_forecast":150,"native_order":[list(w.work_id)for w in current.works]}
try:
 b.publish(publication,current,now_ns=100)
 result["publication_accepted"]=True
 result["selected_order"]=[list(x)for x in b.order(current,current.works,now_ns=100)]
 result["reason"]=b.last_reason
except (ValueError,TypeError)as e:
 result["publication_accepted"]=False;result["rejection"]=str(e)
result["boundary_closed"]=result["publication_accepted"]is False
result["source_inputs"]=[{"path":str((root/"third_party/work/prefix-io-p4-01-cpu/src/prefix_io_control"/n).relative_to(root)),"sha256":hashlib.sha256((root/"third_party/work/prefix-io-p4-01-cpu/src/prefix_io_control"/n).read_bytes()).hexdigest()}for n in("p4_bridge.py","p4_policy.py","p4_types.py")]
result["guard"]={"cuda_visible_devices":"","forbidden_runtime_imports_present":any(n.split(".")[0]in("torch","vllm","py_kvcache","cupy")for n in sys.modules)}
assert result["boundary_closed"] is True
result["created_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
out=root/"artifacts/prefix_io_v1/server08-p4-01-cpu/audit-review/p4-review-forecast-counterexample-02.json"
with out.open("x",encoding="utf-8")as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps(result,ensure_ascii=False))
