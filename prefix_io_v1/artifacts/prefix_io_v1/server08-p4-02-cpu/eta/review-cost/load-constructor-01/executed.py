from pathlib import Path
import sys,json,hashlib
from dataclasses import asdict
sys.path.insert(0,str(Path.cwd()/'third_party/work/prefix-io-p4-02-cpu/src'))
from prefix_io_control.p4_load_observation import SchedulerLoadObservation
p=Path('third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_load_observation.py')
out=Path('artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-cost/load-constructor-01');out.mkdir(exist_ok=False)
(out/'p4_load_observation.py').write_bytes(p.read_bytes())
base=dict(run_id='cpu-review',sequence=1,captured_ns=100,source_sha256='a'*64,
 requests=2,prefill_requests=0,decode_requests=2,prefill_tokens=0,decode_tokens=2,
 min_context_tokens=10,max_context_tokens=10,total_context_tokens=20,geometry_sha256='b'*64)
rows=[]
for name,delta in [
 ('prefill_request_without_prefill_token',dict(prefill_requests=1)),
 ('decode_request_without_decode_token',dict(prefill_requests=2,prefill_tokens=2,decode_requests=1,decode_tokens=0)),
 ('not_all_scheduled_requests_have_a_phase',dict(decode_requests=1,decode_tokens=1)),
 ('fewer_tokens_than_nonempty_decode_requests',dict(requests=3,decode_requests=3,decode_tokens=2,total_context_tokens=30))]:
 args=dict(base,**delta)
 try:v=SchedulerLoadObservation(**args);rows.append(dict(name=name,accepted=True,value=asdict(v),production_gpu_state_qualified=v.production_gpu_state_qualified))
 except Exception as e:rows.append(dict(name=name,accepted=False,error=str(e)))
r=dict(status='INDEPENDENT_CPU_LOAD_SCALAR_REVIEW',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
 results=rows,gpu_workloads_run=0,observation_is_active_GPU_state=False)
(out/'result.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
