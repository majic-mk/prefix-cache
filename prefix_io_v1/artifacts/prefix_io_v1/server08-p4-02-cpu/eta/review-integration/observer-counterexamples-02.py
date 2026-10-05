from pathlib import Path
import sys,json,hashlib,importlib.abc
root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
out=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-integration/observer-counterexamples-02'
out.mkdir(parents=True,exist_ok=False)
attempts=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','py_kvcache','vllm','cupy','numpy'):
   attempts.append(name);raise RuntimeError('CPU review backend guard '+name)
sys.meta_path.insert(0,Guard())
sys.path.insert(0,str(root/'third_party/work/prefix-io-p4-02-cpu/src'))
from types import SimpleNamespace as NS
from dataclasses import asdict
from prefix_io_control.p4_native_window_journal import NativeWindowJournal
from prefix_io_control.p4_gpu_step_observation import ForwardObservationProbe,GPUClockReference,attach_forward_observer
from prefix_io_control.p4_production_table_contract import TableContext
from prefix_io_control import p4_gpu_step_observation as forward
paths=['p4_native_window_journal.py','p4_gpu_step_observation.py']
sources={p:hashlib.sha256((root/'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control'/p).read_bytes()).hexdigest() for p in paths}
assert sources['p4_native_window_journal.py']=='4bab9245b9134f622c109867b8bbb293164a81951ce2c8c542df340378ce5cbd'
assert sources['p4_gpu_step_observation.py']=='6ce4b3a48f81c64dcf0a394dd96a9372749a0ec576df4caca49917e32f85d3f3'
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json'
ledger_before=hashlib.sha256(ledger.read_bytes()).hexdigest()
class Clock:
 value=0
 def __call__(self):return self.value
def case(at,invalid=False):
 c=Clock();j=NativeWindowJournal('independent-cpu','f'*64,enabled=True,clock=c,max_events=32)
 j.bind();j.publish_owner_frame();c.value=at;j.accepted('h2d',7,8)
 c.value=18;j.completed('h2d',7,8);c.value=30;j.publish_owner_frame()
 if invalid:j.invalidate('injected real accounting invalidation')
 w=NS(scope='model_forward',run_id='independent-cpu',reference_valid=True,fallback_used=False,
      clock_domain_valid=True,journal_complete=True,mapped_wall_clock_domain='monotonic_ns_cuda_reference',
      start_ns=10,end_ns=20)
 a=j.attribute(w,'h2d',1,8)
 return dict(attribution=asdict(a),native_valid=j.valid,published_valid=j.published()[2],qualification=a.production_qualified,
             event_trace=[asdict(e) for e in j.published()[0]],frames=[asdict(f) for f in j.published()[1]])
boundary=case(10);invalid=case(12,True)
class Event:
 def __init__(self,ms):self.ms=ms
 def record(self):pass
 def query(self):return True
 def elapsed_time(self,other):return other.ms-self.ms
ctx=TableContext('independent-cpu','a'*64,'GPU-CPU-FIXTURE','b'*64,'c'*64,'d'*64,
 'eager-model-forward','torch-fixture','cuda-fixture','driver-fixture',8,128,64,200,
 'existing_io_plus_delta','paired_residual_margin')
spy=[];ret=object();r=NS(speculative_config=None,use_async_scheduling=False,_profile_step=1,
 parallel_config=NS(pipeline_parallel_size=1,data_parallel_size=1,tensor_parallel_size=1),
 input_batch=NS(num_reqs=1,req_ids=['id'],num_computed_tokens_cpu=[128],num_prompt_tokens=[128]))
r._prepare_inputs=lambda *a,**k:spy.append('prepare') or 'prepared'
r._model_forward=lambda *a,**k:spy.append('forward') or ret
events=[Event(1),Event(2)]
p=ForwardObservationProbe(ctx,'e'*64,enabled=True,event_factory=lambda **k:events.pop(0),
 reference=GPUClockReference(Event(0),1,True,False,'monotonic_ns_cuda_reference',True),max_pending=2)
remove=attach_forward_observer(r,p)
r._prepare_inputs(NS(num_scheduled_tokens={'id':1},scheduled_spec_decode_tokens={}))
originalclock=forward.time.monotonic_ns;clocks=iter([100,50])
try:
 forward.time.monotonic_ns=lambda:next(clocks)
 assert r._model_forward() is ret
finally:forward.time.monotonic_ns=originalclock;remove()
observation=p.resolve_ready()[0]
backwards=asdict(observation);backwards['production_qualified']=observation.production_qualified
result=dict(boundary_acceptance=boundary,inherited_accounting_invalidation=invalid,host_clock_backwards=backwards,
 original_model_calls=spy,backend_import_attempts=attempts,gpu_operations=0,sources=sources,
 ledger_before=ledger_before,ledger_after=hashlib.sha256(ledger.read_bytes()).hexdigest(),
 findings=dict(boundary_double_count=boundary['attribution']['existing_io'][2]['nbytes']==8 and
                boundary['attribution']['added_io'][2]['nbytes']==8,
 invalid_native_still_attributed=invalid['attribution']['status']=='DIAGNOSTIC_SINGLE_STAGE_ONLY' and not invalid['native_valid'],
 host_backwards_still_clock_valid=observation.clock_domain_valid and observation.host_end_ns<observation.host_start_ns))
assert not attempts and result['ledger_before']==result['ledger_after']
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
