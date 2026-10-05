from pathlib import Path
import sys,json,hashlib,shutil,importlib.abc
from dataclasses import asdict
root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
out=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-cost/v2-independent-02'
out.mkdir(parents=True,exist_ok=False)
attempts=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','py_kvcache','vllm','cupy','numpy'):
   attempts.append(name);raise RuntimeError('CPU review backend guard '+name)
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(root/'third_party/work/prefix-io-p4-02-cpu/src'))
from prefix_io_control.p4_production_table_contract import EvidenceRef,TableContext
from prefix_io_control.p4_paired_measurement_verifier import load_verification_plan,build_paired_cell_candidate,MEASUREMENT_ROLES
from prefix_io_control.p4_verified_cost_loader import load_semantically_verified_table
source=root/'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py'
source_sha=hashlib.sha256(source.read_bytes()).hexdigest();assert source_sha=='3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac'
fixture=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/cost-verifier/fixtures-v2-01/continuous-context-full128-selected1'
fixture_before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.iterdir() if p.is_file()}
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json';ledger_sha=hashlib.sha256(ledger.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(obj,sort_keys=True)+'\n',encoding='utf-8')
def ref(p):
 b=p.read_bytes();return dict(path=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def load_actual(d):
 r=read(d/'result.json')
 return load_semantically_verified_table(d,r['candidate_ref']['path'],expected_context=TableContext.from_mapping(r['context']),
  qualification_ref=EvidenceRef.from_mapping(r['qualification_ref']),
  expected_verifier_ref=EvidenceRef.from_mapping(r['verifier_ref']),
  plan_path=r['plan_ref']['path'],expected_plan_ref=EvidenceRef.from_mapping(r['plan_ref']))
baseline=load_actual(fixture);results={'original_fixture':{'status':baseline.status,'gpu_verified':baseline.gpu_verified,'production_qualified':baseline.production_qualified}}
def mutate_case(name):
 d=out/name;d.mkdir()
 for p in fixture.iterdir():
  if p.is_file():shutil.copy2(p,d/p.name)
 for arm in ('baseline','action'):
  wrapper=read(d/(arm+'_wrapper.json'));observations=read(d/(arm+'_observations.json'))
  for run in wrapper['runs']:
   path=d/run['complete_trace_ref']['path'];trace=read(path)
   if name=='concentrated-128-output-single-active':
    for step in trace['steps']:step['outputs']=[]
    trace['steps'][2]['outputs']=trace['outputs']
    run['selected_output_tokens']=128
    for row in observations['windows']:
     if row['pair_id']==run['pair_id']:row['output_tokens']=128 if row['step_offset']==2 else 0
   elif name=='selected-multi-token-request':
    trace['steps'][2]['outputs']=trace['outputs']
    run['selected_output_tokens']=128
    for row in observations['windows']:
     if row['pair_id']==run['pair_id'] and row['step_offset']==2:row['output_tokens']=128
   elif name=='nonselected-pure-output-missing':
    trace['steps'][5]['outputs']=[]
   elif name=='nonselected-batch-over-run-request':
    trace['steps'][5]['load']['active_decode']=2
    trace['steps'][5]['load']['batch']=2
   elif name=='explicit-prefill-prelude':
    trace['steps'][0]['step_kind']='prefill'
    trace['steps'][0]['load']['active_decode']=0
    trace['steps'][0]['load']['prefill_tokens']=1
    for row in observations['windows']:
     if row['pair_id']==run['pair_id'] and row['step_offset']==0:row['load']=trace['steps'][0]['load']
   elif name=='omit-unselected-complete-step':
    trace['steps'].pop(5)
   elif name=='boolean-gpu-duration':
    trace['steps'][2]['timing']['gpu_elapsed_ns']=True
    for row in observations['windows']:
     if row['pair_id']==run['pair_id'] and row['step_offset']==2:row['timing']['gpu_elapsed_ns']=True
   elif name=='invent-selected-native-ordinal':
    trace['steps'][2]['native_step_ordinal']+=10
    for row in observations['windows']:
     if row['pair_id']==run['pair_id'] and row['step_offset']==2:row['native_step_ordinal']+=10
   elif name=='false-native-full-decode-label':
    trace['origin']='native_gpu_recording'
    for step in trace['steps']:step['timing']['timing_scope']='full_decode_step'
   elif name=='hidden-unselected-io':
    trace['steps'][5]['new_io'][2]={'ops':1,'bytes':8}
   write(path,trace);run['complete_trace_ref']=ref(path)
  if name=='false-native-full-decode-label':
   wrapper['origin']='native_gpu_recording';observations['origin']='native_gpu_recording'
   for row in observations['windows']:row['timing']['timing_scope']='full_decode_step'
  write(d/(arm+'_wrapper.json'),wrapper);write(d/(arm+'_observations.json'),observations)
 if name=='false-native-full-decode-label':
  plan=read(d/'plan.json');plan['timing_contract']['timing_scope']='full_decode_step';write(d/'plan.json',plan)
 planref=EvidenceRef.from_mapping(ref(d/'plan.json'))
 raw={role:EvidenceRef.from_mapping(ref(d/(role+'.json'))) for role in MEASUREMENT_ROLES}
 old=read(d/'prepared-candidate.json');c=old['cells'][0]
 geometry={k:c[k] for k in ('cell_id','load','existing_io','stage','physical_bytes')}
 try:
  plan=load_verification_plan(d,'plan.json',expected_plan_ref=planref)
  prepared=build_paired_cell_candidate(d,cell_geometry=geometry,raw_refs=raw,expected_plan=plan,analysis_path='independent-analysis.json')
  candidate={'schema_version':1,'scope':'production_candidate','context':old['context'],'cells':[prepared.to_candidate_mapping()]}
  write(d/'independent-candidate.json',candidate)
  qualification=read(d/'prepared-qualification.json');qualification['candidate_ref']=ref(d/'independent-candidate.json')
  qualification['measurement_refs']=list(candidate['cells'][0]['measurement_refs'].values())
  write(d/'independent-qualification.json',qualification)
  actual=load_semantically_verified_table(d,'independent-candidate.json',expected_context=plan.context,
   qualification_ref=EvidenceRef.from_mapping(ref(d/'independent-qualification.json')),
   expected_verifier_ref=EvidenceRef.from_mapping(ref(d/'verifier.py')),plan_path='plan.json',expected_plan_ref=planref)
  cost=actual.verification.cells[0].cost
  return {'accepted':True,'status':actual.status,'production_qualified':actual.production_qualified,'gpu_verified':actual.gpu_verified,
   'production_lookup':actual.lookup(cost.load_signature,cost.existing_io,cost.stage,cost.physical_bytes),
   'cost':asdict(cost),'selected_output_tokens':read(d/'action_wrapper.json')['runs'][0]['selected_output_tokens'],
   'actual_selected_outputs':read(d/'complete-action-p0.json')['steps'][2]['outputs'],
   'selected_load':read(d/'complete-action-p0.json')['steps'][2]['load']}
 except Exception as e:return {'accepted':False,'error_type':type(e).__name__,'error':str(e)}
for case in ['selected-multi-token-request','nonselected-pure-output-missing','nonselected-batch-over-run-request','explicit-prefill-prelude','concentrated-128-output-single-active','omit-unselected-complete-step','boolean-gpu-duration','invent-selected-native-ordinal','false-native-full-decode-label','hidden-unselected-io']:
 results[case]=mutate_case(case)
selected2root=fixture.parent/'continuous-context-full128-selected2'
selected2=load_actual(selected2root)
results['original_two_request_fixture']={'status':selected2.status,'production_qualified':selected2.production_qualified,'gpu_verified':selected2.gpu_verified}
for bad in ['selected-multi-token-request','nonselected-pure-output-missing','nonselected-batch-over-run-request','concentrated-128-output-single-active','omit-unselected-complete-step','boolean-gpu-duration','invent-selected-native-ordinal','hidden-unselected-io']:
 assert results[bad]['accepted'] is False,(bad,results[bad])
for good in ['explicit-prefill-prelude','false-native-full-decode-label']:
 assert results[good]['accepted'] and not results[good]['production_qualified'] and results[good]['production_lookup'] is None
# The direct old accepted candidate must now fail without mutation or ref rebinding.
oldcase=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-cost/v2-independent-01/concentrated-128-output-single-active'
oldbytes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in oldcase.iterdir() if p.is_file()}
try:
 load_semantically_verified_table(oldcase,'independent-candidate.json',expected_context=TableContext.from_mapping(read(oldcase/'independent-candidate.json')['context']),
  qualification_ref=EvidenceRef.from_mapping(ref(oldcase/'independent-qualification.json')),
  expected_verifier_ref=EvidenceRef.from_mapping(ref(oldcase/'verifier.py')),
  plan_path='plan.json',expected_plan_ref=EvidenceRef.from_mapping(ref(oldcase/'plan.json')))
 raise AssertionError('old packed candidate accepted')
except ValueError as exc:
 results['old_counterexample_direct_replay']={'accepted':False,'error_type':type(exc).__name__,'error':str(exc)}
assert oldbytes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in oldcase.iterdir() if p.is_file()}
assert not attempts and fixture_before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.iterdir() if p.is_file()}
result={'cases':results,'backend_import_attempts':attempts,'gpu_operations':0,'source_sha256':source_sha,
 'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==source_sha,'original_fixture_unchanged':True,
 'ledger_before':ledger_sha,'ledger_after':hashlib.sha256(ledger.read_bytes()).hexdigest()}
write(out/'result.json',result)
print(json.dumps(result))
