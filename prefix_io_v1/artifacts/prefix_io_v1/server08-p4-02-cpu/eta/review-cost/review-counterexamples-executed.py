"""Independent CPU counterexamples, no backend imports or production grants."""
import sys,importlib.abc,json,hashlib,copy
from pathlib import Path
from dataclasses import asdict
root=Path.cwd()
sys.path.insert(0,str(root/'third_party/work/prefix-io-p4-02-cpu/src'))
sys.path.insert(0,str(root/'tests/prefix_io_v1_p4_measurements'))
attempts=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","vllm","py_kvcache","cupy","cuda","numpy"):
   attempts.append(fullname);raise RuntimeError("independent CPU guard forbids backend import")
sys.meta_path.insert(0,Guard())
from test_semantic_verifier import Fixture,io
from prefix_io_control.p4_verified_cost_loader import load_semantically_verified_table
from prefix_io_control.p4_production_table_contract import TableContractError
out=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-cost/counterexamples-01'
out.mkdir(parents=True,exist_ok=False)
source_paths=[root/'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control'/n
 for n in ('p4_paired_measurement_verifier.py','p4_verified_cost_loader.py')]
refs=[]
for p in source_paths:
 data=p.read_bytes();(out/p.name).write_bytes(data)
 refs.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
results=[]
def outcome(name,f):
 try:
  prepared=f.load();cell=prepared.verification.cells[0].cost
  result=dict(name=name,semantic_status=prepared.verification.status,
   accepted=True,cell_count=prepared.cell_count,
   production_qualified=prepared.production_qualified,gpu_verified=prepared.gpu_verified,
   production_lookup=prepared.lookup(cell.load_signature,cell.existing_io,cell.stage,cell.physical_bytes))
 except TableContractError as e:
  result=dict(name=name,accepted=False,error=str(e))
 results.append(result);return result
def fixture(name,**kwargs):
 directory=out/name;directory.mkdir();return Fixture(directory,**kwargs)
f=fixture('workload-hash-overlap')
shared=f.entries[0]['workload_sha256']
f.entries[2]['workload_sha256']=shared
for role in ('baseline_wrapper','action_wrapper'):
 f.docs[role]['runs'][2]['workload_sha256']=shared
f.refresh();outcome('same_workload_content_in_calibration_and_validation',f)
f=fixture('cross-cell-split-overlap')
entries2=[]
for i,e in enumerate(f.entries):
 e=copy.deepcopy(e);e['cell_id']='cell-2'
 if e['split']=='validation':
  for key in ('trace_sha256','prefix_family_sha256','workload_sha256'):
   e[key]=f.entries[0][key]
 else:
  e['trace_sha256']=hashlib.sha256(('cell2-trace-'+str(i)).encode()).hexdigest()
  e['prefix_family_sha256']=hashlib.sha256(b'cell2-calibration-family').hexdigest()
  e['workload_sha256']=hashlib.sha256(('cell2-work-'+str(i)).encode()).hexdigest()
 entries2.append(e)
docs2=copy.deepcopy(f.docs)
for role,obj in docs2.items():
 obj['cell_id']='cell-2'
 if role.endswith('wrapper'):
  for i,run in enumerate(obj['runs']):
   for key,value in entries2[i].items():
    if key!='cell_id':run[key]=value
   if role.startswith('action'):run['completed_new_io']=io(3,24)
 else:
  if role.startswith('action'):
   for row in obj['windows']:row['new_io']=io(1,8)
f.entries.extend(entries2)
f.refresh()
f.plan['action_operations']['cell-2']=1
f.plan_ref=f.dump('plan.json',f.plan)
f.analysis['plan_ref']=asdict(f.plan_ref)
cell1_analysis_ref=f.dump('analysis.json',f.analysis)
f.cell['measurement_refs']['pair_analysis']=asdict(cell1_analysis_ref)
refs2={name:f.dump(name+'_cell2.json',obj) for name,obj in docs2.items()}
a2=copy.deepcopy(f.analysis);a2['cell_id']='cell-2'
a2['evidence_refs']={role:asdict(ref) for role,ref in refs2.items()}
refs2['pair_analysis']=f.dump('analysis_cell2.json',a2)
cell2=copy.deepcopy(f.cell);cell2['cell_id']='cell-2';cell2['physical_bytes']=8
cell2['measurement_refs']={role:asdict(ref) for role,ref in refs2.items()}
f.candidate['cells']=[f.cell,cell2]
f.candidate_ref=f.dump('candidate.json',f.candidate)
f.qualification['candidate_ref']=asdict(f.candidate_ref)
f.qualification['measurement_refs']=[ref for c in f.candidate['cells'] for ref in c['measurement_refs'].values()]
f.qualification_ref=f.dump('qualification.json',f.qualification)
outcome('global_calibration_validation_family_trace_workload_overlap_across_cells',f)
f=fixture('invalid-active-batch')
f.cell['load']['active_decode']=3
for role in ('baseline_observations','action_observations'):
 for row in f.docs[role]['windows']:row['load']['active_decode']=3
f.refresh();outcome('active_decode_3_greater_than_batch_2',f)
f=fixture('forged-native-origin',origin='native_gpu_recording')
outcome('native_origin_label_cannot_open_production',f)
f=fixture('abi-source-drift')
f.docs['action_observations']['context']['native_source_sha256']='f'*64
f.refresh();outcome('raw_native_source_drift_rejected',f)
f=fixture('gpu-identity-drift')
f.docs['action_observations']['context']['gpu_uuid']='GPU-ANOTHER-DECLARED-ID'
f.refresh();outcome('raw_gpu_identity_drift_rejected',f)
receipt=dict(status='INDEPENDENT_CPU_SEMANTIC_REVIEW',results=results,source_inputs=refs,
 forbidden_import_attempts=attempts,gpu_workloads_run=0,production_gate_bypass_found=False,
 margin_interpretation='Validation contributes to empirical residual envelope; not independent evaluation coverage',
 source_files_unchanged_during_review=all(hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in refs))
(out/'result.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
