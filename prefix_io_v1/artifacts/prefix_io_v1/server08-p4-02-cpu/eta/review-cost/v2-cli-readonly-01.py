from pathlib import Path
import sys,json,hashlib,importlib.abc
root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
out=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-cost/v2-cli-readonly-01';out.mkdir(exist_ok=False)
attempts=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','py_kvcache','vllm','cupy','numpy'):
   attempts.append(name);raise RuntimeError('CPU backend guard '+name)
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(root/'third_party/work/prefix-io-p4-02-cpu/src'))
from prefix_io_control.p4_production_table_contract import EvidenceRef,TableContext
from prefix_io_control.p4_verified_cost_loader import load_semantically_verified_table
base=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/cost-verifier/fixtures-v2-cli-01'
def tree():return {p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.is_file()}
before=tree();processes=json.loads((base/'actual-process-results.json').read_text());results=[]
assert len(processes)==5
for run in processes:
 d=base/run['case'];g=json.loads((d/'actual-cli-process-guard.json').read_text())
 assert run['exit']==run['expected_exit']==g['exit']
 assert g['GPU_runs']==0 and g['backend_import_attempts']==[] and g['cuda_initialized'] is False and g['production_qualified'] is False
 r={'case':run['case'],'recorded_process_exit':g['exit'],'recorded_GPU_runs':0,'recorded_backend_imports':[]}
 if run['exit']==0:
  output=d/'artifacts/prefix_io_v1/server08-p4-02-cpu/gpu-next-day/cli-assembled'
  receipt=json.loads((output/'receipt.json').read_text());bundle=json.loads((d/'bundle.json').read_text())
  for key in ['candidate_ref','qualification_ref','analysis_ref','bundle_ref','plan_ref','verifier_ref']:
   EvidenceRef.from_mapping(receipt[key]).verify(d)
  for ref in list(receipt['raw_refs'].values())+receipt['ingested_source_refs']:EvidenceRef.from_mapping(ref).verify(d)
  assert receipt['GPU_collector_implemented'] is False and receipt['effect_verified'] is False and receipt['P5_allowed'] is False
  t=load_semantically_verified_table(d,receipt['candidate_ref']['path'],expected_context=TableContext.from_mapping(bundle['context']),
    qualification_ref=EvidenceRef.from_mapping(receipt['qualification_ref']),
    expected_verifier_ref=EvidenceRef.from_mapping(receipt['verifier_ref']),
    plan_path=receipt['plan_ref']['path'],expected_plan_ref=EvidenceRef.from_mapping(receipt['plan_ref']))
  c=t.verification.cells[0].cost
  assert t.lookup(c.load_signature,c.existing_io,c.stage,c.physical_bytes) is None
  wrapper=json.loads((d/receipt['raw_refs']['action_wrapper']['path']).read_text())
  assert all(x['full_output_tokens']==128 and x['output_tokens']==128 for x in wrapper['runs'])
  r.update(status=t.status,production_qualified=t.production_qualified,gpu_verified=t.gpu_verified,
    production_lookup=None,full_output_tokens=128,selected_output_tokens=wrapper['runs'][0]['selected_output_tokens'],
    ingested_refs_checked=len(receipt['ingested_source_refs']),measurement_refs_checked=len(t.verification.evidence_refs))
 else:r['recorded_rejection']=g['error']
 results.append(r)
assert tree()==before and not attempts
source=root/'experiments/prefix_io_v1/scripts/prepare_p4_raw_pair.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='dae3b4f78d174233ae63eb9f2ca2cd1aefdfd5160d2e09ccf6a872317202c6a4'
result={'status':'PASS_READONLY_CLI_REFS_AND_PUBLIC_LOADER_REBIND','cases':results,'original_fixtures_unchanged':True,
 'current_review_backend_import_attempts':attempts,'current_review_GPU_runs':0,'new_CLI_processes_launched':0,
 'CLI_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'scope':'Five existing actual CLI process receipts independently checked; positive nested refs reverified and public loader rerun; negatives are preserved actual process evidence.'}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
