import argparse,hashlib,json,os,time,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description='Pack actual GPU source and evidence only after the original ledger is idle')
p.add_argument('--project-root',type=Path,required=True)
p.add_argument('--enumerate-only',action='store_true')
a=p.parse_args();root=a.project_root.resolve(strict=True)
d=root/'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005'
prep=root/'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json'; ledger_raw=ledger.read_bytes()
assert json.loads(ledger_raw)['active_reservation'] is None
files={}
def add(path):
 assert not path.is_symlink()
 assert not set(path.parts)&{'private-storage','runtime-cache','__pycache__'}
 if path.is_file():
  assert path.resolve().is_relative_to(root) and path.stat().st_size<=32*1024**2
  files[path.relative_to(root).as_posix()]=path
for base in [d,*[prep/x for x in ('runner','raw_device_binding','raw_relative_config','raw_cache_semantics',
 'raw_capture_diagnostics','raw_no_forward_observation','raw_no_forward_binding','raw_initial_execution_contract',
 'raw_runtime_collector_binding','formal_trace_binding','activation','protocol','source_inputs','review')]]:
 if base.is_dir():
  for path in base.rglob('*'):
   if path.suffix in ('.py','.json','.log','.md','.patch','.yaml') and not set(path.parts)&{'private-storage','runtime-cache','__pycache__'} and not path.name.startswith('GPU_DELIVERY_'):
    add(path)
for path in prep.glob('*'):
 if path.is_file() and (path.name.startswith(('PRERENT_SOURCE_','CPU_','CPU_FULL_SOURCE_FREEZE_','STRONG_','freeze_prerental_sources_','entry_control_v4'))):
  add(path)
for name in ['live-off01/EFFECTIVE_GPU_PERMISSION.json','live-off01/CONFIG.json','live-off01/BOUND_SOURCE_LOCK.json','live-off01/BOUND_SOURCE_PROOF.json']:
 add(prep/name)
for label in ['server12-strong-u-qual-off01']+['server12-strong-exact-cal%02d'%n for n in range(1,10)]:
 base=root/'experiments/prefix_io_v1/runs'/label
 if base.is_dir():
  for path in base.rglob('*'):
   if not set(path.parts)&{'private-storage','runtime-cache','__pycache__'} and path.suffix in ('.json','.log','.md'):
    add(path)
add(ledger); add(root/'permissions.yaml')
add(root/'experiments/prefix_io_v1/scripts/run_gpu_stage.py')
def add_refs(value):
 if isinstance(value,dict):
  if {'path','bytes','sha256'}<=value.keys() and isinstance(value['path'],str):
   p=root/value['path']
   if p.suffix in ('.py','.json','.md','.patch','.yaml') and not set(p.parts)&{'private-storage','runtime-cache','__pycache__'}:
    raw=p.read_bytes()
    assert len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256']
    add(p)
  for v in value.values():add_refs(v)
 elif isinstance(value,list):
  for v in value:add_refs(v)
for source in [d/'raw05/BOUND_PLAN.json',d/'raw05/ACTUAL_COST_ISSUER_REPORT.json']:
 add_refs(json.loads(source.read_bytes()))
assert __import__('shutil').disk_usage(root).free>=8*1024**3
if a.enumerate_only:
 print(json.dumps({'status':'PASS_BOUNDED_ARCHIVE_ENUMERATION','source_members':len(files),
  'source_bytes':sum(p.stat().st_size for p in files.values()),'maximum_source_members':1200,
  'maximum_source_bytes':192*1024**2,'archive_not_written':True}))
 assert 1<=len(files)<=1200 and sum(p.stat().st_size for p in files.values())<=192*1024**2
 raise SystemExit(0)

assert 1<=len(files)<=1200 and sum(p.stat().st_size for p in files.values())<=192*1024**2
rows=[]
for name,path in sorted(files.items()):
 raw=path.read_bytes();rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
manifest=d/'GPU_DELIVERY_SOURCE_EVIDENCE_MANIFEST.json'
with manifest.open('x',encoding='utf-8') as f:
 json.dump(dict(schema='actual_GPU_source_and_evidence_archive_manifest_v1',files=rows,
  copies_models_or_SDK_or_private_payloads=False,contains_changed_sources_and_referenced_small_validation_leaves=True,full_locked_source_closure_contents_included=False,all_original_private_payloads_preserved_on_server=True,
  original_idle_ledger_sha256=hashlib.sha256(ledger_raw).hexdigest()),f,indent=2,sort_keys=True)
archive=d/'GPU_DELIVERY_SOURCE_EVIDENCE.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,path in sorted(files.items()):z.write(path,arcname=name)
 z.write(manifest,arcname=manifest.relative_to(root).as_posix())
 with z.open('ARCHIVE_SCOPE.json','w') as f:f.write(json.dumps(dict(schema='actual_GPU_delivery_scope_v1',GPU_results_are_real_evidence=True,
  private_compilation_caches_and_SSD_payloads_remain_on_server=True,archive_is_not_model_or_data_backup=True)).encode())
assert ledger.read_bytes()==ledger_raw
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.infolist())==len(rows)+2 and len(set(z.namelist()))==len(rows)+2
 for row in rows:
  b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
for row in rows:

 raw=(root/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
result=dict(schema='actual_GPU_delivery_archive_actual_byte_verification_v1',archive=dict(path=archive.relative_to(root).as_posix(),
 bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),member_count=len(rows)+2,
 all_members_sources_rechecked=True,all_ZIP_member_CRC_and_SHA_rechecked=True,original_ledger_idle_unchanged=True,
 actual_GPU_operations_this_packaging_action=0,qualification_is_not_implied_by_archive=True)
with (d/'GPU_DELIVERY_ARCHIVE_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,sort_keys=True)
print(json.dumps(result))
