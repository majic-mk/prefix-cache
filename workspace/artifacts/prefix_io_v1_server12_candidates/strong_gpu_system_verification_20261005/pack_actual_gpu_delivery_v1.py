import argparse,hashlib,json,os,time,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description='Pack actual GPU source and evidence only after the original ledger is idle')
p.add_argument('--project-root',type=Path,required=True)
a=p.parse_args();root=a.project_root.resolve(strict=True)
d=root/'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005'
prep=root/'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json'; ledger_raw=ledger.read_bytes()
assert json.loads(ledger_raw)['active_reservation'] is None
files={}
def add(path):
 assert not path.is_symlink()
 if path.is_file():
  assert path.resolve().is_relative_to(root) and path.stat().st_size<=10*1024**2
  files[path.relative_to(root).as_posix()]=path
for base in [d,*[prep/x for x in ('runner','raw_device_binding','raw_relative_config','raw_cache_semantics',
 'raw_capture_diagnostics','raw_no_forward_observation','raw_no_forward_binding')]]:
 if base.is_dir():
  for path in base.rglob('*'):
   if path.suffix in ('.py','.json','.log','.md','.patch','.yaml') and '__pycache__' not in path.parts and not path.name.startswith('GPU_DELIVERY_'):
    add(path)
for path in prep.glob('*'):
 if path.is_file() and (path.name.startswith(('PRERENT_SOURCE_','CPU_GPU_RAW_','CPU_FULL_SOURCE_FREEZE_','STRONG_','freeze_prerental_sources_','entry_control_v4'))):
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
assert 1<=len(files)<=1200 and sum(p.stat().st_size for p in files.values())<=80*1024**2
rows=[]
for name,path in sorted(files.items()):
 raw=path.read_bytes();rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
manifest=d/'GPU_DELIVERY_SOURCE_EVIDENCE_MANIFEST.json'
with manifest.open('x',encoding='utf-8') as f:
 json.dump(dict(schema='actual_GPU_source_and_evidence_archive_manifest_v1',files=rows,
  copies_models_or_SDK_or_private_payloads=False,all_original_private_payloads_preserved_on_server=True,
  original_idle_ledger_sha256=hashlib.sha256(ledger_raw).hexdigest()),f,indent=2,sort_keys=True)
archive=d/'GPU_DELIVERY_SOURCE_EVIDENCE.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,path in sorted(files.items()):z.write(path,arcname=name)
 z.write(manifest,arcname=manifest.relative_to(root).as_posix())
 with z.open('ARCHIVE_SCOPE.json','w') as f:f.write(json.dumps(dict(schema='actual_GPU_delivery_scope_v1',GPU_results_are_real_evidence=True,
  private_compilation_caches_and_SSD_payloads_remain_on_server=True,archive_is_not_model_or_data_backup=True)).encode())
assert ledger.read_bytes()==ledger_raw
for row in rows:
 raw=(root/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
result=dict(schema='actual_GPU_delivery_archive_actual_byte_verification_v1',archive=dict(path=archive.relative_to(root).as_posix(),
 bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),member_count=len(rows)+2,
 all_members_sources_rechecked=True,original_ledger_idle_unchanged=True,
 actual_GPU_operations_this_packaging_action=0,qualification_is_not_implied_by_archive=True)
with (d/'GPU_DELIVERY_ARCHIVE_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,sort_keys=True)
print(json.dumps(result))
