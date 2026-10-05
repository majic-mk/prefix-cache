from pathlib import Path
import json,hashlib,datetime,time,os
root=Path('.');out=root/'artifacts/prefix_io_v1/server08-p4-01-cpu';start=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
lockpath=root/'artifacts/prefix_io_v1/server08-p3-16/execution-lock-12-final-p3.json';locked=json.loads(lockpath.read_text());bad=[]
for s,expected in locked.items():
 p=Path(s)
 if sha(p)!=expected:bad.append(s)
assert not bad,bad[:8]
base=json.loads((out/'cpu-workspace-base.json').read_text());assert all(sha(Path(s))==v for s,v in base['base_control_files'].items())
for v in base['copied_native_python']:
 assert sha(Path(base['base_native'])/v['relative'])==v['sha256']
old=json.loads((root/'artifacts/prefix_io_v1/server08-p3-16/source-preservation-final-p3.json').read_text());manifestpath=Path(old['source_manifest']['path']);assert sha(manifestpath)==old['source_manifest']['sha256'];manifest=json.loads(manifestpath.read_text());source=Path(old['source_root']);size=0
for ref in manifest:
 p=source/ref['path'];assert not p.is_symlink() and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'];size+=ref['bytes']
assert {str(p.relative_to(source)) for p in source.rglob('*.bin')}=={r['path'] for r in manifest}
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json';j=json.loads(ledger.read_text());assert sha(ledger)=='31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1' and j['active_reservation'] is None
auth=json.loads((out/'CPU_ONLY_AUTHORIZATION.json').read_text());p=root/auth['base_permissions']['path'];assert sha(p)==auth['base_permissions']['sha256']
refs=json.loads((out/'test-input-lock-v2.json').read_text())['files'];assert all(Path(r['path']).stat().st_size==r['bytes'] and sha(Path(r['path']))==r['sha256'] for r in refs)
result=dict(schema_version=1,status='PASS_P4_CPU_INPUTS_AND_ALL_P3_PROTECTED_BYTES_PRESERVED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),P3_locked_inputs=len(locked),P3_locked_inputs_unchanged=True,original_control_files=len(base['base_control_files']),original_control_unchanged=True,original_native_unchanged=True,P4_frozen_inputs=len(refs),P4_frozen_inputs_unchanged=True,registered_cache_files=len(manifest),registered_cache_bytes=size,registered_cache_bytes_unchanged=True,no_extra_registered_bin_files=True,GPU_ledger_sha256=sha(ledger),GPU_ledger_unchanged=True,permissions_unchanged=True,active_GPU_reservation=None,new_GPU_runs=0,GPU_availability_probed=False,model_weights_read=False,downloads_bytes_added=0,cache_merge_or_deletion=False,system_changes=False,seconds=time.monotonic()-start,primary_free_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
p=out/'final-input-and-P3-preservation.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
