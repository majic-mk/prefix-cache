"""Freeze the small device-index compatibility revision; no GPU authority."""
import argparse, pathlib, hashlib, json, datetime
PARENT='artifacts/prefix_io_v1/server11-c5-gpu-entry-delivery-20261004/SOURCE_LOCK_GPU_ENTRY_CPU.json'
PARENT_SHA='e91df96e4203ae4b4925e90613a277632ad2ae3a3a337e9e947ad29c7f7431f5'
C='artifacts/prefix_io_v1/server11-c5-gpu-entry-device-revision-20261004'
D='artifacts/prefix_io_v1/server11-c5-gpu-entry-device-delivery-20261004'
def ref(root,p):
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest())
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True,type=pathlib.Path);root=a.parse_args().root.resolve(strict=True)
 old=json.loads((root/PARENT).read_bytes());assert hashlib.sha256((root/PARENT).read_bytes()).hexdigest()==PARENT_SHA
 rows={}
 for row in old['files']:
  assert ref(root,root/row['path'])==row
  rows[row['path']]=row
 rows[PARENT]=ref(root,root/PARENT)
 for folder in (root/C,root/D):
  for p in sorted(folder.rglob('*')):
   assert not p.is_symlink()
   if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ('.py','.md') or p.name in ('NATIVE_SOURCE_INHERITANCE.json','SESSION_BEFORE_DEVICE_GPU_VALIDATION.json')):
    row=ref(root,p);rows[row['path']]=row
 assert len(rows)<=512
 lock=dict(schema='c5_gpu_entry_cpu_source_lock_v1',gpu_uuid=None,source_only=True,gpu_launch_allowed=False,native_execution_verified=False,native_cost_qualified=False,full_runtime_cost_qualified=False,on_observation_cost_measured=False,parent_ref=ref(root,root/PARENT),files=[rows[k] for k in sorted(rows)])
 out=root/D/'SOURCE_LOCK_DEVICE_CPU.json'
 with out.open('x',encoding='utf-8') as f:json.dump(lock,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(status='PASS_DEVICE_SOURCE_FREEZE',source_count=len(rows),parent_files_verified=len(old['files']),source_lock_ref=ref(root,out),GPU_runs=0)))
if __name__=='__main__':main()

