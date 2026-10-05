"""Append-only P4-02 CPU and future-GPU source freeze. No backend imports."""
from pathlib import Path
import sys,json,hashlib,ast,datetime
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/prefix_io_v1/scripts'))
from prepare_p4_gpu_next_day import required_source_paths
OUT=ROOT/'artifacts/prefix_io_v1/server08-p4-02-cpu'
def ref(relative):
 p=ROOT/relative
 assert not p.is_symlink() and p.is_file()
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return dict(path=relative,bytes=p.stat().st_size,sha256=h.hexdigest())
gpu=set(required_source_paths(ROOT))
inventory=json.loads((OUT/'observation-cpu/source-lock-v1.json').read_text())['old_binary_inventory']
assert len(inventory)==7
for row in inventory:
 assert row['path'].startswith('third_party/work/vllm-author-build/vllm/') and row['path'].endswith('.so')
 assert ref(row['path'])==row
 gpu.add(row['path'])
for directory in ('third_party/work/py-kvcache-p4-02-cpu/tests','experiments/prefix_io_v1/scripts'):
 gpu.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*.py'))
# Freeze actual Python sources plus tests and original helpers without importing engines.
cpu=set(gpu)
for directory in [ROOT/'src/prefix_io_control', *sorted((ROOT/'tests').glob('prefix_io_v1*'))]:
 if directory.is_dir():
  cpu.update(p.relative_to(ROOT).as_posix() for p in directory.rglob('*.py'))
cpu.update(('experiments/prefix_io_v1/configs/permissions.yaml',
            'experiments/prefix_io_v1/gpu-budget-ledger.json',
            'artifacts/prefix_io_v1/server08-p4-02-cpu/CPU_ONLY_AUTHORIZATION.json',
            'artifacts/prefix_io_v1/server08-p3-16/execution-lock-12-final-p3.json'))
for relative in sorted(gpu):
 if relative.endswith('.py'):ast.parse((ROOT/relative).read_text(),filename=relative)
for name,paths in (('gpu-source-lock.json',gpu),('test-input-lock.json',cpu)):
 target=OUT/name
 assert not target.exists(),target
 record=dict(schema_version=1,status='FROZEN_CURRENT_SOURCE_CPU_ONLY_GPU_UNQUALIFIED',
   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   files=[ref(s) for s in sorted(paths)],new_gpu_runs=0,backend_imports=[],
   source_bytes_hashing_is_not_GPU_or_binary_ABI_qualification=True)
 with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps(dict(path=str(target),files=len(paths),sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
