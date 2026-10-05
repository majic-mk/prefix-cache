"""Package reviewed P4-02 CPU evidence and changed Python sources, append-only."""
from pathlib import Path
import json,hashlib,zipfile,datetime
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/prefix_io_v1/server08-p4-02-cpu'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def r(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
close=json.loads((OUT/'P4_02_CPU_ROOT_CLOSEOUT.json').read_text())
assert close['full_P4_complete'] is False and close['GPU']['new_runs']==0
passed=json.loads((OUT/close['integration_result']['path'].split(OUT.name+'/')[-1]).read_text())
assert passed['status']=='PASS_P4_02_CPU_CURRENT_INTEGRATION'
archive=OUT/'P4_02_CPU_EVIDENCE.zip';manifest_path=OUT/'PACKAGE_FILE_MANIFEST.json';receipt_path=OUT/'PACKAGE_RESULT.json'
assert all(not p.exists() for p in (archive,manifest_path,receipt_path))
items={}
for p in OUT.rglob('*'):
 if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.so','.bin','.zip'):
  items['evidence/'+p.relative_to(OUT).as_posix()]=p
dirs=('third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control',
      'third_party/work/py-kvcache-p4-02-cpu/py_kvcache',
      'third_party/work/py-kvcache-p4-02-cpu/tests')
for directory in dirs:
 for p in (ROOT/directory).rglob('*.py'):items['source/'+p.relative_to(ROOT).as_posix()]=p
for directory in (ROOT/'tests').glob('prefix_io_v1_p4*'):
 for p in directory.rglob('*.py'):items['source/'+p.relative_to(ROOT).as_posix()]=p
for p in (ROOT/'experiments/prefix_io_v1/scripts').glob('*p4*.py'):
 items['source/'+p.relative_to(ROOT).as_posix()]=p
for name in ('common','scheduler','worker'):
 p=ROOT/('third_party/work/vllm-author-p4-02-cpu/vllm/distributed/kv_transfer/kv_connector/v1/offloading/'+name+'.py')
 items['source/'+p.relative_to(ROOT).as_posix()]=p
p=ROOT/'third_party/work/vllm-author-p4-02-cpu/vllm/platforms/cuda.py';items['source/'+p.relative_to(ROOT).as_posix()]=p
supp=json.loads((OUT/'gpu-next-day/AUTHOR_PYTHON_OVERLAY_COPY_92.json').read_text())
# The source copy receipt field names are resolved against its individually bound refs.
for key,value in supp.items():
 if isinstance(value,list):
  for entry in value:
   if isinstance(entry,dict):
    for k,v in entry.items():
     if k in ('target','destination','target_ref','destination_ref') and isinstance(v,dict) and 'path' in v:
      p=ROOT/v['path']
      if p.suffix=='.py':items['source/'+p.relative_to(ROOT).as_posix()]=p
# Also include all append-only runtime supplement source names in the independent inventory.
inv=json.loads((OUT/'eta/review-integration/author-overlay-inventory.json').read_text())
for entry in inv.get('missing_refs',[]):
 s=entry.get('path') if isinstance(entry,dict) else None
 if s and s.startswith('third_party/work/vllm-author-build/'):
  p=ROOT/s.replace('third_party/work/vllm-author-build/','third_party/work/vllm-author-p4-02-cpu/',1)
  assert p.is_file() and p.suffix=='.py' and r(p)=={k:entry[k] for k in ('bytes','sha256')}
  items['source/'+p.relative_to(ROOT).as_posix()]=p
for s in ('experiments/prefix_io_v1/execution_state.json','experiments/prefix_io_v1/configs/permissions.yaml',
          'experiments/prefix_io_v1/gpu-budget-ledger.json'):
 p=ROOT/s;items['source/'+s]=p
rows=[dict(path=name,**r(p)) for name,p in sorted(items.items())]
manifest=dict(schema_version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scope='CPU evidence and changed sources/runtime supplement; existing repository/P3/model/binaries required separately',
 files=rows,real_GPU_runs_this_delivery=0,model_or_private_KV_payload_included=False,binary_or_venv_included=False)
with manifest_path.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for row in rows:
  p=items[row['path']];assert r(p)=={k:row[k] for k in ('bytes','sha256')}
  z.write(p,row['path'])
 z.write(manifest_path,'PACKAGE_FILE_MANIFEST.json')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert len(z.infolist())==len(rows)+1
 for row in rows:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
 assert z.read('PACKAGE_FILE_MANIFEST.json')==manifest_path.read_bytes()
result=dict(status='PASS_SERVER_ARCHIVE_CRC_AND_ALL_FILE_SHA256',archive=dict(path=str(archive),**r(archive)),
 manifest=dict(path=str(manifest_path),**r(manifest_path)),files=len(rows),archive_entries=len(rows)+1,
 new_GPU_runs=0,source_files=sum(s.startswith('source/') for s in items),
 evidence_files=sum(s.startswith('evidence/') for s in items))
with receipt_path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
