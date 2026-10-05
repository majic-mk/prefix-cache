from pathlib import Path
import os, json, hashlib, shutil, zipfile
project=Path.cwd()
base=project/'artifacts/prefix_io_v1/server07-p3-05'
aux=Path('/root/prefix-io-v1-validation')
archive=aux/'deliveries/prefix_io_v1_server07_p3_auxiliary_20260929.zip'
assert not archive.exists()
files={}
def add(p, arc=None):
 p=Path(p)
 if not p.is_file() or p.is_symlink():return
 arc=arc or str(p.relative_to(project))
 if arc in files:assert files[arc]==p
 files[arc]=p
def tree(folder, prefix=None):
 folder=Path(folder)
 if not folder.exists():return
 for parent,dirs,names in os.walk(folder,followlinks=False):
  dirs[:]=[n for n in dirs if n not in {'__pycache__','.git','storage','models','Qwen'} and not (Path(parent)/n).is_symlink()]
  for n in names:
   p=Path(parent)/n
   if p.suffix in {'.pyc','.bin','.so','.zip'}:continue
   add(p, str(Path(prefix)/p.relative_to(folder)) if prefix else None)
for folder in [base,project/'docs/prefix_io_v1',project/'src/prefix_io_control',project/'experiments/prefix_io_v1/scripts',project/'experiments/prefix_io_v1/configs',project/'experiments/prefix_io_v1/locks',project/'patches/prefix_io_v1',project/'tests/prefix_io_v1',project/'tests/prefix_io_v1_pilot',project/'tests/prefix_io_v1_native_costs']:
 tree(folder)
for n in ['experiments/prefix_io_v1/execution_state.json','experiments/prefix_io_v1/gpu-budget-ledger.json','artifacts/prefix_io_v1/server07-p2-01/source-lock.json','artifacts/prefix_io_v1/server07-p3-03/calibration-candidate/curves-v2.json','artifacts/prefix_io_v1/server07-p3-04/heldout-cost-result.json','artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json']:
 add(project/n)
for n in json.loads((base/'source-lock.json').read_text()):add(project/n)
events=json.loads((base/'final-health.json').read_text())['events']
for event in events:
 label=event['label']
 tree(project/'experiments/prefix_io_v1/runs'/label)
 tree(aux/'runs'/label, 'auxiliary/runs/'+label)
add(aux/'PROJECT.json','auxiliary/PROJECT.json')
add(aux/'probe/result.json','auxiliary/probe/result.json')
raw_size=sum(p.stat().st_size for p in files.values())
import sys
sys.path.insert(0,str(project/'experiments/prefix_io_v1/scripts'))
from experiment_storage import preflight
pre=preflight(archive,raw_size+4*1024**2)
archive.parent.mkdir(parents=True,exist_ok=True)
records=[]
for name,p in sorted(files.items()):
 records.append({'path':name,'server_path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest={'schema_version':1,'project_root':str(project),'auxiliary_root':str(aux),'excludes':['model weights','bulk KV cache storage','compiled binaries','git databases'],'files':records}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for item in records:z.write(files[item['path']],item['path'])
 z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(records)+1
 for item in records:
  data=z.read(item['path'])
  assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
post=preflight(archive,0)
res={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files_verified':len(records),'raw_file_bytes':raw_size,'preflight':pre,'postflight':post,'project_free_bytes':shutil.disk_usage(project).free}
(base/'package-result.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res))
