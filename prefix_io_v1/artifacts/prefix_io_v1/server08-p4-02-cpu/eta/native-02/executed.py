from pathlib import Path
import os,sys,subprocess,json,time,hashlib,xml.etree.ElementTree as ET
root=Path.cwd();out=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/eta/native-02'
scratch=root/'experiments/prefix_io_v1/runs/server08-p4-02-eta-native-02/cpu-evidence'
sys.path.insert(0,str(root/'experiments/prefix_io_v1/scripts'))
sys.path.insert(0,str(root/'third_party/work/prefix-io-p4-02-cpu/src'))
sys.path.extend(str(p) for p in (root/'.venv/lib').glob('python*/site-packages'))
from experiment_storage import preflight
preflight(scratch,16*1024*1024)
out.mkdir(parents=True,exist_ok=False);scratch.mkdir(parents=True,exist_ok=False)
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json';before=ledger.read_bytes()
assert json.loads(before)['active_reservation'] is None
cmd=[str(root/'.venv/bin/python'),str(root/'experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py'),
 '-q','--import-mode=importlib','--basetemp',str(scratch/'pytest-tmp'),'--junitxml',str(out/'cpu.xml'),
 'tests/prefix_io_v1_p4_eta','tests/prefix_io_v1_p4_02_bridge']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 PYTHONPATH=':'.join(map(str,[root/'third_party/work/py-kvcache-p4-02-cpu',root/'third_party/work/prefix-io-p4-02-cpu/src',root])),
 AIO_CPU_GPU_GUARD_PATH=str(out/'cuda-guard.json'))
(out/'command.json').write_text(json.dumps(dict(cmd=cmd,cwd=str(root),environment_delta={k:env[k] for k in
 ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONPATH','AIO_CPU_GPU_GUARD_PATH']}),indent=2))
start=time.monotonic()
with (out/'process.log').open('xb') as f:p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
tree=ET.parse(out/'cpu.xml');cases=tree.findall('.//testcase')
counts={k:0 for k in ('passed','failed','skipped')}
for c in cases:
 state='failed' if c.find('failure') is not None or c.find('error') is not None else 'skipped' if c.find('skipped') is not None else 'passed'
 counts[state]+=1
guard=json.loads((out/'cuda-guard.json').read_text()) if (out/'cuda-guard.json').exists() else None
result=dict(status='PASS_CPU_ETA_NATIVE_AND_BRIDGE' if p.returncode==0 else 'FAIL_CPU_ETA_NATIVE_AND_BRIDGE',
 exit=p.returncode,seconds=time.monotonic()-start,counts=counts,guard=guard,ledger_unchanged=ledger.read_bytes()==before,
 ledger_sha256=hashlib.sha256(before).hexdigest(),gpu_workloads_run=0,production_eta_qualified=False,
 scope='CPU causal fixtures and fake native completion protocols; no physical GPU measurement or production ETA')
(out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result));print((out/'process.log').read_text()[-12000:])
