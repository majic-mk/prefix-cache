from pathlib import Path
import json,os,subprocess,hashlib,time,xml.etree.ElementTree as ET,sys
r=Path("/root/autodl-tmp/prefix-io-v1-handoff/project");sys.path.insert(0,str(r/'experiments/prefix_io_v1/scripts'))
sys.path.insert(0,str(r/'third_party/work/prefix-io-p4-02-cpu/src'))
from experiment_storage import preflight
out=r/'artifacts/prefix_io_v1/server08-p4-02-cpu/root-load-02';out.mkdir(exist_ok=False)
scratch=r/'experiments/prefix_io_v1/runs/server08-p4-02-root-load-02/cpu-evidence'
preflight(scratch,128*1024*1024);scratch.mkdir(parents=True,exist_ok=False)
cmd=[str(r/'.venv/bin/python'),str(r/'experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py'),'-q','--import-mode=importlib','--basetemp',str(scratch/'pytest-tmp'),'--junitxml',str(out/'cpu.xml'),'tests/prefix_io_v1_p4_load','tests/prefix_io_v1_p4_02_bridge']
envdelta={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
 'PYTHONPATH':':'.join(str(r/p) for p in ['third_party/work/py-kvcache-p4-02-cpu','third_party/work/prefix-io-p4-02-cpu/src','experiments/prefix_io_v1/scripts','.']),
 'AIO_CPU_GPU_GUARD_PATH':str(out/'cuda-guard.json')}
(out/'command.json').write_text(json.dumps(dict(argv=cmd,cwd=str(r),environment_delta=envdelta),indent=2)+'\n')
ledger=r/'experiments/prefix_io_v1/gpu-budget-ledger.json';before=ledger.read_bytes()
env=os.environ.copy();env.update(envdelta)
with (out/'process.log').open('xb') as log:
 p=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=120)
 except BaseException:
  from run_gpu_stage import cleanup_session
  cleanup_session(p);raise
cases=ET.parse(out/'cpu.xml').findall('.//testcase')
counts={s:0 for s in ['passed','failed','skipped']}
for c in cases:
 s='failed' if c.find('failure') is not None or c.find('error') is not None else 'skipped' if c.find('skipped') is not None else 'passed';counts[s]+=1
result=dict(process_exit=code,counts=counts,guard=json.loads((out/'cuda-guard.json').read_text()),ledger_unchanged=ledger.read_bytes()==before,ledger_sha256=hashlib.sha256(before).hexdigest())
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));print((out/'process.log').read_text()[-10000:])

raise SystemExit(code)
