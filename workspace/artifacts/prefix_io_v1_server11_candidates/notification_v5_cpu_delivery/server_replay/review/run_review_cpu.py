"""Freeze evidence from the independent CPU attacks; no server/GPU operations."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REL=Path('source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py')
EXPECTED={
 'reactor':'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47',
 'collector':'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf',
 'frozen_worker':'096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b',
 'frozen_scalar':'347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb'}
def metadata(path):
 raw=path.read_bytes()
 return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,default=HERE.parent/'p4_single_file_candidate_v5_cpu')
 p.add_argument('--previous-candidate',type=Path,default=None);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();candidate=a.candidate.resolve();previous=(a.previous_candidate or candidate.with_name('p4_single_file_candidate_v4')).resolve()
 a.output.mkdir(parents=True,exist_ok=False)
 refs={'reactor':candidate/REL,'collector':candidate/'native_full_step_collector.py',
  'frozen_worker':HERE/'frozen_observer_sources/g2_worker_observation.py',
  'frozen_scalar':HERE/'frozen_observer_sources/p4_runtime_scalar_connector.py',
  'previous_reactor':previous/REL,'independent_test':HERE/'test_notification_adversarial.py',
  'independent_runner':Path(__file__).resolve()}
 for name in ('test_single_file_retry_observation.py','test_single_file_policy.py','test_single_file_receipt.py','test_reactor_single_file.py','test_retained_idle_wait.py'):
  refs['fixture_'+name]=candidate/name
 before={key:metadata(path) for key,path in refs.items()}
 assert all(before[key]['sha256']==digest for key,digest in EXPECTED.items()),'frozen source mismatch'
 env=os.environ.copy();env['SERVER11_C5_CANDIDATE']=str(candidate);env['SERVER11_PREVIOUS_CANDIDATE_ROOT']=str(previous)
 command=[sys.executable,'-B','-I','-S',str(HERE/'test_notification_adversarial.py')]
 start=time.monotonic();completed=subprocess.run(command,env=env,capture_output=True,timeout=45)
 after={key:metadata(path) for key,path in refs.items()}
 (a.output/'STDOUT.txt').write_bytes(completed.stdout);(a.output/'STDERR.txt').write_bytes(completed.stderr)
 summary=completed.stderr.decode('utf-8',errors='replace');match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',summary)
 result={'schema':'notification_v5_independent_cpu_run_v1','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'command':command,'python_version':sys.version,'candidate':str(candidate),'previous_candidate':str(previous),
  'returncode':completed.returncode,'elapsed_wall_seconds':time.monotonic()-start,
  'tests_run':int(match.group(1)) if match else None,'unittest_elapsed_seconds':float(match.group(2)) if match else None,
  'success':completed.returncode==0 and match is not None and int(match.group(1))==32 and before==after,
  'source_before':before,'source_after':after,'source_stable':before==after,
  'GPU_operations':0,'RPC_operations':0,'CUDA_imports':False,'GPU_ready':False,
  'performance_effect_verified':False,'production_qualified':False,
  'limitations':['Queue/threads are real Python objects; raw Event and backend I/O are synthetic.',
   'Native intake/submission/mandatory/STOP AST is exercised with synthetic backend release bookkeeping.',
   'The full native _run/_pump_once concurrent performance gate is a separate root-owned experiment.',
   'off/uninstalled retains original work path but adds one helper call plus getattr fast return per reactor loop.',
   'Copies of the frozen observer/scalar modules are byte/SHA checked; this is not runtime GPU source qualification.']}
 (a.output/'TEST_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'success':result['success'],'tests_run':result['tests_run'],'source_stable':result['source_stable'],
  'output':str(a.output.resolve()),'GPU_ready':False},ensure_ascii=False))
 return 0 if result['success'] else 1
if __name__=='__main__':raise SystemExit(main())
