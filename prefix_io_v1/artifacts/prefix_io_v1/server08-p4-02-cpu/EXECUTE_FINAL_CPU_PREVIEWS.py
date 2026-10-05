"""Final CPU-only launcher previews/denial and JUnit evidence summary; no GPU."""
from pathlib import Path
import json,hashlib,subprocess,os,sys,collections,xml.etree.ElementTree as ET
ROOT=Path.cwd()
OUT=ROOT/'artifacts/prefix_io_v1/server08-p4-02-cpu'
ART=OUT.relative_to(ROOT).as_posix()
LOCK=ART+'/gpu-source-lock.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
before={name:ref(ROOT/name) for name in ('experiments/prefix_io_v1/gpu-budget-ledger.json','experiments/prefix_io_v1/configs/permissions.yaml',LOCK,ART+'/test-input-lock.json',ART+'/CPU_ONLY_AUTHORIZATION.json')}
assert not (OUT/'GPU_STAGE_AUTHORIZATION.json').exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 PYTHONPATH=str(ROOT/'third_party/work/prefix-io-p4-02-cpu/src')+':'+str(ROOT/'experiments/prefix_io_v1/scripts'))
prefix=['.venv/bin/python']
commands=[
(prefix+['experiments/prefix_io_v1/scripts/prepare_p4_gpu_next_day.py','--project','.','--output',ART+'/gpu-next-day/plan-final.json','--source-lock',LOCK],0),
(prefix+['experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py','--mode','off','--name','server08-p4-02-off-final-preview','--dry-run','--source-lock',LOCK,'--receipt',ART+'/gpu-next-day/native-off-final-preview.json'],0),
(prefix+['experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py','--mode','shadow','--name','server08-p4-02-shadow-final-preview','--dry-run','--source-lock',LOCK,'--receipt',ART+'/gpu-next-day/native-shadow-final-preview.json'],0),
(prefix+['experiments/prefix_io_v1/scripts/prepare_p4_calibration_startup.py','--project','.','--name','server08-p4-02-final-startup','--stage','ssd_read','--units','1','--source-lock',LOCK,'--output',ART+'/gpu-next-day/calibration-startup-final.json'],0),
(prefix+['experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py','--mode','off','--name','server08-p4-02-final-denied','--launch','--source-lock',LOCK,'--receipt',ART+'/gpu-next-day/native-launch-final-denied.json'],78)]
rows=[]
for index,(argv,expected) in enumerate(commands):
 p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
 log=OUT/'gpu-next-day'/('final-cli-%02d.stdout.json'%index)
 with log.open('x') as f:json.dump(dict(stdout=p.stdout,stderr=p.stderr),f,indent=2);f.write('\n')
 assert p.returncode==expected,(argv,p.returncode,p.stderr)
 parsed=json.loads(p.stdout)
 assert parsed.get('new_gpu_runs')==0 and parsed.get('gpu_initialized') is False
 rows.append(dict(argv=argv,expected_exit=expected,actual_exit=p.returncode,stdout=ref(log),status=parsed['status'],GPU_runs=0,GPU_initialized=False))
after={name:ref(ROOT/name) for name in before}
assert before==after
for name in ('gpu-source-lock.json','test-input-lock.json'):
 frozen=json.loads((OUT/name).read_text())['files']
 assert all(ref(ROOT/r['path'])==r for r in frozen)
result=dict(status='PASS_FINAL_SOURCE_LOCKED_CPU_PREVIEWS_AND_LAUNCH_DENIAL',commands=rows,source_and_ledger_before=before,source_and_ledger_after=after,
 source_files_unchanged=True,ledger_unchanged=True,new_GPU_runs=0,GPU_initialized=False,GPU_availability_probed=False,guard_invoked=False,
 scope='Pure CPU previews and denied outer launch only; no GPU execution authorization')
with (OUT/'gpu-next-day/FINAL_CPU_CLI_RESULTS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
cases=ET.parse(OUT/'current-integration-01/cpu.xml').findall('.//testcase')
skips=collections.Counter()
new_tests=[]
for c in cases:
 skip=c.find('skipped')
 if skip is not None:skips[skip.attrib.get('message','')]+=1
 if 'prefix_io_v1_p4_' in c.attrib.get('classname',''):new_tests.append(dict(classname=c.attrib.get('classname'),name=c.attrib.get('name'),
  status='skipped' if skip is not None else 'failed' if c.find('failure') is not None or c.find('error') is not None else 'passed'))
assert len({(x['classname'],x['name']) for x in new_tests})==len(new_tests)
counts=collections.Counter(x['status'] for x in new_tests)
summary=dict(status='FINAL_UNIQUE_JUNIT_READ_ONLY_SUMMARY',total_unique=len(cases),P4_current_directory_unique=len(new_tests),
 P4_current_directory_status=dict(counts),skip_reasons=dict(skips),source_XML=ref(OUT/'current-integration-01/cpu.xml'),
 microbench=ref(OUT/'current-integration-01/cpu-observation-overhead.json'),
 repeated_agent_runs_added=False,historical_12_fixtures_rerun=False)
with (OUT/'CPU_UNIQUE_TEST_SUMMARY.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],commands=len(rows),summary=summary)))
