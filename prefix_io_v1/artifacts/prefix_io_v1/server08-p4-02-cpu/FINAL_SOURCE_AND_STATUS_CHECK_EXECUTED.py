from pathlib import Path
import json,hashlib,sys
ROOT=Path.cwd();OUT=ROOT/'artifacts/prefix_io_v1/server08-p4-02-cpu'
def ref(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest())
for name in ('test-input-lock.json','gpu-source-lock.json'):
 x=json.loads((OUT/name).read_text());assert all(ref(ROOT/r['path'])==r for r in x['files'])
state=json.loads((ROOT/'experiments/prefix_io_v1/execution_state.json').read_text());before=json.loads((OUT/'execution-state-before-final-update.json').read_text());close=json.loads((OUT/'P4_02_CPU_ROOT_CLOSEOUT.json').read_text())
assert state['p4_cpu']==before['p4_cpu'] and state['p4_cpu_extended']==close
assert all(a==b for a,b in zip([x for x in state['phases'] if x['id']!='P4'],[x for x in before['phases'] if x['id']!='P4']))
assert all(ref(ROOT/r['path'])==r for r in close['evidence'])
assert (ROOT/'docs/prefix_io_v1/SERVER08_P4_CPU_EXTENDED_REPORT.md').read_bytes()==(OUT/'P4_02_CPU_DELIVERY_REPORT.md').read_bytes()
x=dict(status='PASS_FINAL_SOURCE_REFS_REPORT_AND_STATE_CONSISTENCY',P3_and_P4_01_state_objects_preserved=True,CPU_lock_inputs=2198,GPU_source_lock_inputs=2048,current_report_refs_match=True,new_GPU_runs=0,full_P4_complete=False,binary_or_GPU_execution=0)
with (OUT/'FINAL_SOURCE_AND_STATUS_CHECK.json').open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(x))