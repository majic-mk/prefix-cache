from pathlib import Path
import json,hashlib
ROOT=Path.cwd();OUT=ROOT/'artifacts/prefix_io_v1/server08-p4-02-cpu';p=ROOT/'experiments/prefix_io_v1/execution_state.json'
before=p.read_text();state=json.loads(before);close=json.loads((OUT/'P4_02_CPU_ROOT_CLOSEOUT.json').read_text())
with (OUT/'execution-state-before-latest-repair.json').open('x') as f:f.write(before)
updates=dict(latest_cpu_test_evidence='artifacts/prefix_io_v1/server08-p4-02-cpu/current-integration-01/result.json',
 latest_cpu_passed=close['CPU']['unique_counts']['passed'],latest_cpu_skipped=close['CPU']['unique_counts']['skipped'],
 latest_cpu_failed=close['CPU']['unique_counts']['failed'],latest_delivery_report='docs/prefix_io_v1/SERVER08_P4_CPU_EXTENDED_REPORT.md',
 latest_cpu_scope='Current P4-02 broad regression: 1956 passed, 16 skipped, 0 failed; 547 current P4 directory tests; unique counts, no repeated agent totals; historical 12 fixtures separate')
old_fields={k:state.get(k) for k in updates};state.update(updates)
assert state['p4_cpu']==json.loads(before)['p4_cpu'] and state['p4_cpu_extended']==close
assert state['p316']==json.loads(before)['p316']
p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
result=dict(status='PASS_LATEST_CPU_DELIVERY_FIELDS_SYNCHRONIZED',before=old_fields,after=updates,
 P3_and_P4_01_objects_preserved=True,source_implementation_changed=False,GPU_runs=0,
 state_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
with (OUT/'STATE_LATEST_FIELDS_REPAIR.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))