"""Supervise one unchanged GPU guard; record actual exit and ledger change."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
p=argparse.ArgumentParser()
p.add_argument('--project-root',type=Path,required=True)
a=p.parse_args()
root=a.project_root.resolve(strict=True)
d=root/'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005'
tag='GPU_STRONG_RAW_20261005_01'
preflight=json.loads((d/'CPU_STRONG_RAW_BOUND_PREFLIGHT_01_STDOUT.log').read_bytes())
assert preflight['status']=='PASS_CPU_STRONG_RAW_ENTRY_INPUTS'
argv=preflight['original_guard_command']
ledger_path=root/'experiments/prefix_io_v1/gpu-budget-ledger.json'
before=json.loads(ledger_path.read_bytes())
assert before['active_reservation'] is None and before['gpu_wall_seconds']+920<=28800
assert not (root/'experiments/prefix_io_v1/runs/server12-strong-exact-cal01').exists()
probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
assert probe.stdout.strip()==''
def put(suffix,doc):
    with (d/(tag+suffix)).open('x',encoding='utf-8') as f:
        json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')
put('_COMMAND.json',dict(argv=argv,maximum_GPU_reserved_seconds=920,original_guard_owns_limits=True,
    source_before_ref='artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw01/SOURCE_BEFORE.json',
    UTC=datetime.now(timezone.utc).isoformat()))
start=time.monotonic()
with (d/(tag+'_STDOUT.log')).open('xb') as out,(d/(tag+'_STDERR.log')).open('xb') as err:
    child=subprocess.Popen(argv,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err)
    put('_PROCESS.json',dict(supervisor_pid=os.getpid(),guard_pid=child.pid,supervisor_sid=os.getsid(0)))
    code=child.wait()
after=json.loads(ledger_path.read_bytes())
put('_RESULT.json',dict(exit_code=code,elapsed_wall_seconds=time.monotonic()-start,
    ledger_before_used_seconds=before['gpu_wall_seconds'],ledger_after_used_seconds=after['gpu_wall_seconds'],
    original_guard_accounted_seconds=after['gpu_wall_seconds']-before['gpu_wall_seconds'],
    active_reservation_after=after['active_reservation'],performance_gain_verified=False))
