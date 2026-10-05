"""Record the actual exit of one entry; original guard owns GPU limits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
p=argparse.ArgumentParser()
p.add_argument('--project-root',type=Path,required=True)
p.add_argument('--entry',required=True)
p.add_argument('--tag',required=True)
a=p.parse_args()
root=a.project_root.resolve(strict=True)
d=root/'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
entry=d/a.entry
if entry.parent!=d or entry.is_symlink() or not entry.is_file() or not a.tag.isascii() or '/' in a.tag:
    raise ValueError('contained explicit entry and tag')
def put(suffix,doc):
    with (d/(a.tag+suffix)).open('x',encoding='utf-8') as f:
        json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')
argv=[str(root/'.venv/bin/python'),'-B',str(entry),'launch','--project-root',str(root)]
ledger_path=root/'experiments/prefix_io_v1/gpu-budget-ledger.json'
before=json.loads(ledger_path.read_bytes())
put('_COMMAND.json',dict(argv=argv,cwd=str(root),standing_human_authorization_reused=True,
    maximum_GPU_reserved_seconds=320,UTC=datetime.now(timezone.utc).isoformat(),
    original_guard_owns_GPU_timeout_budget_and_cleanup=True))
start=time.monotonic()
with (d/(a.tag+'_STDOUT.log')).open('xb') as out,(d/(a.tag+'_STDERR.log')).open('xb') as err:
    child=subprocess.Popen(argv,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err)
    put('_PROCESS.json',dict(supervisor_pid=os.getpid(),entry_pid=child.pid,supervisor_session_id=os.getsid(0),
        UTC=datetime.now(timezone.utc).isoformat(),GPU_job_started=False,
        GPU_job_status_must_be_read_from_original_guard_and_ledger=True))
    result=child.wait()
after=json.loads(ledger_path.read_bytes())
put('_RESULT.json',dict(exit_code=result,elapsed_wall_seconds=time.monotonic()-start,
    UTC=datetime.now(timezone.utc).isoformat(),ledger_used_seconds_before=before['gpu_wall_seconds'],
    ledger_used_seconds_after=after['gpu_wall_seconds'],original_guard_GPU_accounted_seconds=after['gpu_wall_seconds']-before['gpu_wall_seconds'],
    original_active_reservation_after=after['active_reservation'],
    status='ENTRY_RETURNED_SUCCESS_REQUIRES_ORIGINAL_GUARD_AND_OUTPUT_JOIN' if result==0 else 'ENTRY_FAILED_SEE_ACTUAL_LOGS',
    performance_gain_verified=False))
