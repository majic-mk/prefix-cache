"""Independent parent full-byte receipts; never import model/native GPU code."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
p=argparse.ArgumentParser()
p.add_argument('--project-root',type=Path,required=True)
p.add_argument('--plan',required=True)
p.add_argument('--phase',choices=('before','after'),required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=a.project_root.resolve(strict=True)
require=lambda ok,why: None if ok else (_ for _ in ()).throw(ValueError(why))
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU parent byte verification')
script=root/'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_trace_runner.py'
spec=importlib.util.spec_from_file_location('_independent_raw_parent_byte_verifier',script)
R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)
plan_ref=R.ref(root,a.plan);plan=R.read(R.check_ref(root,plan_ref))
require(plan['cpu_preparation_only'] is False and plan['gpu_uuid'] is not None,'actual live plan')
start=time.monotonic()
rows=R.source_rows(root,plan['source_lock_ref'],full=True)
ledger=R.read(R.safe(root,R.LEDGER))
require(ledger['active_reservation'] is None,'independent before/after only with idle original guard')
receipt=dict(schema='strong_exact_cell_independent_actual_source_byte_receipt_v1',
 origin='actual_parent_full_file_byte_verification',phase=a.phase,plan_ref=plan_ref,
 source_lock_ref=plan['source_lock_ref'],failed=[],files_verified=len(rows),
 total_bytes_verified=sum(r['bytes'] for r in rows.values()),actual_GPU_runs_this_CPU_action=0,
 model_or_native_backend_imported=False,elapsed_CPU_seconds=time.monotonic()-start,
 UTC=datetime.now(timezone.utc).isoformat(),verifier_ref=R.ref(root,Path(__file__).resolve().relative_to(root).as_posix()),
 original_idle_ledger_ref=R.ref(root,R.LEDGER))
require(not any(n.split('.')[0] in ('torch','vllm','py_kvcache','cupy') for n in sys.modules),'CPU only')
R.new_json(R.safe(root,a.output),receipt)
print(json.dumps(receipt))
