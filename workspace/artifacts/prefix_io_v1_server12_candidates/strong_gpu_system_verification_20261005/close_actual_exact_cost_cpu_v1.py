import argparse,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
from dataclasses import asdict
p=argparse.ArgumentParser(description='Close original native raw and call unchanged actual exact-cell issuer on CPU')
p.add_argument('--project-root',type=Path,required=True)
p.add_argument('--attempt',type=int,required=True)
p.add_argument('--wrapper',required=True)
p.add_argument('--expected-plan-sha256',required=True)
p.add_argument('--expected-guard-sha256',required=True)
a=p.parse_args();root=a.project_root.resolve(strict=True)
rent='artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
delivery='artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005'
tag=f'raw{a.attempt:02d}'; job=f'server12-strong-exact-cal{a.attempt:02d}'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
d=root/delivery/tag; run=root/'experiments/prefix_io_v1/runs'/job
spec=importlib.util.spec_from_file_location('_actual_closed_raw_wrapper',root/rent/'runner'/a.wrapper)
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W);R=W.driver_module()
report=dict(schema='actual_gpu_exact_cell_parent_cost_qualification_report_v1',actual_GPU_operations_this_CPU_action=0,
 source_records_origin='actual_original_guard_and_native_GPU_children',formal_SLO_qualified=False,
 strategy_effect_verified=False,json_report_is_not_reusable_GPU_capability=True)
try:
 plan_ref=R.ref(root,delivery+'/'+tag+'/BOUND_PLAN.json')
 guard_ref=R.ref(root,'experiments/prefix_io_v1/runs/'+job+'/result.json')
 assert plan_ref['sha256']==a.expected_plan_sha256 and guard_ref['sha256']==a.expected_guard_sha256
 ledger=R.read(root/R.LEDGER);assert ledger['active_reservation'] is None
 raw_ref=W.close_actual_raw(root,pending_ref=R.ref(root,'experiments/prefix_io_v1/runs/'+job+'/details/strong-exact-cell-raw-pending-parent-closure.json'),
  plan_ref=plan_ref,before_ref=R.ref(root,delivery+'/'+tag+'/SOURCE_BEFORE.json'),
  after_ref=R.ref(root,delivery+'/'+tag+'/SOURCE_AFTER.json'),guard_ref=guard_ref,output=d/'ACTUAL_RAW_PARENT_CLOSED.json')
 plan=R.read(root/plan_ref['path']);intent_ref=R.ref(root,delivery+'/'+tag+'/PRELAUNCH_INTENT.json')
 sys.path.insert(0,str(root/rent/'activation/source'))
 from prefix_io_control import gpu_cell_issuer as I
 table=I.issue_verified_gpu_table(root,plan_ref=plan_ref,measurements_ref=raw_ref,guard_ref=guard_ref,intent_ref=intent_ref,
  expected_plan_ref=plan_ref,expected_guard_ref=guard_ref,expected_intent_ref=intent_ref,
  expected_runtime_domain_sha256=plan['common_runtime_domain_sha256'],expected_gpu_uuid=plan['gpu_uuid'],
  expected_source_lock_sha256=plan['source_lock_ref']['sha256'],expected_collector_source_ref=plan['collector_source_ref'])
 assert table.production_qualified and I.qualified_identity(table) is not None
 report.update(status='PASS_ACTUAL_EXACT_GPU_COST_CELL',plan_ref=plan_ref,guard_ref=guard_ref,measurements_ref=raw_ref,
  intent_ref=intent_ref,qualified_identity=asdict(I.qualified_identity(table)),cell_count=len(table.cells),
  cells=[dict(asdict(c),total_ns=c.total_ns) for c in table.cells],
  table_scope=table.scope,raw_source_sha256=table.source_sha256,qualification_scope='these_actual_exact_cells_only',
  independent_holdout_used_to_fit=False,resource_release_credit=False)
except Exception as exc:
 report.update(status='ACTUAL_EXACT_GPU_COST_ISSUER_REJECTED',error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(limit=8))
assert not any(x=='torch' or x.startswith(('torch.','vllm','py_kvcache')) for x in sys.modules)
R.new_json(d/'ACTUAL_COST_ISSUER_REPORT.json',report)
print(json.dumps(report))
raise SystemExit(0 if report['status'].startswith('PASS_ACTUAL_') else 1)
