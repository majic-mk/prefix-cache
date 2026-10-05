import argparse,importlib.util,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description='Append one metadata-bound exact-cell config; model imports are forbidden')
p.add_argument('--project-root',type=Path,required=True)
p.add_argument('--wrapper',required=True)
p.add_argument('--source-version',type=int,required=True)
p.add_argument('--attempt',type=int,required=True)
a=p.parse_args();root=a.project_root.resolve(strict=True)
rent='artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004';delivery='artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005'
spec=importlib.util.spec_from_file_location('_bounded_actual_cost_prepare',root/rent/'runner'/a.wrapper)
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W);R=W.driver_module()
tag=f'raw{a.attempt:02d}';job=f'server12-strong-exact-cal{a.attempt:02d}'
template=rent+'/'+a.wrapper.removesuffix('.py').upper()+'_CPU_PLAN_TEMPLATE.json'
old=R.read(root/delivery/'raw01/CONFIG.json')
directory=root/delivery/tag
assert not directory.exists()
directory.mkdir()
source_lock=R.ref(root,rent+f'/PRERENT_SOURCE_LOCK_V{a.source_version}.json')
source_proof=R.ref(root,rent+f'/PRERENT_SOURCE_PROOF_V{a.source_version}.json')
bound=W.bind_live_plan(root,template_ref=R.ref(root,template),source_lock_ref=source_lock,source_proof_ref=source_proof,
 off_qualification_ref=old['off_qualification_ref'],off_guard_ref=old['off_guard_ref'],
 permissions_ref=old['permissions_ref'],gpu_uuid=old['gpu_uuid'],output=directory/'BOUND_PLAN.json')
intent=R.read(root/delivery/'raw01/PRELAUNCH_INTENT.json')
intent.update(job_id=job,plan_ref=bound,source_lock_ref=source_lock,
 preparation_source_ref=R.ref(root,delivery+'/prepare_actual_exact_attempt_v1.py'),
 cached_semantics_ref=R.ref(root,template))
R.new_json(directory/'PRELAUNCH_INTENT.json',intent)
config=W.make_actual_configuration(root,bound_plan_ref=bound,source_proof_ref=source_proof,
 intent_ref=R.ref(root,delivery+'/'+tag+'/PRELAUNCH_INTENT.json'),permissions_ref=old['permissions_ref'],
 off_qualification_ref=old['off_qualification_ref'],off_guard_ref=old['off_guard_ref'],
 output_relative='experiments/prefix_io_v1/runs/'+job+'/details',storage_template_relative=old['storage_template_relative'],
 output=directory/'CONFIG.json')
assert not any(x=='torch' or x.startswith(('torch.','vllm','py_kvcache')) for x in sys.modules)
print(json.dumps(dict(status='ACTUAL_CPU_CONFIG_WITH_LIVE_DEVICE_METADATA_BOUND',plan_ref=bound,
 config_ref=config,model_execution_started=False,formal_performance_experiment=False)))
