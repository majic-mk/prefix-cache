"""Real unauthorized normal entry check on CPU; never loads a model."""
import hashlib, importlib.abc, json, os, pathlib, runpy, sys, traceback
ROOT = pathlib.Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = ROOT/'artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004'
OUT = ROOT/'artifacts/prefix_io_v1/server11-c5-normal-native-delivery-20261004/UNAUTHORIZED_ENTRY_CPU_RESULT.json'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ledger = ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json'
before = ledger.read_bytes()
assert json.loads(before)['active_reservation'] is None
absent = ['AUTHORITY_off.json','HUMAN_GPU_GRANT_off.json','EFFECTIVE_GPU_PERMISSION_off.json','SCOPE_off.json','LAUNCH_INTENT_off.json']
assert all(not (D/name).exists() for name in absent)
assert not (ROOT/'experiments/prefix_io_v1/runs/server11-c5-normal-notification-off-gpu01').exists()
lock_raw = (D/'COMMON_SOURCE_LOCK.json').read_bytes()
lock = json.loads(lock_raw)
target = D/'run_p4_single_file_experiment.py'
relative = target.relative_to(ROOT).as_posix()
row = next(x for x in lock['files'] if x['path'] == relative)
assert hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']
attempts = []
class DenyModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','vllm','py_kvcache'):
            attempts.append(fullname)
            raise RuntimeError('CPU forbidden model import attempted: ' + fullname)
finder = DenyModelImports()
sys.meta_path.insert(0,finder)
sys.argv = [str(target),'--execute','--config',str(D/'CONFIG_off.json')]
error = None
blocked_reference = None
try:
    runpy.run_path(str(target),run_name='__main__')
except Exception as exc:
    error = {'type':type(exc).__name__, 'message':str(exc)}
    current = exc.__traceback__
    while current:
        frame = current.tb_frame
        if frame.f_code.co_name == 'read' and pathlib.Path(frame.f_code.co_filename) == D/'control_p4_single_file.py':
            blocked_reference = frame.f_locals.get('relative')
        current = current.tb_next
else:
    raise RuntimeError('unauthorized entry did not reject')
finally:
    sys.meta_path.remove(finder)
assert error == {'type':'ValueError','message':'NORMAL_ENTRY_REJECTED: bounded JSON evidence'}, error
assert blocked_reference == (D/'AUTHORITY_off.json').relative_to(ROOT).as_posix(), blocked_reference
imports = sorted(x for x in sys.modules if x.split('.')[0] in ('torch','vllm','py_kvcache'))
assert not attempts and not imports
assert ledger.read_bytes() == before
assert (D/'COMMON_SOURCE_LOCK.json').read_bytes() == lock_raw
assert all(not (D/name).exists() for name in absent)
assert not (ROOT/'experiments/prefix_io_v1/runs/server11-c5-normal-notification-off-gpu01').exists()
result = dict(schema='normal_real_unauthorized_entry_CPU_result_v1', status='PASS_REJECTED_BEFORE_MODEL_OR_GPU',
    origin='actual_frozen_normal_entry_CPU_invocation', target_ref=row, config_relative='artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004/CONFIG_off.json',
    source_lock_sha256=hashlib.sha256(lock_raw).hexdigest(), rejection=error, rejected_reference=blocked_reference,
    absent_authorization_files=absent, forbidden_import_attempts=attempts, forbidden_imports=imports,
    gpu_ledger_sha256_before=hashlib.sha256(before).hexdigest(), gpu_ledger_sha256_after=hashlib.sha256(ledger.read_bytes()).hexdigest(),
    GPU_jobs=0, model_loads=0, new_run_directory_created=False, normal_runtime_qualified=False, strategy_effect_verified=False)
with OUT.open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2,sort_keys=True)
    stream.write('\n')
print(json.dumps(result,sort_keys=True))

