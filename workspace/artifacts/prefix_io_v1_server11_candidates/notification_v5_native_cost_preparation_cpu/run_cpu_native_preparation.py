"""Bounded explicit CPU replay under the supplied source lock; no native grant."""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
SCHEMA = 'c5_native_cost_preparation_cpu_source_lock_v1'


def require(condition, message):
    if not condition: raise ValueError(message)


def relative(root, name):
    require(type(name) is str and name and ':' not in name and '\\' not in name and
        not name.startswith('/') and all(part not in ('', '.', '..') for part in name.split('/')),
        'bounded project relative path')
    path = root
    for part in name.split('/'):
        path = path / part
        require(not path.is_symlink(), 'source symlink refused')
    require(path.resolve().is_relative_to(root), 'project path escape')
    return path


def read_lock(path):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 4 * 1024**2,
        'bounded source lock')
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate lock key'); result[key] = value
        return result
    raw = path.read_bytes()
    doc = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite lock')))
    require(doc.get('schema') == SCHEMA or doc.get('scope') == SCHEMA, 'CPU source lock schema')
    for name in ('gpu_launch_allowed', 'native_execution_verified', 'native_cost_qualified',
                 'full_runtime_cost_qualified', 'on_observation_cost_measured'):
        require(doc.get(name) is False, 'CPU source lock must refuse ' + name)
    require(doc.get('gpu_uuid') is None, 'CPU lock GPU UUID unresolved')
    rows = doc.get('files')
    require(type(rows) is list and 1 <= len(rows) <= 512, 'bounded finite source lock')
    require(len({row['path'] for row in rows}) == len(rows), 'unique source lock paths')
    return rows, sha256(raw).hexdigest()


def verify(root, rows):
    values = []
    for row in rows:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
            type(row['bytes']) is int and 0 < row['bytes'] <= 10 * 1024**2 and
            type(row['sha256']) is str and len(row['sha256']) == 64 and
            all(char in '0123456789abcdef' for char in row['sha256']), 'exact bounded source ref')
        path = relative(root, row['path'])
        require(path.is_file() and path.stat().st_size == row['bytes'], 'source size drift: ' + row['path'])
        raw = path.read_bytes()
        actual = dict(path=row['path'], bytes=len(raw), sha256=sha256(raw).hexdigest())
        require(actual == row, 'source SHA drift: ' + row['path'])
        values.append(actual)
    return sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--source-lock', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--location', default='server_cpu', choices=('server_cpu', 'local_cpu'))
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--native-root', type=Path)
    parser.add_argument('--original-estimator', type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    source_lock = args.source_lock.resolve(strict=True)
    require(source_lock.is_relative_to(root), 'source lock outside project')
    output = args.output_dir.resolve()
    require(output.is_relative_to(root) and not output.exists(), 'new project output dir required')
    rows, lock_sha = read_lock(source_lock)
    for filename in ('run_cpu_native_preparation.py', 'test_native_preparation.py', 'synthetic_raw_fixture.py',
                     'native_preparation_contract.py', 'native_conditional_cost.py',
                     'prepare_and_verify_native_cost.py', 'run_native_cost_experiment.py',
                     'control_native_cost_job.py', 'SOURCE_INHERITANCE.json'):
        relative_path = (HERE / filename).relative_to(root).as_posix()
        require(sum(row['path'] == relative_path for row in rows) == 1, 'own code unfrozen: ' + filename)
    before = verify(root, rows)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    candidate = args.candidate_root or root / 'artifacts/prefix_io_v1/server11-p4-notification-candidate-v5-cpu-20261003'
    native = args.native_root or root / 'artifacts/prefix_io_v1/server11-native-cost-v6-20261003'
    original = args.original_estimator or root / 'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py'
    for key, value in (('C5_FROZEN_CANDIDATE', candidate), ('C5_NATIVE_PARENT', native), ('C5_ORIGINAL_ESTIMATOR', original)):
        os.environ[key] = str(value.resolve(strict=True))
    name = '_c5_cpu_native_preparation_target_tests'
    spec = importlib.util.spec_from_file_location(name, HERE / 'test_native_preparation.py')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'TEST_STDOUT.log').open('x', encoding='utf-8', newline='\n') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    after = verify(root, rows)
    imports = sorted(name for name in sys.modules if name == 'torch' or name == 'vllm' or
        name.startswith(('torch.', 'vllm.', 'py_kvcache.')))
    require(imports == [], 'GPU/model/native backend imported')
    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    doc = dict(schema='c5_native_cost_preparation_cpu_result_v1',
        status='PASS_C5_NATIVE_COST_CPU_PREPARATION_ONLY' if result.wasSuccessful() and not result.skipped else 'FAIL_C5_NATIVE_COST_CPU_PREPARATION',
        tests=result.testsRun, passed=passed, failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_before=before, source_after=after, source_count=len(rows), source_lock_sha256=lock_sha,
        location=args.location, gpu_uuid=None, actual_gpu_runs=0, gpu_runs=0, gpu_launch_allowed=False,
        native_execution_verified=False, native_cost_qualified=False, full_runtime_cost_qualified=False,
        on_observation_cost_measured=False, valid_native_receipt=None, effective_cost_upper_ns=None,
        effective_step_budget_ns=None, production_qualified=False, performance_claim=False,
        torch_vllm_imports=imports, synthetic_raw_test_origin='synthetic_cpu_contract',
        actual_model_processes_started=0, original_numerical_ast_changed=False)
    with (output / 'CPU_RESULT.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(doc, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(doc, sort_keys=True))
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == '__main__':
    raise SystemExit(main())
