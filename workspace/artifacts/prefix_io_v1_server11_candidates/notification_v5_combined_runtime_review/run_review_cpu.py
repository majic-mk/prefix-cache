"""Read-only-source, CPU-only independent combined runtime review."""
from __future__ import annotations
import argparse
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
SCHEMA = 'c5_combined_runtime_cpu_source_lock_v1'
OWN_FILES = ('ACCEPTANCE_CONTRACT.md', 'run_review_cpu.py', 'test_combined_runtime_boundaries.py')
RUNTIME_FILES = ('notification_runtime_adapter.py', 'run_p4_single_file_experiment.py',
                 'control_p4_single_file.py', 'verify_p4_single_file.py')


def require(value, reason):
    if not value: raise ValueError(reason)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def reference(path):
    require(path.is_file() and not path.is_symlink(), 'regular source required')
    require(0 < path.stat().st_size <= 10 * 1024**2, 'bounded source required')
    value = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(value), sha256=sha256(value).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'source-lock', 'stage-root', 'candidate-root',
                 'previous-runtime-root', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--location', choices=('server_cpu', 'local_cpu'), required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    stage = args.stage_root.resolve(strict=True)
    candidate = args.candidate_root.resolve(strict=True)
    previous = args.previous_runtime_root.resolve(strict=True)
    lockpath = args.source_lock.resolve(strict=True)
    lockraw = lockpath.read_bytes()
    lock = json.loads(lockraw, object_pairs_hook=unique)
    require(lock['schema'] == SCHEMA, 'new combined runtime CPU source-lock schema')
    for key in ('gpu_launch_allowed', 'native_execution_verified', 'native_cost_qualified',
                'full_runtime_cost_qualified', 'on_observation_cost_measured'):
        require(lock.get(key) is False, 'CPU-only source-lock field: '+key)
    require(lock.get('gpu_uuid', 'absent') is None, 'unknown GPU UUID must remain null')
    require(type(lock['files']) is list and 1 <= len(lock['files']) <= 500, 'bounded source closure')
    frozen = {}
    for row in lock['files']:
        require(type(row) is dict and set(row) == {'path','bytes','sha256'}, 'exact source row')
        name = row['path']
        require(type(name) is str and not any(c in name for c in ('\\', ':', '\0'))
                and not name.startswith('/') and all(p not in ('','.','..') for p in name.split('/')),
                'bounded project-relative source path')
        path = root
        for part in name.split('/'):
            path /= part
            require(not path.is_symlink(), 'source symlink refused')
        resolved = path.resolve(strict=True)
        require(resolved.is_relative_to(root) and resolved not in frozen, 'confined unique source path')
        actual = reference(path)
        require(type(row['bytes']) is int and actual['bytes'] == row['bytes']
                and actual['sha256'] == row['sha256'], 'source drift: '+name)
        frozen[resolved] = row
    required = {HERE / name for name in OWN_FILES}
    required.update(stage / name for name in RUNTIME_FILES)
    required.add(stage / 'combined_runtime_contract.py')
    required.update(previous / name for name in RUNTIME_FILES)
    required.update(candidate.rglob('*.py'))
    required.add(candidate.parent / 'native_conditional_cost.py')
    require(all(path.resolve(strict=True) in frozen for path in required), 'independent dependency absent from source lock')
    paths = {*frozen, lockpath}
    before = [reference(path) for path in sorted(paths)]
    require(not args.output_dir.exists(), 'new review output required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        C5_COMBINED_REVIEW_STAGE_ROOT=str(stage), C5_COMBINED_REVIEW_CANDIDATE_ROOT=str(candidate),
        C5_COMBINED_REVIEW_PREVIOUS_ROOT=str(previous))
    spec = importlib.util.spec_from_file_location('_independent_combined_boundary',
        HERE / 'test_combined_runtime_boundaries.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    stream = io.StringIO(); started = time.monotonic_ns()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    after = [reference(path) for path in sorted(paths)]
    forbidden = [name for name in sys.modules if name == 'torch' or name.startswith('torch.')
                 or name == 'vllm' or name.startswith('vllm.')]
    passed = result.wasSuccessful() and before == after and not forbidden
    document = dict(status='PASS_COMBINED_RUNTIME_CPU_BOUNDARIES' if passed else 'FAIL',
        tests=result.testsRun, passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_before=before, source_after=after, source_count=len(before),
        source_lock_sha256=sha256(lockraw).hexdigest(), source_lock_files=len(lock['files']),
        gpu_runs=0, model_loads=0, formal_benchmark_runs=0, gpu_jobs_created=0,
        permission_scopes_created=0, gpu_uuid=None, origin='synthetic_cpu_contract', location=args.location,
        gpu_launch_allowed=False, native_execution_verified=False, native_cost_qualified=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        performance_claim=False, valid_native_receipt=None, effective_cost_upper_ns=None,
        effective_step_budget_ns=None, positive_on_native_integration_verified=False,
        full_entry_cpu_timing_qualified=False, forbidden_modules_imported=forbidden,
        elapsed_ns=time.monotonic_ns()-started, command=[sys.executable,'-B','-I','-S']+sys.argv)
    (args.output_dir/'TEST_LOG.log').write_text(stream.getvalue(), encoding='utf-8')
    (args.output_dir/'TEST_RESULT.json').write_text(json.dumps(document, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in document.items() if k not in ('source_before','source_after')}, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
