"""Independent source-locked CPU review; no model or native acquisition."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
TEST_MODULES = ('test_native_preparation_attacks_v2',)
REVIEW_FILES = ('ACCEPTANCE_CONTRACT.md', 'run_review_cpu.py',
                'test_native_preparation_attacks.py', 'run_review_cpu_v2.py',
                'test_native_preparation_attacks_v2.py')
SCHEMA = 'c5_native_cost_preparation_cpu_source_lock_v1'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def reference(path):
    require(path.is_file() and not path.is_symlink(), 'regular source required')
    require(0 < path.stat().st_size <= 10 * 1024**2, 'bounded source required')
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=digest(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'candidate-root', 'native-root', 'stage-root',
                 'original-estimator', 'source-lock', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-lock-sha256', required=True)
    parser.add_argument('--location', choices=('local_windows', 'server_cpu'), required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    candidate = args.candidate_root.resolve(strict=True)
    native = args.native_root.resolve(strict=True)
    stage = args.stage_root.resolve(strict=True)
    original = args.original_estimator.resolve(strict=True)
    lockpath = args.source_lock.resolve(strict=True)
    raw = lockpath.read_bytes()
    require(digest(raw) == args.source_lock_sha256, 'external source-lock SHA mismatch')
    lock = json.loads(raw, object_pairs_hook=unique)
    require(lock['schema'] == SCHEMA, 'new native CPU source-lock schema required')
    for name in ('gpu_launch_allowed', 'native_execution_verified', 'native_cost_qualified',
                 'full_runtime_cost_qualified', 'on_observation_cost_measured'):
        require(lock.get(name) is False, 'CPU-only source-lock boundary: ' + name)
    require('gpu_uuid' in lock and lock['gpu_uuid'] is None, 'unbound GPU UUID required')
    paths = {lockpath}
    frozen = {}
    require(type(lock['files']) is list and 1 <= len(lock['files']) <= 500, 'bounded source closure')
    for row in lock['files']:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'exact source row')
        name = row['path']
        require(type(name) is str and not any(c in name for c in ('\\', ':', '\0'))
                and not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/')),
                'relative source path')
        require(name not in frozen, 'duplicate source path')
        path = root
        for component in name.split('/'):
            path = path / component
            require(not path.is_symlink(), 'source symlink refused')
        require(path.resolve(strict=True).is_relative_to(root), 'source escaped project')
        actual = reference(path)
        require(type(row['bytes']) is int and actual['bytes'] == row['bytes']
                and actual['sha256'] == row['sha256'], 'source drift: ' + name)
        frozen[path.resolve()] = row
        paths.add(path.resolve())
    required = {HERE / name for name in REVIEW_FILES}
    required.update(stage.glob('*.py'))
    required.add(stage / 'SOURCE_INHERITANCE.json')
    require((stage / 'common_candidate/source').is_dir(), 'actual new common overlay required')
    required.update((stage / 'common_candidate').rglob('*.py'))
    required.update((native / name for name in ('native_conditional_cost.py',
        'prepare_and_verify_native_cost.py', 'run_native_cost_experiment.py',
        'control_native_cost_job.py')))
    required.update((candidate / 'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control' / name
                     for name in ('p4_policy.py', 'p4_bridge.py', 'p4_single_file_receipt.py')))
    required.add(original)
    require(all(p.resolve() in frozen for p in required), 'review source dependency missing from lock')
    paths.update(p.resolve() for p in required)
    before = [reference(path) for path in sorted(paths)]
    require(not args.output_dir.exists(), 'new review output directory required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    temporary = args.output_dir / 'TEMP_CASES'
    temporary.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        C5_NATIVE_REVIEW_PROJECT_ROOT=str(root), C5_NATIVE_REVIEW_CANDIDATE_ROOT=str(candidate),
        C5_NATIVE_REVIEW_NATIVE_ROOT=str(native), C5_NATIVE_REVIEW_STAGE_ROOT=str(stage),
        C5_NATIVE_REVIEW_ORIGINAL_ESTIMATOR=str(original), C5_NATIVE_REVIEW_TEMP_ROOT=str(temporary))
    suite = unittest.TestSuite()
    for name in TEST_MODULES:
        spec = importlib.util.spec_from_file_location('_independent_' + name, HERE / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    stream = io.StringIO()
    started = time.monotonic_ns()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.monotonic_ns() - started
    after = [reference(path) for path in sorted(paths)]
    forbidden = [n for n in sys.modules if n == 'torch' or n.startswith('torch.')
                 or n == 'vllm' or n.startswith('vllm.')]
    okay = result.wasSuccessful() and before == after and not forbidden
    document = dict(status='PASS' if okay else 'FAIL', tests=result.testsRun,
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_before=before, source_after=after, source_count=len(before),
        source_lock_sha256=digest(raw), source_lock_files=len(lock['files']),
        sources_unchanged=before == after, gpu_runs=0, gpu_jobs_created=0,
        permission_scopes_created=0, formal_benchmark_runs=0, model_loads=0,
        forbidden_modules_imported=forbidden, gpu_uuid=None, location=args.location,
        origin='synthetic_cpu_contract', elapsed_ns=elapsed, python=sys.version,
        platform=platform.platform(), command=[sys.executable, '-B', '-I', '-S']+sys.argv,
        native_execution_verified=False, native_cost_qualified=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        gpu_launch_allowed=False, performance_claim=False, valid_native_receipt=None,
        effective_cost_upper_ns=None, effective_step_budget_ns=None)
    (args.output_dir / 'TEST_STDERR.log').write_text(stream.getvalue(), encoding='utf-8')
    (args.output_dir / 'TEST_RESULT.json').write_text(
        json.dumps(document, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in document.items() if k not in ('source_before', 'source_after')}, sort_keys=True))
    return 0 if okay else 1


if __name__ == '__main__':
    raise SystemExit(main())
