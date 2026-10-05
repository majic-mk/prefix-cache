"""Run the finite authority rejects under a frozen CPU source lock, without CUDA."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_entry_binding as B


def verify_cpu_lock(root, lock_path):
    root = Path(root).resolve(strict=True)
    lock_path = Path(lock_path)
    if lock_path.is_symlink() or not lock_path.resolve().is_relative_to(root):
        B.reject('CPU source lock outside project or symlink')
    raw = lock_path.read_bytes()
    document = B.parse(raw)
    if (document.get('schema') != 'c5_gpu_entry_cpu_source_lock_v1'
            or document.get('gpu_uuid') is not None
            or document.get('gpu_launch_allowed') is not False
            or document.get('native_execution_verified') is not False
            or document.get('native_cost_qualified') is not False
            or not isinstance(document.get('files'), list)
            or not 1 <= len(document['files']) <= 512):
        B.reject('specific bounded CPU source lock required')
    refs = B.checked_rows(root, document['files'])
    for name in ('run_cpu_entry_binding.py', 'test_gpu_entry_binding.py',
                 'gpu_entry_binding.py', 'control_native_cost_job.py'):
        relative = Path(__file__).with_name(name).resolve().relative_to(root).as_posix()
        if relative not in refs or refs[relative]['bytes'] <= 0:
            B.reject('actual CPU binding source missing from frozen closure')
    actual = [refs[key] for key in sorted(refs)]
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return digest, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--source-lock', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--location', required=True, choices=['local_cpu', 'server_cpu'])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    source_before, lock_sha = verify_cpu_lock(args.project_root, args.source_lock)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('c5_cpu_gpu_entry_binding_tests',
                                                Path(__file__).with_name('test_gpu_entry_binding.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    expected_tests = suite.countTestCases()
    with (args.output_dir / 'TEST_STDERR.log').open('x', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            suite)
    source_after, lock_sha_after = verify_cpu_lock(args.project_root, args.source_lock)
    forbidden = sorted(name for name in sys.modules if name.split('.')[0] in
                       ('torch', 'vllm', 'ray', 'kvcache', 'py_kvcache', 'pynvml'))
    attempts = list(module.FORBIDDEN_IMPORTS)
    success = (result.wasSuccessful() and not result.skipped and result.testsRun == expected_tests
               and expected_tests > 0 and source_before == source_after
               and lock_sha == lock_sha_after and not forbidden and not attempts)
    value = dict(status='PASS' if success else 'FAIL', tests=result.testsRun,
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_before=source_before, source_after=source_after, source_lock_sha256=lock_sha,
        location=args.location, gpu_runs=0, actual_gpu_runs=0, gpu_uuid=None,
        gpu_launch_allowed=False, native_execution_verified=False,
        native_cost_qualified=False, full_runtime_cost_qualified=False,
        on_observation_cost_measured=False, valid_native_receipt=None,
        effective_cost_upper_ns=None, effective_step_budget_ns=None,
        forbidden_imports=forbidden, forbidden_import_attempts=attempts,
        tests_are_CPU_boundary_checks=True, test_suite_source='test_gpu_entry_binding.py',
        original_engine_tests_or_benchmarks_rerun=False)
    with (args.output_dir / 'CPU_BINDING_RESULT.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(value))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
