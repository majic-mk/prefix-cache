"""Run source-locked independent CPU rejection tests for new GPU entry code."""
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
REVIEW_FILES = ('ACCEPTANCE_CONTRACT.md', 'run_review_cpu.py', 'test_gpu_entry_boundaries.py')
FORBIDDEN = ('torch', 'vllm', 'py_kvcache', 'cupy', 'cuda')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def reference(path):
    require(path.is_file() and not path.is_symlink(), 'regular source required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def source_path(root, value):
    require(type(value) is str and not any(c in value for c in ('\\', ':', '\0'))
            and not value.startswith('/') and all(p not in ('', '.', '..') for p in value.split('/')),
            'bounded relative source path')
    path = root
    for part in value.split('/'):
        path /= part
        require(not path.is_symlink(), 'source symlink refused')
    require(path.resolve(strict=True).is_relative_to(root), 'source escaped root')
    return path.resolve()


class CPUImportBoundary:
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + '.') for n in FORBIDDEN):
            self.attempts.append(fullname)
            raise ImportError('CPU review forbids GPU/model/native import: ' + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'source-lock', 'output-dir', 'candidate-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-lock-sha256')
    parser.add_argument('--location', choices=('local_windows', 'server_cpu'), required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    candidate = args.candidate_root.resolve(strict=True)
    require(candidate.is_relative_to(root), 'candidate outside project')
    lockpath = args.source_lock.resolve(strict=True)
    require(lockpath.is_relative_to(root), 'lock outside project')
    raw = lockpath.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if args.source_lock_sha256:
        require(sha == args.source_lock_sha256, 'external lock SHA mismatch')
    lock = json.loads(raw, object_pairs_hook=unique)
    require(type(lock.get('files')) is list and 1 <= len(lock['files']) <= 10000,
            'bounded CPU source closure')
    require(lock.get('gpu_launch_allowed') is False and lock.get('gpu_uuid') is None,
            'CPU lock cannot grant GPU authority')
    for key in ('native_execution_verified', 'native_cost_qualified', 'full_runtime_cost_qualified'):
        if key in lock:
            require(lock[key] is False, 'CPU lock qualification boundary: ' + key)
    paths = {lockpath}
    frozen = {}
    for row in lock['files']:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'exact source row')
        require(row['path'] not in frozen, 'duplicate source path')
        path = source_path(root, row['path'])
        actual = reference(path)
        require(type(row['bytes']) is int and actual['bytes'] == row['bytes']
                and actual['sha256'] == row['sha256'], 'source drift: ' + row['path'])
        frozen[row['path']] = row
        paths.add(path)
    required = {HERE / name for name in REVIEW_FILES}
    required.update(candidate.glob('*.py'))
    required.update((candidate / 'common_candidate').rglob('*.py'))
    original = candidate.parent / 'native_cost_v6/native_conditional_cost.py'
    if not original.exists():
        original = root / 'artifacts/prefix_io_v1/server11-native-cost-v6-20261003/native_conditional_cost.py'
    required.add(original)
    require(all(p.resolve() in paths for p in required), 'review dependency not frozen')
    before = [reference(p) for p in sorted(paths)]
    require(not args.output_dir.exists(), 'new output directory required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Keep fixture paths under Windows MAX_PATH while retaining the exact
    # server-relative input paths. This does not alter any candidate rule.
    temporary = args.output_dir / 'T'
    temporary.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        C5_GPU_ENTRY_REVIEW_PROJECT_ROOT=str(root), C5_GPU_ENTRY_REVIEW_CANDIDATE_ROOT=str(candidate),
        C5_GPU_ENTRY_REVIEW_TEMP_ROOT=str(temporary))
    boundary = CPUImportBoundary()
    sys.meta_path.insert(0, boundary)
    spec = importlib.util.spec_from_file_location('_c5_independent_gpu_entry_tests',
                                                 HERE / 'test_gpu_entry_boundaries.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    started = time.monotonic_ns()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.monotonic_ns() - started
    after = [reference(p) for p in sorted(paths)]
    imported = [n for n in sys.modules if any(n == x or n.startswith(x + '.') for x in FORBIDDEN)]
    okay = result.wasSuccessful() and before == after and not boundary.attempts and not imported
    doc = dict(status='PASS' if okay else 'FAIL', tests=result.testsRun,
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_before=before, source_after=after, source_count=len(before),
        source_lock_sha256=sha, source_lock_files=len(lock['files']), sources_unchanged=before == after,
        location=args.location, python=sys.version, platform=platform.platform(),
        command=[sys.executable, '-B', '-I', '-S']+sys.argv, elapsed_ns=elapsed,
        origin='synthetic_cpu_rejection_review', gpu_uuid=None, gpu_runs=0, model_loads=0,
        forbidden_import_attempts=boundary.attempts, forbidden_modules_imported=imported,
        forbidden_imports=sorted(set(boundary.attempts + imported)),
        gpu_jobs_created=0, permission_scopes_created=0, formal_benchmark_runs=0,
        gpu_launch_allowed=False, native_execution_verified=False, native_cost_qualified=False,
        actual_on_installed=False, full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        valid_native_receipt=None, effective_cost_upper_ns=None, effective_step_budget_ns=None,
        performance_claim=False)
    (args.output_dir / 'TEST_STDERR.log').write_text(stream.getvalue(), encoding='utf-8')
    (args.output_dir / 'TEST_RESULT.json').write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in doc.items() if k not in ('source_before', 'source_after')}, sort_keys=True))
    return 0 if okay else 1


if __name__ == '__main__':
    raise SystemExit(main())
