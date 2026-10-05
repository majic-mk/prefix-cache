"""Explicit C5 CPU suite selector; no backend/model import, GPU or source edits."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, default=here)
    parser.add_argument('--author-root', type=Path, required=True)
    parser.add_argument('--previous-root', type=Path, required=True)
    parser.add_argument('--v3-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_root.resolve()
    env = dict(os.environ, SERVER11_AUTHOR_SOURCE_ROOT=str(args.author_root.resolve()),
        SERVER11_PREVIOUS_CANDIDATE_ROOT=str(args.previous_root.resolve()),
        SERVER11_V3_CANDIDATE_ROOT=str(args.v3_root.resolve()))
    # These old source-version assertions are retained verbatim, not edited to
    # accept C5. Its complete replacement AST scope check is a new C5 test.
    empty_source = candidate/'test_empty_p4_window.py'
    cls = next(n for n in ast.parse(empty_source.read_text(encoding='utf-8-sig')).body
               if isinstance(n, ast.ClassDef) and n.name == 'EmptyWindowTests')
    empty_tests = ['EmptyWindowTests.' + n.name for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith('test_') and
        n.name != 'test_only_documented_advice_function_bodies_changed']
    if len(empty_tests) != 7:
        raise ValueError('the seven inherited functional tests must remain unchanged')
    jobs = [('test_single_file_wait.py', [], 26),
            ('test_single_file_retry_observation.py', [], 16),
            ('test_single_file_policy.py', [], 14),
            ('test_single_file_receipt.py', [], 15),
            ('test_single_file_runtime_binding.py', [], 5),
            ('test_reactor_single_file.py', [], 10),
            ('test_enriched_snapshot_reuse.py', [], 13),
            ('test_empty_p4_window.py', empty_tests, 7)]
    rows = []
    for name, selectors, expected in jobs:
        command = [sys.executable, '-B', '-I', '-S', str(candidate/name), *selectors, '-q']
        started = time.monotonic_ns()
        proc = subprocess.run(command, env=env, text=True, encoding='utf-8', errors='replace',
                              capture_output=True, timeout=120)
        matches = re.findall(r'Ran (\d+) tests? in ', proc.stderr)
        count = int(matches[-1]) if matches else None
        passed = proc.returncode == 0 and count == expected
        rows.append(dict(test_file=ref(candidate/name), command=command, tests=count,
            expected_tests=expected, passed=passed, exit_code=proc.returncode,
            wall_ns=time.monotonic_ns()-started, stdout=proc.stdout, stderr=proc.stderr))
        print(f'{name}: {count}/{expected}, passed={passed}', flush=True)
    sources = [candidate/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
               candidate/'native_full_step_collector.py',
               candidate/'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_policy.py']
    result = dict(schema_version=1, scope='C5_notification_candidate_CPU_semantics',
        passed=all(row['passed'] for row in rows), tests=sum(row['tests'] or 0 for row in rows),
        candidate_root=str(candidate), author_root=str(args.author_root.resolve()),
        previous_root=str(args.previous_root.resolve()), v3_root=str(args.v3_root.resolve()),
        source_refs=[ref(path) for path in sources], groups=rows,
        synthetic_events=True, native_physical_io=False, GPU_operations=0,
        GPU_qualified=False, performance_effect_verified=False,
        exclusions=[dict(file='test_empty_p4_window.py',
            test='test_only_documented_advice_function_bodies_changed',
            reason='V1-to-C4 source edit allowlist cannot describe C5; replacement C4-to-C5 AST check runs in test_single_file_wait'),
            dict(file='test_retained_idle_wait.py',
            reason='unchanged V3-to-C4 source diff assertion; run complete 15 tests against frozen C4 separately')],
        notes=['Explicit V3 root is required for the unchanged enriched 156-to-2 comparison.',
               'Original 101 policy/ABI tests are a separate run_original_cpu_regression.py invocation.',
               'This selector claims neither server execution nor original native integration tests.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
