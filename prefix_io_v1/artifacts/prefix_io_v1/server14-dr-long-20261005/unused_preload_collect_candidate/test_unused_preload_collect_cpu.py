"""CPU-only routing of the actual old/new native dispatch function; no GPU claims."""
import ast
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / 'experiments/prefix_io_v1/gpu-budget-ledger.json').is_file()), HERE.parents[2])
if str(ROOT).startswith('/root/'):
    CONTROL = ROOT / 'artifacts/prefix_io_v1/server14-dr-long-20261005/runtime_source_v2/control'
    BASE = ROOT / 'artifacts/prefix_io_v1/server14-dr-long-20261005/runtime_source_v2/native/py_kvcache/reactor.py'
else:
    CONTROL = None
    BASE = HERE.parent / 'bounded_d_r_runtime_20261005/native/py_kvcache/reactor.py'
if CONTROL is not None:
    sys.path.insert(0, str(CONTROL))

def function(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'IoReactor')
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_prefix_stage_decide')
    scope = dict(time=SimpleNamespace(monotonic_ns=lambda: 1_000_000_000),
        _ReadyFd=type('UnusedReadyFd', (), {}),
        _StageDispatch=lambda decision, work_id: SimpleNamespace(decision=decision, work_id=work_id))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope[fn.name]

class Controller:
    run_id = 'cpu-routing-only'
    def __init__(self):
        self.calls = []
        self.errors = []
    def decide(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(action='issue')
    def fail(self, error):
        self.errors.append(error)

class Owner:
    def __init__(self):
        self._prefix_dispatch_controller = Controller()
        self._prefix_p4_bridge = SimpleNamespace(run_id='cpu-routing-only',
            policy=SimpleNamespace(single_file=None), fail=lambda error: self.errors.append(error))
        self._prefix_stage_accounting = SimpleNamespace(valid=False)
        self._stop = False
        self.collect_calls = 0
        self.errors = []
    def _prefix_clean_reclaimable_bytes(self):
        return 10_000_000
    def accepted_parent_count(self):
        return 0
    def is_mandatory(self, future):
        return False
    def _prefix_p4_collect(self):
        self.collect_calls += 1
        return None  # Explicitly unknown observation, never a qualified policy input.

class RoutingTests(unittest.TestCase):
    def test_only_dispatch_function_changes(self):
        before = ast.parse(BASE.read_text(encoding='utf-8'))
        after = ast.parse((HERE / 'native/py_kvcache/reactor.py').read_text(encoding='utf-8'))
        def remove(tree):
            cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'IoReactor')
            cls.body = [x for x in cls.body if not isinstance(x, ast.FunctionDef) or x.name != '_prefix_stage_decide']
            return ast.dump(tree, include_attributes=False)
        self.assertEqual(remove(before), remove(after))

    def test_unused_observation_removed_and_controller_preserved(self):
        old, new = function(BASE), function(HERE / 'native/py_kvcache/reactor.py')
        a, b = Owner(), Owner()
        ra = old(a, 'ssd_read', 917504, ('ssd_read', 1), reserved_bytes=917504)
        rb = new(b, 'ssd_read', 917504, ('ssd_read', 1), reserved_bytes=917504)
        self.assertEqual((a.collect_calls, b.collect_calls), (1, 0))
        self.assertEqual(a._prefix_dispatch_controller.calls, b._prefix_dispatch_controller.calls)
        self.assertEqual((ra.decision.action, ra.work_id), (rb.decision.action, rb.work_id))
        self.assertEqual(a.errors + b.errors + a._prefix_dispatch_controller.errors + b._prefix_dispatch_controller.errors, [])

    def test_progress_and_real_parent_keep_observation(self):
        old, new = function(BASE), function(HERE / 'native/py_kvcache/reactor.py')
        for kwargs in [dict(continuation=True), dict(support=True),
                       dict(job=SimpleNamespace(failed=None, future=object()))]:
            a, b = Owner(), Owner()
            old(a, 'ssd_read', 917504, ('ssd_read', 1), **kwargs)
            new(b, 'ssd_read', 917504, ('ssd_read', 1), **kwargs)
            self.assertEqual((a.collect_calls, b.collect_calls), (1, 1))
            self.assertEqual(a._prefix_dispatch_controller.calls, b._prefix_dispatch_controller.calls)
            self.assertEqual(a.errors + b.errors + a._prefix_dispatch_controller.errors + b._prefix_dispatch_controller.errors, [])

if __name__ == '__main__':
    print(json.dumps(dict(CPU_only=True, GPU_jobs=0, observation='UNKNOWN in routing fixture')))
    unittest.main()
