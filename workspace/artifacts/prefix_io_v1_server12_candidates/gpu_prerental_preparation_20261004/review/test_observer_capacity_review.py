"""Replay new install keyword values against original event-observer CPU bounds."""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
NEW = HERE.parent / "runner/bounded_native_full_step_collector.py"
OLD = PROJECT / "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/g2_worker_observation.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def observer_constructor():
    tree = ast.parse(OLD.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QueryOnlyEventObserver")
    node = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    namespace = {"require": require}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(OLD), "exec"), namespace)
    return namespace["__init__"]


def actual_install_keywords(max_steps):
    tree = ast.parse(NEW.read_text(encoding="utf-8"))
    install = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "install")
    calls = [node for node in ast.walk(install) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and node.func.attr == "connect_worker_observation"]
    require(len(calls) == 1, "unique original observer connection")
    values = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    return {name: eval(compile(ast.Expression(values[name]), str(NEW), "eval"), {"__builtins__": {}, "min": min}, {"max_steps": max_steps})
            for name in ("max_pending", "max_steps")}


class ObserverCapacityReview(unittest.TestCase):
    def test_actual_new_install_values_fit_original_observer_bounds(self):
        initialize = observer_constructor()
        for maximum in (128, 512, 4096):
            with self.subTest(max_steps=maximum):
                keywords = actual_install_keywords(maximum)
                owner = SimpleNamespace()
                initialize(owner, "cpu-fixture-only", "cpu_fixture", **keywords)
                self.assertEqual(owner.max_steps, maximum)
                self.assertLessEqual(owner.max_pending, 128)

    def test_original_pending_bound_remains_128(self):
        initialize = observer_constructor()
        for pending in (True, 0, 129, 4096, 1.5):
            with self.subTest(max_pending=pending), self.assertRaises(ValueError):
                initialize(SimpleNamespace(), "cpu-fixture-only", "cpu_fixture", max_pending=pending, max_steps=4096)

    def test_original_step_bound_still_accepts_4096(self):
        owner = SimpleNamespace()
        observer_constructor()(owner, "cpu-fixture-only", "cpu_fixture", max_pending=128, max_steps=4096)
        self.assertEqual(owner.max_steps, 4096)
        self.assertEqual(owner.max_pending, 128)


def refs():
    return [dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in (OLD, NEW)]


def main():
    global OLD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-source", type=Path, default=OLD,
                        help="Explicit unchanged original G2 worker observation source; local default is preserved.")
    args = parser.parse_args()
    OLD = args.observer_source.resolve(strict=True)
    require(OLD.is_file() and not OLD.is_symlink(), "original regular CPU observer source")
    before = refs()
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ObserverCapacityReview))
    after = refs()
    passed = result.wasSuccessful() and before == after
    document = dict(schema="independent_original_event_capacity_CPU_review_v1",
        status="PASS_CPU_INSTALL_KEYWORDS_FIT_ORIGINAL_BOUNDS" if passed else "FAIL_CPU_INSTALL_KEYWORD_BOUNDS",
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_refs_before=before, source_refs_after=after, source_bytes_unchanged=before == after,
        replay="unique original QueryOnlyEventObserver.__init__ body plus actual new install keyword AST; no package imports",
        actual_GPU_runs=0, actual_RPC_calls=0, runtime_qualified=False, test_log=log.getvalue())
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: document[key] for key in ("status", "tests_run", "failures", "errors", "skipped", "actual_GPU_runs")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
