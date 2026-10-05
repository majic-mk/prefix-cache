"""CPU branch fixtures only: no actual GPU cost input, receipt, or activation.

Synthetic in-memory tables below deliberately bypass the private loader solely
to exercise its arithmetic consumer. They are never written as evidence or
presented to an engine, owner resource, GPU validator, or actual raw-pair factory.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import replace
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
import weakref

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--original-control-root", type=Path)
args, rest = parser.parse_known_args()
ORIGINAL = args.original_control_root or (HERE.parents[2] / "artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/activation/source/prefix_io_control")
assert ORIGINAL.is_dir(), "provide the original CPU control source directory"


class SourceOnly(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "prefix_io_control":
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        if fullname.startswith("prefix_io_control."):
            leaf = fullname.split(".")[-1] + ".py"
            for directory in (HERE / "control", ORIGINAL):
                source = directory / leaf
                if source.is_file():
                    return importlib.util.spec_from_loader(fullname, self)
            raise ImportError("missing original CPU source: " + fullname)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        if module.__name__ == "prefix_io_control":
            module.__path__ = [str(HERE / "control"), str(ORIGINAL)]
            module.__file__ = str(ORIGINAL / "__init__.py")
            return
        leaf = module.__name__.split(".")[-1] + ".py"
        source = next(d / leaf for d in (HERE / "control", ORIGINAL) if (d / leaf).is_file())
        module.__file__ = str(source)
        exec(compile(source.read_bytes(), str(source), "exec", dont_inherit=True), module.__dict__)


sys.meta_path.insert(0, SourceOnly())
from prefix_io_control import p4_cost_table as C
from prefix_io_control import p4_policy as P
from prefix_io_control import p4_bridge as B
from prefix_io_control.p4_types import P4Config, SystemSnapshot, WorkDescriptor
from prefix_io_control.dispatch_budget import Amount, ZERO
from prefix_io_control.dispatch_shadow import ShadowState


def source_module(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


F = source_module("_TEST_ONLY_finite_binding", HERE / "runner/finite_current_binding.py")


def test_only_table(a=10, b=20):
    key = ("TEST_ONLY_CPU", "a" * 64, "b" * 64, "c" * 64, 1, 1, 0, 784, 917504)
    cell = C.CostCell(key, ZERO, "ssd_read", 917504, "existing_io_plus_delta", a, max(0, b - a), 0)
    table = C.CostTable((cell,), scope="conditional", source_sha256="d" * 64)
    # Explicit memory fixture, not a loader/actual-GPU evidence test.
    value = C.DevelopmentCostInput(key[0], key[1], "e" * 64, key[2], key[3], "eager", "f" * 64,
        "1" * 64, "2" * 64, (), (key,), a, a, b)
    object.__setattr__(table, "scope", "gpu_development")
    object.__setattr__(table, "_development_input", value)
    return table


def values(table, *, age=100, inflight=ZERO):
    snapshot = SystemSnapshot("TEST_ONLY", 0, age, frozenset(("native_ready_work",)),
        native_state=ShadowState("TEST_ONLY", age, inflight), load_signature=table.cells[0].load_signature)
    work = WorkDescriptor("TEST_ONLY", 0, 0, "preload:TEST_ONLY", "ssd_read", 917504, 0, age,
        False, minimum_unit_bytes=917504)
    return snapshot, work


class DevelopmentCPU(unittest.TestCase):
    def test_missing_actual_pair_fails_before_validator(self):
        called = []
        driver = types.SimpleNamespace(validate_actual_development_pair=lambda *a, **k: called.append(True))
        row = dict(path="MISSING_TEST_ONLY_INPUT.json", bytes=3, sha256="a" * 64)
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_cpu_") as root:
            with self.assertRaisesRegex(ValueError, "missing"):
                C.load_gpu_development_table(root, {row["path"]: row}, row, driver=driver,
                    expected_gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
                    expected_common_runtime_domain_sha256="b" * 64)
        self.assertEqual(called, [])

    def test_caller_scope_cannot_create_actual_development_table(self):
        with self.assertRaises(ValueError):
            C.CostTable((), scope="gpu_development", source_sha256="a" * 64)
        with self.assertRaises(ValueError):
            C.CostTable((), scope="gpu_verified_exact_cells", source_sha256="a" * 64)

    def test_exact_cell_flags_and_unsupported_conditions(self):
        table = test_only_table()
        cell = table.cells[0]
        result = table.lookup(*cell.key, execution="gpu_development")
        self.assertEqual(result.total_ns, 20)
        self.assertFalse(result.mock_only)
        self.assertFalse(result.production_qualified)
        self.assertFalse(table.production_qualified)
        self.assertIsNone(table.lookup(*cell.key, execution="production"))
        self.assertIsNone(table.lookup(*cell.key, execution="cpu_mock"))
        for key in (
            (cell.load_signature[:-2] + (785, 917504), ZERO, "ssd_read", 917504),
            (cell.load_signature, (Amount(1, 917504),) + ZERO[1:], "ssd_read", 917504),
            (cell.load_signature, ZERO, "h2d", 917504),
            (cell.load_signature, ZERO, "ssd_read", 1835008)):
            self.assertIsNone(table.lookup(*key, execution="gpu_development"))

    def test_original_issue_arithmetic_and_native_overrides(self):
        table = test_only_table()
        snapshot, work = values(table)
        policy = P.make_p4_policy("TEST_ONLY", P4Config("interference", 30, 60, 10), table=table)
        advice = policy.issue_preview(work, snapshot, now_ns=110, expected_epoch=0, execution="gpu_development")
        self.assertEqual((advice.action, advice.predicted_total_ns), ("defer", 20))
        self.assertFalse(advice.production_qualified)
        for override in (replace(work, progress="mandatory"), replace(work, created_ns=1)):
            advice = policy.issue_preview(override, snapshot, now_ns=110, expected_epoch=0, execution="gpu_development")
            self.assertEqual(advice.action, "issue")
            self.assertTrue(advice.progress_override)
        stale = policy.issue_preview(work, snapshot, now_ns=200, expected_epoch=0, execution="gpu_development")
        self.assertEqual(stale.action, "native_fallback")
        unsupported = policy.issue_preview(work, replace(snapshot, load_signature=("UNMEASURED_TEST_ONLY",)), now_ns=110,
            expected_epoch=0, execution="gpu_development")
        self.assertEqual(unsupported.action, "native_fallback")
        issue_table = test_only_table(a=20, b=10)
        snapshot, work = values(issue_table)
        policy = P.make_p4_policy("TEST_ONLY", P4Config("interference", 30, 60, 20), table=issue_table)
        self.assertEqual(policy.issue_preview(work, snapshot, now_ns=110, expected_epoch=0,
            execution="gpu_development").action, "issue")

    def test_development_never_opens_production_or_cpu_mock_branch(self):
        table = test_only_table()
        snapshot, work = values(table)
        policy = P.make_p4_policy("TEST_ONLY", P4Config("interference", 30, 60, 10), table=table)
        self.assertEqual(policy.issue_preview(work, snapshot, now_ns=110, expected_epoch=0,
            execution="production").action, "native_fallback")
        self.assertEqual(policy.issue_preview(work, snapshot, now_ns=110, expected_epoch=0,
            execution="cpu_mock").action, "native_fallback")
        for mode in ("shadow", "joint", "dependency_only"):
            policy = P.make_p4_policy("TEST_ONLY", P4Config(mode, 30, 60, 10), table=table)
            with self.assertRaises(ValueError):
                policy.issue_preview(work, snapshot, now_ns=110, expected_epoch=0, execution="gpu_development")

    def test_bridge_preserves_owner_and_rejects_unobserved_tables(self):
        table = test_only_table()
        snapshot, work = values(table)
        bridge = B.make_native_bridge("TEST_ONLY", P4Config("interference", 30, 60, 10), table=table)
        with self.assertRaisesRegex(RuntimeError, "binding"):
            bridge.preview_issue(work, snapshot, now_ns=110)
        bridge.bind()
        self.assertEqual(bridge.preview_issue(work, snapshot, now_ns=110).action, "defer")
        for scope in ("mock_only", "conditional"):
            unobserved = C.CostTable(table.cells, scope=scope, source_sha256="a" * 64)
            with self.assertRaises(ValueError):
                B.make_native_bridge("TEST_ONLY", P4Config("interference", 30, 60, 10), table=unobserved)
        with self.assertRaises(ValueError):
            B.make_native_bridge("TEST_ONLY", P4Config("joint", 30, 60, 10), table=table)

    def test_generation_mismatch_remains_native_fallback(self):
        from prefix_io_control.dependencies import ResourceId
        table = test_only_table()
        snapshot, work = values(table)
        resource = ResourceId("TEST_ONLY", "staging", 0, 0, 1)
        work = replace(work, resource_identity=resource, generation=1)
        snapshot = replace(snapshot, generations=((resource, 2),))
        policy = P.make_p4_policy("TEST_ONLY", P4Config("interference", 30, 60, 10), table=table)
        self.assertEqual(policy.issue_preview(work, snapshot, now_ns=110, expected_epoch=0,
            execution="gpu_development").action, "native_fallback")

    def test_binding_requires_same_actual_identity_and_preserves_no_batch(self):
        table = test_only_table()
        proof, development = F.cost_identity(table, None)
        self.assertIs(proof, table.development_input)
        self.assertTrue(development)
        with self.assertRaises(ValueError):
            F.cost_identity(C.CostTable(table.cells, source_sha256="a" * 64), None)
        class Runner: pass
        runner = Runner()
        identity = F.RuntimeFiniteIdentity(_token=F._TOKEN, signature_prefix=table.cells[0].load_signature[:4],
            source_refs=(), calibration_source_lock_sha256="e" * 64, gpu_uuid="TEST_ONLY_CPU",
            geometry_sha256="c" * 64, collector_source_sha256="1" * 64, cuda_event_source_sha256="2" * 64, runner=runner)
        state = [16, 100, 105, 1, 1, 0, 784]
        class Capture: pass
        capture = Capture()
        capture.runner_ref, capture.origin, capture.valid = weakref.ref(runner), "native_gpu_recording", True
        capture.current_single_file_step = lambda: tuple(state)
        bridge = B.make_native_bridge("TEST_ONLY", P4Config("interference", 30, 60, 10), table=table)
        bridge.bind()
        binding = F.CurrentFiniteBinding(bridge, capture, identity, issuer=None, observation_only=False)
        snapshot, work = values(table)
        self.assertEqual(bridge.preview_issue(work, snapshot, now_ns=110).action, "defer")
        state[-1] = 785
        self.assertEqual(bridge.preview_issue(work, snapshot, now_ns=110).action, "native_fallback")
        state[-1], state[3] = 784, 2
        self.assertEqual(bridge.preview_issue(work, snapshot, now_ns=110).action, "native_fallback")
        self.assertIsNone(bridge.batch_prefix(snapshot, (work,), now_ns=110))
        self.assertTrue(binding.detach())
        self.assertNotIn("preview_issue", vars(bridge))
        self.assertNotIn("batch_prefix", vars(bridge))

    def test_original_lifecycle_and_batch_AST_are_unchanged(self):
        def function(path, cls, name):
            tree = ast.parse(path.read_bytes())
            body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body if cls else tree.body
            return ast.dump(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
        runner = ORIGINAL.parents[2] / "runner"
        for cls, name in ((None, "ordinary_method_source"), ("FiniteStartup", "_factory"),
                          ("FiniteStartup", "detach_after_original_shutdown")):
            self.assertEqual(function(runner / "finite_startup.py", cls, name),
                             function(HERE / "runner/finite_startup.py", cls, name))
        for name in ("batch_prefix", "bind", "_owner"):
            self.assertEqual(function(ORIGINAL / "p4_bridge.py", "NativeP4Bridge", name),
                             function(HERE / "control/p4_bridge.py", "NativeP4Bridge", name))
        self.assertEqual(function(ORIGINAL / "p4_policy.py", "P4Policy", "choose_batch"),
                         function(HERE / "control/p4_policy.py", "P4Policy", "choose_batch"))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + rest)
