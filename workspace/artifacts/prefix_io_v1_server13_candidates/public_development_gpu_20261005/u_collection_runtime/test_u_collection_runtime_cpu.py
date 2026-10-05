"""Targeted CPU source/gate fixtures. No model, GPU, cost table or receipt."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def source_module(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


V = source_module(HERE / "native_runtime_v5.py", "_U_collection_runtime_CPU_fixture")


def original_base():
    for root in [Path.cwd(), *HERE.parents]:
        for rel in ("artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004",
                    "artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004"):
            base = root / rel
            if (base / "runner/bounded_native_full_step_collector_v2.py").is_file():
                return base
    raise RuntimeError("UNBOUND: original cloned collector source required")


BASE = original_base()


def original_v4():
    candidates = [BASE / "runner/native_runtime_v4.py"]
    candidates.extend(root / "artifacts/prefix_io_v1_server12_candidates/natural_source_audit_20261005/"
                      "development_runtime_wiring/native_runtime_v4.py" for root in [Path.cwd(), *HERE.parents])
    for path in candidates:
        if path.is_file():
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() == "ce7c22bba2027626fcabbea4df52d94aaea6b1fdf3d5d2b673e879c1680a60bd":
                return raw
    raise RuntimeError("UNBOUND: exact original V4 source required")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="U_collection_CPU_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.refs = {}
        self.checks, self.gate_calls = [], []
        self.config = dict(phase="development", mode="off", arm="U", gpu_uuid="CPU-fixture-no-real-GPU",
                           run_id="CPU-fixture", runtime_ref={"path":"prep/runner/native_runtime_v5.py"})
        descriptor = dict(schema="uncalibrated_original_U_development_collection_v1")
        for name in ("bounded_native_full_step_collector_v2.py", "bounded_native_full_step_collector.py"):
            row = self.put("prep/runner/" + name, (BASE / "runner" / name).read_bytes())
            descriptor["collector_ref" if "_v2" in name else "original_collector_source_ref"] = row
        self.config["collection_ref"] = self.put("fixture-descriptor.json", json.dumps(descriptor).encode())
        self.actual = dict(schema="uncalibrated_original_U_collection_CPU_gate_v1", descriptor=descriptor,
            runtime_refs=self.refs, gpu_uuid=self.config["gpu_uuid"], common_runtime_domain_sha256="fixture-domain",
            collector_ref=descriptor["collector_ref"], table_issued=False, cost_qualified=False,
            ordinary_I_authorized=False, formal_goodput_allowed=False)
        self.gates = dict(config=self.config, refs=self.refs, pair={},
            common_runtime_domain_sha256="fixture-domain", uncalibrated_u_collection=self.actual,
            finite_activation=None)
        self.driver = types.SimpleNamespace(require=self.require, read=lambda p:json.loads(p.read_bytes()),
            check_ref=self.check_ref, u_collection_gate=self.gate)

    @staticmethod
    def require(value, reason):
        if not value:
            raise ValueError(reason)

    def put(self, relative, raw):
        dest = self.root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        row = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.refs[relative] = row
        return row

    def check_ref(self, root, row):
        raw = (root / row["path"]).read_bytes()
        self.require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "byte drift")
        self.checks.append(row["path"])
        return root / row["path"]

    def gate(self, root, config, refs, pair):
        self.gate_calls.append(config["collection_ref"])
        return self.actual

    def test_actual_descriptor_is_replayed_and_exact_original_collector_selected(self):
        row = V.preflight_collector_binding(self.root, self.gates, driver=self.driver)
        self.assertEqual(row, self.actual["collector_ref"])
        self.assertEqual(len(self.gate_calls), 1)
        self.assertIn("prep/runner/bounded_native_full_step_collector.py", self.checks)
        self.assertIn("prep/runner/bounded_native_full_step_collector_v2.py", self.checks)

    def test_no_actual_ref_or_non_U_role_never_calls_gate(self):
        for role in (("effect","off","U"), ("effect","on","I"), ("development","shadow","I")):
            self.config.update(zip(("phase","mode","arm"), role))
            with self.subTest(role=role), self.assertRaises(ValueError):
                V._u_collection_gate(self.root, self.gates, driver=self.driver)
        self.config.pop("collection_ref")
        with self.assertRaisesRegex(ValueError, "metadata cannot replace"):
            V._u_collection_gate(self.root, self.gates, driver=self.driver)
        self.assertEqual(self.gate_calls, [])

    def test_drift_forged_cost_authority_or_stale_metadata_rejected(self):
        for key, value in (("table_issued", True), ("cost_qualified", True), ("gpu_uuid", "other-device")):
            old = self.actual[key]
            self.actual[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                V._u_collection_gate(self.root, self.gates, driver=self.driver)
            self.actual[key] = old
        self.gates["uncalibrated_u_collection"] = dict(self.actual, schema="caller-flag")
        with self.assertRaises(ValueError):
            V._u_collection_gate(self.root, self.gates, driver=self.driver)
        self.gates["uncalibrated_u_collection"] = self.actual
        (self.root / "fixture-descriptor.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "byte drift"):
            V._u_collection_gate(self.root, self.gates, driver=self.driver)

    def test_legacy_gate_has_no_new_driver_dependency(self):
        self.config.pop("collection_ref")
        self.gates["uncalibrated_u_collection"] = None
        del self.driver.u_collection_gate
        self.assertIsNone(V._u_collection_gate(self.root, self.gates, driver=self.driver))

    def test_new_capture_delegates_without_a_private_table(self):
        capture = dict(valid=True, frames=[dict(native_step_ordinal=4)])
        calls = []
        def load(root, gate, *, driver):
            calls.append(gate)
            return types.SimpleNamespace(_capture=lambda *a, **k:dict(CPU_spy_only=True, kwargs=k))
        with mock.patch.object(V, "_load_original_reserve_join", side_effect=load):
            result = V.validate_formal_capture(self.root, self.gates, capture, {},
                source_ref=self.actual["collector_ref"], driver=self.driver)
        self.assertEqual(calls, [dict(descriptor=self.actual["descriptor"], runtime_refs=self.refs)])
        self.assertTrue(result["CPU_spy_only"])
        self.assertEqual(result["kwargs"]["expected_ordinals"], [4])
        with mock.patch.object(V, "_load_original_reserve_join", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "UNKNOWN"):
                V.validate_formal_capture(self.root, self.gates, dict(valid=False), {}, source_ref={}, driver=self.driver)

    def test_original_finally_and_old_cost_helpers_AST_preserved(self):
        old = ast.parse(original_v4())
        new = ast.parse((HERE / "native_runtime_v5.py").read_bytes())
        nodes = lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        o, n = nodes(old), nodes(new)
        for name in ("preflight_formal_reserve_replay", "_replay_original_reserve", "prepare_finite_controller",
                     "verify_formal_peer_native_prerequisite", "verify_original_tail", "_load_original_reserve_join"):
            with self.subTest(name=name):self.assertEqual(ast.dump(o[name]), ast.dump(n[name]))
        final = lambda f:ast.dump(ast.Module(body=next(x for x in f.body if isinstance(x,ast.Try)).finalbody,type_ignores=[]))
        self.assertEqual(final(o["execute"]), final(n["execute"]))

    def test_collection_skips_all_table_sites_and_legacy_keeps_them(self):
        execute = next(n for n in ast.parse((HERE / "native_runtime_v5.py").read_bytes()).body
                       if isinstance(n,ast.FunctionDef) and n.name == "execute")
        body = next(n for n in execute.body if isinstance(n,ast.Try)).body
        required = {"activation_gate_module", "issue", "verify_running_identity"}
        checked = set()
        for branch in (n for n in body if isinstance(n,ast.If)):
            names = {n.func.attr for n in ast.walk(branch) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
            sites = required & names
            if sites:
                condition = compile(ast.Expression(body=branch.test), "<CPU branch inspection>", "eval")
                self.assertFalse(eval(condition, {"config":self.config, "collection":self.actual}))
                for phase in ("development", "effect"):
                    self.assertTrue(eval(condition, {"config":dict(phase=phase), "collection":None}))
                checked.update(sites)
        self.assertEqual(checked, required)


if __name__ == "__main__":
    unittest.main(verbosity=2)
