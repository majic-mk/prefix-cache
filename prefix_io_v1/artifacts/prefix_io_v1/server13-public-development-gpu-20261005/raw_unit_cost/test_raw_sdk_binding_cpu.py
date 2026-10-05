"""CPU source/dispatch spies only; no actual SDK layout, CUDA or receipts."""
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load_source(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), vars(module))
    return module


SOURCE = HERE / "raw_sdk_binding.py"
TEMPLATE = load_source(SOURCE, "_raw_SDK_CPU_source_template")


def actual_file(relative, local_fallback):
    for path in [local_fallback, *(root / relative for root in (Path.cwd(), *HERE.parents))]:
        if path.is_file():
            return path
    raise RuntimeError("UNBOUND: actual cloned SDK source/audit required for CPU source test")


class RawSdkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="raw_SDK_CPU_spy_only_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve(strict=True)
        self.refs = {}
        for relative, actual in (
                (TEMPLATE.SOURCE, SOURCE),
                (TEMPLATE.SDK_REF["path"], actual_file(TEMPLATE.SDK_REF["path"],
                    HERE.parent / "u_collection_runtime/site_sdk_migration.py")),
                (TEMPLATE.MIGRATION_REF["path"], actual_file(TEMPLATE.MIGRATION_REF["path"],
                    HERE.parent / "ACTUAL_DRIVER_MIGRATION_AUDIT_01.json"))):
            raw = actual.read_bytes()
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            self.refs[relative] = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.V = load_source(self.root / TEMPLATE.SOURCE, "_raw_SDK_CPU_fixture_" + str(id(self)))
        self.evidence = dict(CPU_spy_only=True, current_driver_ref=dict(self.V.CURRENT_DRIVER),
            migration_ref=dict(self.V.MIGRATION_REF), compiler_proof_current_driver_matched=False,
            production_qualified=False, GPU_model_or_JIT_runtime_qualified=False,
            stubs_on_runtime_library_path=False, compiler_executions_this_action=0,
            GPU_operations_this_action=0, shared_objects_loaded_this_action=False)
        self.calls = []

    def prepare_spy(self, root, refs, out, *, migration_ref):
        self.calls.append((root, refs, out, migration_ref))
        return dict(self.evidence)

    def test_original_three_argument_API_routes_exact_actual_migration_without_authority(self):
        helper = types.SimpleNamespace(prepare_site_sdk=self.prepare_spy)
        with mock.patch.object(self.V, "_load_source", return_value=helper):
            result = self.V.prepare_site_sdk(self.root, self.refs, "CPU-spy-not-an-actual-layout")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0], (self.root, self.refs, "CPU-spy-not-an-actual-layout", self.V.MIGRATION_REF))
        self.assertTrue(result["CPU_spy_only"])
        for key in ("private_cost_table_issued", "ordinary_I_strategy_authorized",
                    "formal_strategy_effect_qualified", "historical_compiler_receipts_refit"):
            self.assertIs(result[key], False)

    def test_actual_source_drift_stops_before_any_SDK_call(self):
        (self.root / self.V.SDK_REF["path"]).write_bytes(b"CPU-spy-intentional-byte-drift")
        with mock.patch.object(self.V, "_load_source", side_effect=AssertionError("must reject before load")):
            with self.assertRaisesRegex(ValueError, "bytes unchanged"):
                self.V.prepare_site_sdk(self.root, self.refs, "CPU-spy-not-an-actual-layout")
        self.assertEqual(self.calls, [])

    def test_new_compile_qualification_or_wrong_driver_is_rejected(self):
        helper = types.SimpleNamespace(prepare_site_sdk=self.prepare_spy)
        for key, value in (("compiler_proof_current_driver_matched", True),
                           ("current_driver_ref", {"CPU_spy_wrong_driver": True}),
                           ("GPU_operations_this_action", True)):
            previous = self.evidence[key]
            self.evidence[key] = value
            with self.subTest(key=key), mock.patch.object(self.V, "_load_source", return_value=helper):
                with self.assertRaisesRegex(ValueError, "never a new compiler or GPU qualification"):
                    self.V.prepare_site_sdk(self.root, self.refs, "CPU-spy-not-an-actual-layout")
            self.evidence[key] = previous

    def test_migration_audit_drift_during_preparation_is_rejected_after_call(self):
        def mutate_spy(*args, **kwargs):
            result = self.prepare_spy(*args, **kwargs)
            (self.root / self.V.MIGRATION_REF["path"]).write_bytes(b"CPU-spy-intentional-after-call-drift")
            return result
        helper = types.SimpleNamespace(prepare_site_sdk=mutate_spy)
        with mock.patch.object(self.V, "_load_source", return_value=helper):
            with self.assertRaisesRegex(ValueError, "bytes unchanged"):
                self.V.prepare_site_sdk(self.root, self.refs, "CPU-spy-not-an-actual-layout")
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
