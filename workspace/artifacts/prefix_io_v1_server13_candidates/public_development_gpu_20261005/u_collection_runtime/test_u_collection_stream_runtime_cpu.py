"""CPU branch/AST spies only; no actual capture, CUDA receipt or qualification."""
import ast
import hashlib
import io
import json
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


V = load_source(HERE / "native_runtime_v7.py", "_mixed_U_runtime_CPU_spy")


class StreamRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mixed_U_runtime_CPU_spy_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        raw = json.dumps({"schema": "CPU_fixture_descriptor_only"}).encode()
        path = self.root / "CPU-fixture-descriptor.json"
        path.write_bytes(raw)
        ref = dict(path=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.refs = {path.name: ref}
        self.config = dict(phase="development", mode="off", arm="U", gpu_uuid="CPU-spy-no-GPU",
                           run_id="CPU-spy-no-real-job", collection_ref=ref)
        self.actual = dict(schema="uncalibrated_original_U_collection_CPU_gate_v1",
            descriptor=json.loads(raw), runtime_refs=self.refs, gpu_uuid=self.config["gpu_uuid"],
            common_runtime_domain_sha256="CPU-spy-domain", table_issued=False, cost_qualified=False,
            ordinary_I_authorized=False, formal_goodput_allowed=False)
        self.gates = dict(config=self.config, refs=self.refs, pair={},
            common_runtime_domain_sha256="CPU-spy-domain", uncalibrated_u_collection=self.actual,
            finite_activation=None)
        self.calls = []
        self.driver = types.SimpleNamespace(require=self.require, read=lambda path: json.loads(path.read_bytes()),
            check_ref=self.check_ref, u_collection_gate=lambda *args: self.actual,
            validate_u_collection_capture=self.validate_spy)

    @staticmethod
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    def check_ref(self, root, row):
        path = root / row["path"]
        raw = path.read_bytes()
        self.require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                     "CPU fixture bytes changed")
        return path

    def validate_spy(self, root, gates, capture, frontend, **kwargs):
        self.calls.append((root, gates, capture, frontend, kwargs))
        return dict(CPU_spy_only=True, cost_qualified=False, exact_finite_cost_cells_qualified=False)

    @staticmethod
    def mixed_fixture():
        return dict(valid=True, frames=[
            dict(native_step_ordinal=3, prepared=dict(context_length=None, exact_cell_eligible=False)),
            dict(native_step_ordinal=4, prepared=dict(context_length=17, exact_cell_eligible=False))])

    def test_U_preserves_every_mixed_frame_and_never_enters_old_exact_cell_join(self):
        capture, frontend, source = self.mixed_fixture(), {"CPU_spy_only": True}, {"CPU_spy_only": True}
        with mock.patch.object(V, "_load_original_reserve_join", side_effect=AssertionError("old cell join forbidden")):
            result = V.validate_formal_capture(self.root, self.gates, capture, frontend,
                                               source_ref=source, driver=self.driver)
        self.assertTrue(result["CPU_spy_only"])
        self.assertEqual(len(self.calls), 1)
        _, _, actual, outputs, kwargs = self.calls[0]
        self.assertIs(actual, capture)
        self.assertIs(outputs, frontend)
        self.assertEqual(kwargs, dict(source_ref=source, expected_ordinals=[3, 4]))
        self.assertIsNone(actual["frames"][0]["prepared"]["context_length"])

    def test_old_I_retains_original_capture_API_and_has_no_new_validator_dependency(self):
        self.config.pop("collection_ref")
        self.config.update(phase="effect", mode="on", arm="I")
        self.gates["uncalibrated_u_collection"] = None
        self.gates["finite_activation"] = {"CPU_old_gate_spy_only": True}
        del self.driver.validate_u_collection_capture
        join_calls = []
        join = types.SimpleNamespace(_capture=lambda *args, **kwargs:
            join_calls.append((args, kwargs)) or {"CPU_original_join_spy_only": True})
        with mock.patch.object(V, "_load_original_reserve_join", return_value=join) as load:
            result = V.validate_formal_capture(self.root, self.gates, self.mixed_fixture(), {},
                                               source_ref={}, driver=self.driver)
        self.assertTrue(result["CPU_original_join_spy_only"])
        self.assertEqual(load.call_args.args[1], self.gates["finite_activation"])
        self.assertEqual(join_calls[0][1]["expected_ordinals"], [3, 4])
        self.assertEqual(self.calls, [])

    def test_invalid_or_discontinuous_stream_is_rejected_before_any_validator(self):
        for capture in (dict(valid=False, frames=[]), dict(valid=True, frames=[]),
                        dict(valid=True, frames=[dict(native_step_ordinal=3), dict(native_step_ordinal=5)]),
                        dict(valid=True, frames=[dict(native_step_ordinal=True)])):
            with self.subTest(capture=capture), self.assertRaises(ValueError):
                V.validate_formal_capture(self.root, self.gates, capture, {}, source_ref={}, driver=self.driver)
        self.assertEqual(self.calls, [])

    def test_new_U_source_preflight_requires_V3_and_both_actual_adapter_dependencies(self):
        original = None
        for root in (Path.cwd(), *HERE.parents):
            for directory in ("artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner",
                    "artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/runner"):
                if (root / directory / "bounded_native_full_step_collector.py").is_file():
                    original = root / directory
                    break
            if original is not None:
                break
        self.assertIsNotNone(original, "actual frozen original collector sources required for CPU source test")
        descriptor = self.actual["descriptor"]
        files = (("collector_ref", HERE / "bounded_native_full_step_collector_v3.py"),
                 ("heterogeneous_frame_adapter_ref", HERE / "heterogeneous_full_step_frame_adapter.py"),
                 ("passive_collector_source_ref", original / "bounded_native_full_step_collector_v2.py"),
                 ("original_collector_source_ref", original / "bounded_native_full_step_collector.py"))
        for key, source in files:
            raw = source.read_bytes()
            relative = "prep/runner/" + source.name
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            row = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            self.refs[relative] = row
            descriptor[key] = row
        raw = json.dumps(descriptor).encode()
        target = self.root / "CPU-fixture-descriptor.json"
        target.write_bytes(raw)
        row = dict(path=target.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.refs[target.name] = self.config["collection_ref"] = row
        self.config["runtime_ref"] = {"path": "prep/runner/native_runtime_v7.py"}
        self.actual["collector_ref"] = descriptor["collector_ref"]
        self.assertEqual(V.preflight_collector_binding(self.root, self.gates, driver=self.driver),
                         descriptor["collector_ref"])
        missing = descriptor.pop("heterogeneous_frame_adapter_ref")
        raw = json.dumps(descriptor).encode()
        target.write_bytes(raw)
        row = dict(path=target.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.refs[target.name] = self.config["collection_ref"] = row
        with self.assertRaisesRegex(ValueError, "dependencies frozen"):
            V.preflight_collector_binding(self.root, self.gates, driver=self.driver)
        descriptor["heterogeneous_frame_adapter_ref"] = missing

    def test_actual_frontend_progress_boundary_queries_once_only_for_new_U(self):
        tree = ast.parse((HERE / "native_runtime_v7.py").read_bytes())
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                        and node.name == "capture_progress")
        calls = []
        capture = types.SimpleNamespace(resolve_after_original_step=lambda: calls.append("query-only CPU spy"))
        namespace = dict(progress=io.StringIO(), counts={}, completions=set(), json=json,
                         collection=self.actual, active_capture=capture)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<actual CPU progress branch>", "exec"), namespace)
        namespace["capture_progress"]([], 1)
        self.assertEqual(calls, ["query-only CPU spy"])
        namespace["collection"] = None
        namespace["active_capture"] = None
        namespace["capture_progress"]([], 2)
        self.assertEqual(calls, ["query-only CPU spy"])

    def test_original_shutdown_finally_helpers_and_drive_call_are_preserved(self):
        old_raw = (HERE / "native_runtime_v6.py").read_bytes()
        self.assertEqual(hashlib.sha256(old_raw).hexdigest(),
                         "c29b8982ac995c81244cb260a2854f31f08fcf5f0126afee938c9aca6a19b44f")
        old, new = ast.parse(old_raw), ast.parse((HERE / "native_runtime_v7.py").read_bytes())
        functions = lambda tree: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        o, n = functions(old), functions(new)
        for name in ("verify_original_tail", "drain_original", "preflight_formal_reserve_replay",
                     "_replay_original_reserve", "prepare_finite_controller", "verify_formal_peer_native_prerequisite",
                     "_load_original_reserve_join"):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(o[name]), ast.dump(n[name]))
        final = lambda node: ast.dump(ast.Module(body=next(part for part in node.body
            if isinstance(part, ast.Try)).finalbody, type_ignores=[]))
        self.assertEqual(final(o["execute"]), final(n["execute"]))
        drive = lambda node: [ast.dump(call) for call in ast.walk(node) if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute) and call.func.attr == "drive_original_engine"]
        self.assertEqual(drive(o["execute"]), drive(n["execute"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
