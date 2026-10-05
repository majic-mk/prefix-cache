"""Read-only CPU scalar/window tests; no mock table becomes a GPU capability."""
from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import weakref

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = load("_finite_current_CPU_readonly", "finite_current_binding.py")
C = load("_bounded_collector_CPU_readonly", "bounded_native_full_step_collector.py")


class RunnerFixture:
    pass


def identity_fixture(runner):
    # Explicit private value fixture tests read-only tuple validation, not
    # actual GPU issuance. It cannot attach without the separate true issuer.
    return B.RuntimeFiniteIdentity(_token=B._TOKEN, signature_prefix=("GPU-cpu", "a" * 64, "b" * 64, "c" * 64),
        source_refs=(), calibration_source_lock_sha256="d" * 64, gpu_uuid="GPU-cpu", geometry_sha256="c" * 64,
        collector_source_sha256="e" * 64, cuda_event_source_sha256="f" * 64, runner=runner)


class FiniteBindingCPU(unittest.TestCase):
    def test_actual_original_event_capacity_accepts4096_with128_pending(self):
        old = HERE.parents[2] / "prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/g2_worker_observation.py"
        if not old.exists():
            old = HERE.parents[1] / "server09-g2-normal-worker-site-cache-v4-final-20261002/g2_worker_observation.py"
        if not old.exists():
            project = HERE.parents[3]
            old = project / "artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/g2_worker_observation.py"
        if not old.exists():
            # This fallback is the actual pinned source copied into the parent
            # artifact, never a mirrored implementation of the capacity rule.
            old = HERE.parents[3] / "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/g2_worker_observation.py"
        tree = ast.parse(old.read_bytes())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QueryOnlyEventObserver")
        node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        cls.body = [node]
        namespace = {"require": C.require}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(old), "exec"), namespace)
        original = namespace["QueryOnlyEventObserver"]("cpu-capacity", "cpu_fixture", max_pending=128, max_steps=4096)
        self.assertEqual((original.max_pending, original.max_steps), (128, 4096))
        with self.assertRaises(ValueError):
            namespace["QueryOnlyEventObserver"]("cpu-capacity", "cpu_fixture", max_pending=4096, max_steps=4096)
        self.assertIn("max_pending=min(128, max_steps)", (HERE / "bounded_native_full_step_collector.py").read_text())
    def test_private_runtime_identity_cannot_be_public_dict_or_boolean(self):
        with self.assertRaisesRegex(ValueError, "only from actual"):
            B.RuntimeFiniteIdentity(signature_prefix=(), source_refs=(), calibration_source_lock_sha256="a" * 64,
                gpu_uuid="GPU-cpu", geometry_sha256="b" * 64, collector_source_sha256="c" * 64,
                cuda_event_source_sha256="d" * 64, runner=RunnerFixture())

    def test_issuer_real_scalar_runtime_reference_shape(self):
        row = ("actual/source.py", 123, "a" * 64)
        self.assertEqual(B.identity_runtime_ref(row), dict(path=row[0], bytes=row[1], sha256=row[2]))
        self.assertEqual(B.identity_runtime_ref(NS(path=row[0], bytes=row[1], sha256=row[2])),
                         dict(path=row[0], bytes=row[1], sha256=row[2]))
        with self.assertRaises(ValueError):
            B.identity_runtime_ref(("actual/source.py", True, "a" * 64))

    def test_eager_identifier_does_not_remove_attention_validation(self):
        self.assertEqual(B.KERNEL, "eager")
        source = (HERE / "finite_current_binding.py").read_text(encoding="utf-8")
        self.assertIn('group.backend.get_name() == "TRITON_ATTN"', source)
        self.assertIn('len(layers) == len(set(layers)) == hf.num_hidden_layers', source)

    def test_current_prepared_decode_scalar_supported_only_exact_batch1(self):
        runner = RunnerFixture()
        identity = identity_fixture(runner)
        state = (16, 100, 110, 1, 1, 0, 527)
        capture = NS(runner_ref=weakref.ref(runner), origin="native_gpu_recording", valid=True,
                     current_single_file_step=lambda: state)
        self.assertEqual(B.current_step(capture, identity, now_ns=120, max_age_ns=50), state)
        for changed in ((16, 100, 110, 2, 2, 0, 527), (16, 100, 110, 1, 1, 1, 527), None):
            capture.current_single_file_step = lambda: changed
            self.assertIsNone(B.current_step(capture, identity, now_ns=120, max_age_ns=50))

    def test_missing_start_query_stale_or_changed_runner_falls_back(self):
        runner, other = RunnerFixture(), RunnerFixture()
        identity = identity_fixture(runner)
        capture = NS(runner_ref=weakref.ref(runner), origin="native_gpu_recording", valid=True,
                     current_single_file_step=lambda: (16, 100, 110, 1, 1, 0, 527))
        self.assertIsNone(B.current_step(capture, identity, now_ns=200, max_age_ns=50))
        capture.runner_ref = weakref.ref(other)
        self.assertIsNone(B.current_step(capture, identity, now_ns=120, max_age_ns=50))
        capture.runner_ref = weakref.ref(runner)
        capture.current_single_file_step = lambda: (16, 100, None, 1, 1, 0, 527)
        self.assertIsNone(B.current_step(capture, identity, now_ns=120, max_age_ns=50))

    def test_cpu_mock_issuer_is_rejected_before_any_bridge_modification(self):
        runner = RunnerFixture()
        identity = identity_fixture(runner)
        capture = NS(runner_ref=weakref.ref(runner))
        bridge = NS(policy=NS(table=NS(production_qualified=False), single_file=None))
        before = vars(bridge).copy()
        with self.assertRaisesRegex(ValueError, "only privately issued real"):
            B.CurrentFiniteBinding(bridge, capture, identity, issuer=NS(qualified_identity=lambda table: None))
        self.assertEqual(vars(bridge), before)

    def test_counter_map_has_fixed_bound(self):
        value = B.CurrentFiniteBinding.__new__(B.CurrentFiniteBinding)
        value.counters = {}
        for number in range(128):
            value.count("reason-" + str(number))
        self.assertLessEqual(len(value.counters), 16)
        self.assertEqual(sum(value.counters.values()), 128)

    def test_original_cuda_window_body_unchanged_and_end_record_closes_it(self):
        old = HERE.parents[1] / "i_pilot_cpu_preparation_20261004/i_bridge/frozen/native_full_step_collector.py"
        if not old.exists():
            old = HERE.parents[1] / "server12-i-pilot-cpu-preparation-20261004/i_bridge/frozen/native_full_step_collector.py"
        def member(source, name):
            tree = ast.parse(source)
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullStepCapture")
            return ast.dump(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name),
                            include_attributes=False)
        previous = old.read_text(encoding="utf-8")
        current = (HERE / "bounded_native_full_step_collector.py").read_text(encoding="utf-8")
        for method in ("current_single_file_step", "after_prepare", "arm_single_file_wait", "detach"):
            self.assertEqual(member(previous, method), member(current, method))
        capture = C.FullStepCapture(run_id="cpu-fixture", origin="cpu_fixture", selected_offsets=(), action=None, max_steps=4096)
        capture.origin = "native_gpu_recording"  # CPU tuple test only, never a qualified issuer.
        begin = NS(record_before_ns=100, completed_query_ns=110)
        end = NS(record_before_ns=None)
        frame = NS(step_kind="decode", batch=1, active_decode=1, prefill_tokens=0, context_length=527)
        pending = dict(phase="prepared", ordinal=16, frame=frame)
        capture.observer = NS(enabled=True, scalar=NS(enabled=True, _adapter=NS(_pending=pending)),
                              events=NS(active=(16, begin, end)))
        self.assertEqual(capture.current_single_file_step(), (16, 100, 110, 1, 1, 0, 527))
        end.record_before_ns = 120
        self.assertIsNone(capture.current_single_file_step())
        begin.completed_query_ns, end.record_before_ns = None, None
        self.assertIsNone(capture.current_single_file_step())

    def test_bounded_observer_default128_and_stream4096_are_not_cost_cells(self):
        original = C.FullStepCapture(run_id="cpu", origin="cpu_fixture", selected_offsets=(16,), action=None)
        self.assertEqual(original.max_steps, 128)
        stream = C.FullStepCapture(run_id="cpu", origin="cpu_fixture", selected_offsets=(), action=None, max_steps=4096)
        self.assertEqual(stream.max_steps, 4096)
        for maximum in (True, 1, 4097):
            with self.assertRaises(ValueError):
                C.FullStepCapture(run_id="cpu", origin="cpu_fixture", selected_offsets=(), action=None, max_steps=maximum)
        with self.assertRaises(ValueError):
            C.FullStepCapture(run_id="cpu", origin="cpu_fixture", selected_offsets=(), action=lambda *args: None, max_steps=4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)
