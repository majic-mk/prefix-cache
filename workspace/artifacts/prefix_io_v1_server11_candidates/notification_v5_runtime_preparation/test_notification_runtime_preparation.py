"""CPU ONLY. Source-extracted C5 control methods; SYNTHETIC events/metadata.

Does not call collector.install, actual vLLM, CUDA, or native receipt issuance.
The synthetic fixture's private bridge binding is not native qualification.
"""
import ast
from copy import deepcopy
import gc
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
import weakref

HERE = Path(__file__).resolve().parent
C5 = Path(os.environ.get("SERVER11_C5_CANDIDATE_ROOT", str(HERE.parent / "p4_single_file_candidate_v5_cpu"))).resolve()
REACTOR = C5 / "source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
COLLECTOR = C5 / "native_full_step_collector.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = load(HERE / "notification_runtime_adapter.py", "_notification_preparation_under_test")
W = load(C5 / "test_single_file_wait.py", "_notification_c5_wait_support")
sys.modules[W.collector.FullStepCapture.__module__] = W.collector


class Scalar:
    def __init__(self, old): self.__dict__.update(vars(old))
    def _fail(self, reason): self.enabled = False
    def detach(self): self.enabled = False


class Observer:
    def __init__(self, old):
        self.__dict__.update(vars(old)); self.__dict__.pop("detach", None)
        self.scalar = Scalar(old.scalar)
    def invalidate(self, reason): self.enabled = False
    def detach(self): self.enabled = False; self.scalar.detach()


def fixture(mode="on", *, max_wait_ns=100000000):
    owner = W.WaitOwner(clock=time.monotonic_ns, max_wait=max_wait_ns, sample_age=100000000,
                        off=mode == "off", shadow=mode == "shadow")
    owner._incoming = queue.Queue()
    capture, start, end = W.make_capture(owner, attach=False)
    capture.observer = Observer(capture.observer)
    # Explicit synthetic metadata binding. Real launcher first calls the strict
    # original bridge.attach_single_file_capture with native identity/receipt.
    owner.bridge._single_file_capture = weakref.ref(capture)
    return owner, capture, start, end


def attach(owner, capture, mode="on", **overrides):
    arguments = dict(run_id="r", request_id="r-p0-B", reactor_source=REACTOR,
                     collector_source=COLLECTOR, wake_type=W.Wake)
    arguments.update(overrides)
    audit = A.prepare_notification_capture(mode, owner, owner._prefix_p4_bridge, capture, **arguments)
    capture.after_prepare()  # SYNTHETIC next prepare callback, as during a request.
    return audit


def finish(owner, capture, audit):
    capture.detach()
    audit.close()
    return audit.export()


class NotificationPreparationTests(unittest.TestCase):
    def test_same_shared_launcher_helper_and_native_namespace(self):
        launcher = load(HERE / "run_p4_single_file_experiment.py", "_preparation_launcher")
        self.assertEqual(launcher.notification_adapter().C5_REACTOR_SHA256, A.C5_REACTOR_SHA256)
        owner, capture, *_ = fixture()
        audit = launcher.notification_adapter().prepare_notification_capture("on", owner, owner.bridge, capture,
            run_id="r", request_id="r-p0-B", reactor_source=REACTOR, collector_source=COLLECTOR, wake_type=W.Wake)
        self.assertTrue(audit.installed)
        self.assertEqual(len(capture._wait_observer_bindings), 4)
        self.assertIs(owner._prefix_single_file_wait_capture(), capture)
        finish(owner, capture, audit)

    def test_legacy_request_as_capture_run_rejected_before_install(self):
        owner, capture, *_ = fixture(); capture.run_id = "r-p0-B"
        with self.assertRaisesRegex(ValueError, "native owner run_id"): attach(owner, capture)
        self.assertFalse(hasattr(owner, "_prefix_single_file_wait_capture"))

    def test_capture_frontend_native_ids_are_independent_and_checked(self):
        output = dict(request_id="r-p0-B", native_request_id="original-17")
        A.validate_capture_identity(output, dict(run_id="r"), run_id="r", request_id="r-p0-B")
        for wrong in (dict(run_id="r-p0-B"), dict(run_id="old-run")):
            with self.assertRaises(ValueError): A.validate_capture_identity(output, wrong, run_id="r", request_id="r-p0-B")

    def test_off_shadow_do_not_attach_or_wrap_existing_queue(self):
        for mode in ("off", "shadow"):
            owner, capture, *_ = fixture(mode); queue_obj = owner._incoming
            audit = attach(owner, capture, mode)
            self.assertIs(owner._incoming, queue_obj)
            self.assertNotIn("get", vars(queue_obj))
            self.assertFalse(owner._prefix_wait_single_file_retry())
            doc = finish(owner, capture, audit)
            result = A.validate_notification_evidence(doc, mode=mode, run_id="r", request_id="r-p0-B")
            self.assertFalse(result["notification_exercised"])

    def test_queue_real_matching_wake_before_get_and_cleanup(self):
        owner, capture, _, end = fixture(); audit = attach(owner, capture)
        self.assertFalse(owner._drain_ready_preload_fds())
        original = audit._before_get
        def before(*args):
            row = original(*args); end.record(); return row
        audit._before_get = before
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertIsNone(owner._prefix_single_file_wait_armed)
        owner._drain_ready_preload_fds()
        self.assertEqual(owner.submitted, [(1, 3)])
        doc = finish(owner, capture, audit)
        self.assertEqual(doc["rows"][0]["outcome"], "matching_wake")
        self.assertIsNotNone(doc["rows"][0]["end_record_before_ns"])
        self.assertFalse(doc["gpu_completion_proved"])
        self.assertTrue(A.validate_notification_evidence(doc, mode="on", run_id="r", request_id="r-p0-B")["notification_exercised"])

    def test_independent_producer_real_queue_get(self):
        owner, capture, _, end = fixture(); audit = attach(owner, capture)
        owner._drain_ready_preload_fds(); entered = threading.Event()
        original = audit._before_get
        def before(*args):
            row = original(*args); entered.set(); return row
        audit._before_get = before
        worker = threading.Thread(target=lambda:(entered.wait(1), end.record()))
        worker.start()
        self.assertTrue(owner._prefix_wait_single_file_retry()); worker.join(1)
        self.assertFalse(worker.is_alive())
        doc = finish(owner, capture, audit)
        A.validate_notification_evidence(doc, mode="on", run_id="r", request_id="r-p0-B")

    def test_original_deadline_real_queue_timeout(self):
        owner, capture, *_ = fixture(max_wait_ns=10000000); audit = attach(owner, capture)
        # Start the original ready-arrival clock after CPU source installation.
        owner.ready.open_start_ns = time.monotonic_ns()
        owner._drain_ready_preload_fds()
        self.assertTrue(owner._prefix_wait_single_file_retry())
        doc = finish(owner, capture, audit)
        self.assertEqual(doc["rows"][0]["outcome"], "original_deadline_timeout")
        A.validate_notification_evidence(doc, mode="on", run_id="r", request_id="r-p0-B")

    def test_deferral_without_wait_is_not_notification_evidence(self):
        owner, capture, *_ = fixture(); audit = attach(owner, capture)
        owner._drain_ready_preload_fds(); self.assertGreater(owner.bridge.single_file_blocked_attempts, 0)
        doc = finish(owner, capture, audit)
        with self.assertRaisesRegex(ValueError, "deferral alone"):
            A.validate_notification_evidence(doc, mode="on", run_id="r", request_id="r-p0-B")

    def test_old_wake_or_native_message_not_promoted(self):
        for value in (W.Wake(("r", "old", 4, 1)), "synthetic mandatory"):
            owner, capture, *_ = fixture(); audit = attach(owner, capture)
            owner._drain_ready_preload_fds(); owner._incoming.put(value)
            self.assertTrue(owner._prefix_wait_single_file_retry())
            doc = finish(owner, capture, audit)
            with self.assertRaisesRegex(ValueError, "deferral alone"):
                A.validate_notification_evidence(doc, mode="on", run_id="r", request_id="r-p0-B")

    def test_wrong_source_or_wake_class_is_rejected(self):
        for override in (dict(reactor_source=COLLECTOR), dict(collector_source=REACTOR), dict(wake_type=object)):
            owner, capture, *_ = fixture()
            with self.assertRaises(ValueError): attach(owner, capture, **override)

    def test_missing_original_invalidation_hook_is_rejected(self):
        owner, capture, *_ = fixture(); capture.observer.invalidate = lambda reason: None
        with self.assertRaisesRegex(ValueError, "four actual"): attach(owner, capture)

    def test_second_capture_same_owner_rejected(self):
        owner, capture, *_ = fixture(); audit = attach(owner, capture); finish(owner, capture, audit)
        other, *_ = W.make_capture(owner, attach=False)
        with self.assertRaisesRegex(ValueError, "fresh reactor"): attach(owner, other)

    def test_native_intake_exception_propagates_through_observation(self):
        owner, capture, *_ = fixture(); audit = attach(owner, capture); owner._drain_ready_preload_fds()
        owner._incoming.put("synthetic mandatory")
        def bad(_item): raise RuntimeError("original intake error")
        owner._intake = bad
        with self.assertRaisesRegex(RuntimeError, "original intake error"): owner._prefix_wait_single_file_retry()
        self.assertIsNone(capture._wait_registration)
        finish(owner, capture, audit)

    def test_observation_failure_never_skips_original_queue(self):
        owner, capture, *_ = fixture(); audit = attach(owner, capture); owner._drain_ready_preload_fds()
        def bad(*_args): raise ValueError("synthetic audit failure")
        audit._before_get = bad; owner._incoming.put("synthetic mandatory")
        self.assertTrue(owner._prefix_wait_single_file_retry()); self.assertEqual(owner.seen, ["synthetic mandatory"])
        doc = finish(owner, capture, audit)
        with self.assertRaisesRegex(ValueError, "hidden notification"): A.validate_notification_evidence(doc, mode="on", run_id="r", request_id="r-p0-B")

    def test_notification_fault_recorded_and_rejected(self):
        owner, capture, *_ = fixture(); audit = attach(owner, capture); owner.bridge.fail("synthetic fault")
        doc = finish(owner, capture, audit)
        self.assertEqual(doc["bridge_fault"], "synthetic fault")
        with self.assertRaisesRegex(ValueError, "hidden notification"): A.validate_notification_evidence(doc, mode="on", run_id="r", request_id="r-p0-B")

    def test_observer_does_not_retain_queue_or_owner(self):
        owner, capture, *_ = fixture(); audit = attach(owner, capture)
        queue_ref, owner_ref = weakref.ref(owner._incoming), weakref.ref(owner)
        capture.detach(); del owner; gc.collect()
        self.assertIsNone(owner_ref()); self.assertIsNone(queue_ref())

    def test_restore_does_not_overwrite_later_wrapper(self):
        owner, capture, *_ = fixture(); audit = attach(owner, capture)
        later = lambda **kw: "later"; owner._incoming.get = later
        doc = finish(owner, capture, audit); self.assertIs(owner._incoming.get, later)
        self.assertFalse(doc["queue_get_restored"])

    def test_bounded_observation_overflow_not_promoted(self):
        owner, capture, *_ = fixture(); audit = attach(owner, capture); owner._drain_ready_preload_fds()
        audit.rows = [{}] * A.MAX_WITNESSES
        owner._incoming.put("synthetic mandatory"); owner._prefix_wait_single_file_retry()
        doc = finish(owner, capture, audit); self.assertTrue(doc["overflow"])
        with self.assertRaises(ValueError): A.validate_notification_evidence(doc, mode="on", run_id="r", request_id="r-p0-B")

    def test_entrypoints_block_before_config_io_or_gpu_import(self):
        for name in ("run_p4_single_file_experiment.py", "control_p4_single_file.py", "verify_p4_single_file.py"):
            proc = subprocess.run([sys.executable, "-B", "-I", "-S", str(HERE / name)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["status"], A.BLOCKED)
            self.assertEqual(proc.stderr, "")

    def test_direct_gpu_functions_block_and_no_legacy_receipt_lift(self):
        R = load(HERE / "run_p4_single_file_experiment.py", "_prep_direct_runner")
        C = load(HERE / "control_p4_single_file.py", "_prep_direct_controller")
        V = load(HERE / "verify_p4_single_file.py", "_prep_direct_verifier")
        callbacks = [lambda:R.execute_window(None,None,None,None), lambda:R.load_configuration(None),
            lambda:R.verify_guard(None), C.load_receipt, lambda:C.freeze_common("old-v6"),
            lambda:C.prepare("on"), lambda:C.launch("on"), lambda:C.after("on"),
            lambda:V.verify_runtime(None,config_ref=None,result_ref=None,guard_ref=None,before_ref=None,after_ref=None)]
        for callback in callbacks:
            with self.assertRaisesRegex(RuntimeError, A.BLOCKED): callback()

    def test_launcher_source_calls_adapter_and_preserves_request_id(self):
        source = (HERE / "run_p4_single_file_experiment.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        installs = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "install"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "collector"]
        self.assertEqual(len(installs), 1)
        self.assertEqual(next(kw.value.id for kw in installs[0].keywords if kw.arg == "run_id"), "LABEL")
        self.assertIn("request_id=rid", source)
        self.assertIn("server11-p4-notification-candidate-v5-cpu-20261003", source)
        self.assertEqual(sum(isinstance(node.func, ast.Attribute) and node.func.attr == "prepare_notification_capture" for node in calls), 1)

    def test_freezer_ignores_parent_gpu_outputs_and_locks_actual_candidate_tree(self):
        freezer = load(HERE / "freeze_cpu_preparation.py", "_cpu_freezer_test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); target = root / "new"; parent = root / "parent"; candidate = root / "candidate"
            for path in (target, parent, candidate / "source/nested"): path.mkdir(parents=True)
            (target / "freeze_cpu_preparation.py").write_bytes((HERE / "freeze_cpu_preparation.py").read_bytes())
            for name in freezer.BASELINE_FILES:
                (parent / name).write_text("synthetic ancestor\n", encoding="utf-8")
                (target / name).write_text("synthetic ancestor\n", encoding="utf-8")
            (parent / "GPU_GUARD_on_STDOUT.log").write_text("DO NOT COPY/READ AS BASELINE", encoding="utf-8")
            (parent / "CONFIG_on.json").write_text("{}", encoding="utf-8")
            (candidate / "source/nested/inherited.py").write_text("# synthetic inherited source\n", encoding="utf-8")
            (candidate / "collector.py").write_text("# synthetic collector source\n", encoding="utf-8")
            command = [sys.executable, "-B", "-I", "-S", str(target / "freeze_cpu_preparation.py"),
                       "--candidate-root", str(candidate), "--runtime-v4-root", str(parent)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            baseline = json.loads((target / "BASELINE_INHERITANCE.json").read_text())
            self.assertEqual(len(baseline["files"]), 8)
            lock = json.loads((target / "PREPARATION_SOURCE_LOCK.json").read_text())
            self.assertEqual({r["path"] for r in lock["files"] if r["scope"] == "candidate"},
                             {"collector.py", "source/nested/inherited.py"})
            self.assertFalse(lock["gpu_launch_allowed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
