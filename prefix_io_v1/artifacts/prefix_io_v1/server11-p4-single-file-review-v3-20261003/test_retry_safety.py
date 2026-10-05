"""Independent CPU tests of v3 source; never native/GPU qualification."""
import ast
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
import importlib
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock

HERE = Path(__file__).resolve().parent
CANDIDATE = Path(os.environ["SERVER11_CANDIDATE_ROOT"]).resolve() if os.environ.get("SERVER11_CANDIDATE_ROOT") else HERE.parent / "p4_single_file_candidate_v3"
WORKSPACE = Path(os.environ["SERVER11_AUTHOR_SOURCE_ROOT"]).resolve() if os.environ.get("SERVER11_AUTHOR_SOURCE_ROOT") else next(parent for parent in HERE.parents if (parent / "artifacts/prefix_io_v1_server09_candidates").is_dir())
CONTROL = Path("third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control")
BASE = WORKSPACE / CONTROL if os.environ.get("SERVER11_AUTHOR_SOURCE_ROOT") else WORKSPACE / "artifacts/prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source" / CONTROL
REACTOR = CANDIDATE / "source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
package = ModuleType("prefix_io_control")
package.__path__ = [str(CANDIDATE / "source" / CONTROL), str(BASE)]
sys.modules["prefix_io_control"] = package
T = importlib.import_module("prefix_io_control.p4_types")
P = importlib.import_module("prefix_io_control.p4_policy")
B = importlib.import_module("prefix_io_control.p4_bridge")
R = importlib.import_module("prefix_io_control.p4_single_file_receipt")
D = importlib.import_module("prefix_io_control.dispatch_budget")

tree = ast.parse(REACTOR.read_bytes())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IoReactor")
METHODS = {"_prefix_stage_decide", "_prefix_ready_read_decision", "_prefix_single_file_retry_key"}
nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in
         ("_ReadyFd", "_PreloadInfo", "_StageDispatch", "_P4Deferred")]
nodes += [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
CLOCK = [1000]
N = dict(dataclass=dataclass, threading=threading, time=SimpleNamespace(monotonic_ns=lambda: CLOCK[0]))
unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
                  type_ignores=[])
exec(compile(ast.fix_missing_locations(unit), str(REACTOR), "exec"), N)
Q = 917504


def cpu_receipt():
    # Deliberate test-only bypass; production issuance still requires native raw evidence.
    receipt = object.__new__(R.ExactSingleFileReceipt)
    object.__setattr__(receipt, "signature", ("m"*64, "GPU-CPU-TEST", "l"*64, "cpu-fixture", 1, 1, 0, 144, Q))
    object.__setattr__(receipt, "cost_upper_ns", 20)
    object.__setattr__(receipt, "step_budget_ns", 10)
    return receipt


class Owner:
    _prefix_stage_decide = N["_prefix_stage_decide"]
    _prefix_ready_read_decision = N["_prefix_ready_read_decision"]
    _prefix_single_file_retry_key = N["_prefix_single_file_retry_key"]

    def __init__(self):
        self.receipt = cpu_receipt()
        config = T.P4Config(mode="interference", sample_max_age_ns=10000, max_wait_ns=3000,
                            internal_step_budget_ns=10)
        self.bridge = B.NativeP4Bridge(P.P4Policy("independent-cpu-fixture", config, single_file=self.receipt))
        self.bridge.bind()
        self.bridge.single_file_shadow = False
        self.bridge._single_file_runtime = SimpleNamespace(signature_prefix=self.receipt.signature[:4])
        self.state = (16, 100, 200, 1, 1, 0, 144)
        self.bridge._single_file_state = lambda: self.state
        self._prefix_p4_bridge = self.bridge
        self._prefix_dispatch_controller = None
        self._prefix_stage_accounting = SimpleNamespace(valid=True, journal_valid=True, event_sequence=0,
            _owner=lambda: None, stats={stage: dict(inflight_ops=0, inflight_bytes=0) for stage in D.STAGES})
        self._worker = SimpleNamespace(ident=threading.get_ident())
        self._submit_lock = threading.Lock()
        self._prefix_p4_load_frame = None
        self._prefix_p4_load_sequence = 0
        self._stop = False
        self._active, self._inflight, self._pending_copies, self._copy_ready = [], [], [], []
        self._ready_fds_load, self._preload_pending = [], []
        self._open_inflight = self._data_inflight = 0
        self._preload_waiters = {}
        self._has_preload_waiter = lambda key: bool(self._preload_waiters.get(key))
        self.is_mandatory = lambda future: future == "mandatory"
        self.file_store = SimpleNamespace(io_size=Q)
        self.layout = SimpleNamespace(bytes_per_kernel_block=(Q,))
        self.capacity = (8, Q, 7, 0, 0, 0, 0, 0, 0, 0, 0, 7)
        self._prefix_capacity_components = lambda: self.capacity
        self._prefix_clean_reclaimable_bytes = lambda: self.capacity[-1]*Q
        self.accepted_parent_count = lambda: 0
        self.ready = N["_ReadyFd"](fd=7, job=None, file_index=-1, preload_hash=b"x"*32,
            open_start_ns=250, preload_info=N["_PreloadInfo"](b"x"*32, "preload", "request", "profile", 0, 1), sequence=1)
        self._ready_fds_preload = deque([self.ready])
        self.collect_calls = 0

    def _prefix_p4_collect(self):
        self.collect_calls += 1
        epoch = self.bridge.epoch(("cpu-value-publication",), CLOCK[0])
        return SimpleNamespace(snapshot=T.SystemSnapshot(self.bridge.run_id, epoch, CLOCK[0],
                                                         frozenset(("native_ready_work",))))

    def decision(self):
        return self._prefix_ready_read_decision(self.ready)


class IndependentSafetyTests(unittest.TestCase):
    def setUp(self):
        CLOCK[0] = 1000
        self.o = Owner()

    def populated(self):
        result = self.o.decision()
        self.assertEqual(result.action, "defer")
        self.assertIsNone(self.o.bridge.fault)
        self.assertIsNotNone(self.o._prefix_single_file_retry)
        return self.o._prefix_single_file_retry

    def test_reuse_keeps_both_observation_timestamps_and_real_created_age(self):
        cached = self.populated()
        CLOCK[0] = 1200
        self.assertEqual(self.o.decision().action, "defer")
        self.assertEqual(self.o.collect_calls, 1)
        self.assertEqual(self.o.bridge.single_file_observation_reuses, 1)
        self.assertIs(self.o._prefix_single_file_retry[1], cached[1])
        self.assertIs(self.o._prefix_single_file_retry[2], cached[2])
        self.assertEqual(cached[1].monotonic_ns, 1000)
        self.assertEqual(cached[1].native_state.captured_ns, 1000)
        self.assertEqual(cached[2].created_ns, 250)

    def test_mandatory_waiter_immediately_restores_original_progress(self):
        self.populated()
        self.o._preload_waiters[b"x"*32] = [(SimpleNamespace(future="mandatory"), 0)]
        self.assertIsNone(self.o.decision())
        self.assertEqual(self.o.bridge.last_preview.reason, "native_progress_override")
        self.assertIsNone(self.o._prefix_single_file_retry)

    def test_context_end_or_changed_context_never_reuses_prior_observation(self):
        for value in (None, (17, 100, 200, 1, 1, 0, 145)):
            with self.subTest(value=value):
                self.o = Owner()
                self.populated()
                self.o.state = value
                self.assertIsNone(self.o.decision())
                self.assertIsNone(self.o._prefix_single_file_retry)
                self.assertEqual(self.o.bridge.single_file_observation_reuses, 0)

    def test_replaced_step_same_shape_still_invalidates_old_key(self):
        self.populated()
        self.o.state = (17, 150, 300, 1, 1, 0, 144)
        self.assertEqual(self.o.decision().action, "defer")
        self.assertEqual(self.o.collect_calls, 2)
        self.assertEqual(self.o.bridge.single_file_observation_reuses, 0)

    def test_maxwait_is_not_extended_by_reuse(self):
        self.populated()
        CLOCK[0] = 3250
        self.assertIsNone(self.o.decision())
        self.assertEqual(self.o.bridge.last_preview.reason, "native_progress_override")
        self.assertIsNone(self.o._prefix_single_file_retry)

    def test_stale_cache_recollects_instead_of_refreshing_it(self):
        cached = self.populated()
        # Same condition becomes a new valid observation, never mutates old sample.
        CLOCK[0] = 1500
        object.__setattr__(self.o.bridge.policy.config, "sample_max_age_ns", 300)
        self.assertIsNone(self.o.decision())  # The real start-query age is also stale.
        self.assertEqual(cached[1].monotonic_ns, 1000)
        self.assertEqual(cached[1].native_state.captured_ns, 1000)
        self.assertIsNone(self.o._prefix_single_file_retry)

    def test_capacity_or_journal_change_recollects(self):
        for mutation in (lambda o: setattr(o._prefix_stage_accounting, "event_sequence", 2),
                         lambda o: setattr(o, "capacity", (8, Q, 6, 0, 0, 0, 0, 0, 0, 0, 0, 6))):
            with self.subTest(mutation=mutation):
                self.o = Owner()
                self.populated()
                mutation(self.o)
                self.o.decision()
                self.assertEqual(self.o.collect_calls, 2)
                self.assertEqual(self.o.bridge.single_file_observation_reuses, 0)

    def test_stop_and_downstream_progress_bypass_cache(self):
        self.populated()
        self.o._stop = True
        self.assertIsNone(self.o.decision())
        self.assertEqual(self.o.bridge.last_preview.reason, "native_progress_override")
        self.o = Owner()
        self.populated()
        self.assertIsNone(self.o._prefix_stage_decide("h2d", Q, ("copy", 1), continuation=True))
        self.assertEqual(self.o.bridge.last_preview.reason, "native_progress_override")
        self.assertIsNone(self.o._prefix_single_file_retry)

    def test_other_stage_or_size_cannot_borrow_cached_ssd_descriptor(self):
        for stage, nbytes in (("h2d", Q), ("ssd_read", 2*Q)):
            with self.subTest(stage=stage, nbytes=nbytes):
                self.o = Owner()
                self.populated()
                self.assertIsNone(self.o._prefix_stage_decide(stage, nbytes, (stage, 1),
                    ready=self.o.ready, reserved_bytes=Q))
                self.assertEqual(self.o.bridge.single_file_observation_reuses, 0)
                self.assertIsNone(self.o._prefix_single_file_retry)

    def test_nonempty_native_state_and_scheduler_publication_exclude_reuse(self):
        for name, value in (("_pending_copies", [object()]), ("_active", [object()]),
                ("_ready_fds_load", [object()]), ("_prefix_p4_load_frame", object()),
                ("_prefix_p4_load_sequence", 1)):
            with self.subTest(name=name):
                self.o = Owner()
                self.populated()
                setattr(self.o, name, value)
                self.o.decision()
                self.assertEqual(self.o.collect_calls, 2)
                self.assertEqual(self.o.bridge.single_file_observation_reuses, 0)
                self.assertIsNone(self.o._prefix_single_file_retry)

    def test_cache_only_holds_immutable_values_not_native_owners(self):
        cached = self.populated()
        def check(value):
            if value is None or type(value) in (int, bool, str, bytes):
                return
            if type(value) in (tuple, frozenset):
                for child in value:
                    check(child)
                return
            if is_dataclass(value) and value.__dataclass_params__.frozen:
                for field in fields(value):
                    check(getattr(value, field.name))
                return
            self.fail("retained mutable/native owner: " + type(value).__name__)
        check(cached)

    def test_off_stays_before_clock_or_cache_access(self):
        self.populated()
        self.o._prefix_p4_bridge = None
        previous = N["time"].monotonic_ns
        N["time"].monotonic_ns = Mock(side_effect=AssertionError("off queried clock"))
        try:
            self.assertIsNone(self.o.decision())
        finally:
            N["time"].monotonic_ns = previous


if __name__ == "__main__":
    unittest.main()
