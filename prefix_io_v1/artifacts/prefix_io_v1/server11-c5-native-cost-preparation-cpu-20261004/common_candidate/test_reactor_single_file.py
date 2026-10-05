"""CPU execution of source-extracted native decision/queue paths; no GPU claim."""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, replace
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock


HERE = Path(__file__).resolve().parent
REL = Path("third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py")
CONTROL = Path("third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control")
CANDIDATE = HERE / "source" / REL
ROOT = Path(os.environ["SERVER11_AUTHOR_SOURCE_ROOT"]).resolve() if os.environ.get(
    "SERVER11_AUTHOR_SOURCE_ROOT") else next(
        p for p in HERE.parents if (p / "artifacts/prefix_io_v1_server09_candidates").is_dir())
ORIGINAL = ROOT / REL if os.environ.get("SERVER11_AUTHOR_SOURCE_ROOT") else (
    ROOT / "artifacts/prefix_io_v1_server09_candidates/g2_source_readonly" / REL)
BASE_CONTROL = ROOT / CONTROL if os.environ.get("SERVER11_AUTHOR_SOURCE_ROOT") else (
    ROOT / "artifacts/prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source" / CONTROL)


def module_sources():
    # Import only unchanged pure-CPU value definitions and the candidate policy.
    # The reactor itself is never imported: its CUDA dependencies cannot run.
    package = ModuleType("prefix_io_control")
    package.__path__ = [str(HERE / "source" / CONTROL), str(BASE_CONTROL)]
    sys.modules["prefix_io_control"] = package
    for suffix in ("dispatch_budget", "dispatch_shadow", "p4_types", "p4_policy"):
        importlib.import_module("prefix_io_control." + suffix)


module_sources()
from prefix_io_control.dispatch_budget import STAGES
from prefix_io_control.p4_types import P4Config, SystemSnapshot, WorkDescriptor
from prefix_io_control.p4_policy import P4Policy


def extracted(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IoReactor")
    names = {"_prefix_stage_decide", "_prefix_ready_read_decision", "_drain_ready_preload_fds"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in (
        "_ReadyFd", "_PreloadInfo", "_StageDispatch", "_P4Deferred")]
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"dataclass": dataclass, "time": SimpleNamespace(monotonic_ns=lambda: 1000)}
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(
        name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace


NATIVE = extracted(CANDIDATE)
Ready = NATIVE["_ReadyFd"]
Info = NATIVE["_PreloadInfo"]
HASH = b"x" * 32
IO_SIZE = 917504


class Bridge:
    run_id = "cpu-source-fixture"

    def __init__(self, single_file=True, action="defer"):
        self.policy = SimpleNamespace(single_file=object() if single_file else None)
        self.action, self.seen, self.failures, self.deferrals = action, [], [], []

    def preview_issue(self, work, snapshot, *, now_ns):
        self.seen.append((work, snapshot, now_ns))
        return SimpleNamespace(action=self.action)

    def fail(self, reason):
        self.failures.append(reason)

    def record_single_file_deferral(self, work_id, *, at_ns):
        self.deferrals.append((work_id, at_ns))
        return True


class Owner:
    _prefix_stage_decide = NATIVE["_prefix_stage_decide"]
    _prefix_ready_read_decision = NATIVE["_prefix_ready_read_decision"]
    _drain_ready_preload_fds = NATIVE["_drain_ready_preload_fds"]

    def __init__(self, bridge):
        self._prefix_p4_bridge, self._prefix_dispatch_controller = bridge, None
        self._prefix_stage_accounting = SimpleNamespace(valid=True, _owner=lambda: None,
            stats={s: {"inflight_ops": 0, "inflight_bytes": 0} for s in STAGES})
        self._stop = False
        self._preload_waiters = {}
        self.file_store = SimpleNamespace(io_size=IO_SIZE)
        self.layout = SimpleNamespace(bytes_per_kernel_block=(IO_SIZE,))
        self._ready_sequence = 30
        self._prefix_clean_reclaimable_bytes = lambda: 8 * IO_SIZE
        self.accepted_parent_count = lambda: 0
        self.is_mandatory = lambda future: future == "mandatory"
        self._has_preload_waiter = lambda value: False
        self._prefix_p4_collect = lambda: SimpleNamespace(snapshot=SystemSnapshot(
            "cpu-source-fixture", 1, 1000, frozenset(("native_ready_work",))))


def ready(**changes):
    item = Ready(fd=7, job=None, file_index=-1, preload_hash=HASH,
        open_start_ns=250, preload_info=Info(HASH, "preload1", "request1", "profile1", 0, 1),
        sequence=31)
    for key, value in changes.items():
        setattr(item, key, value)
    return item


class NativeSingleFileTests(unittest.TestCase):
    def setUp(self):
        NATIVE["time"].monotonic_ns = lambda: 1000

    def test_off_has_no_clock_snapshot_or_metadata_mutation(self):
        owner, item = Owner(None), ready(sequence=0)
        NATIVE["time"].monotonic_ns = Mock(side_effect=AssertionError("off clock"))
        owner._prefix_p4_collect = Mock(side_effect=AssertionError("off snapshot"))
        self.assertIsNone(owner._prefix_ready_read_decision(item))
        self.assertEqual(item.sequence, 0)
        self.assertIsNone(owner._prefix_stage_decide("ssd_read", IO_SIZE, ("x", 1)))

    def test_exact_native_ready_age_identity_and_capacity_are_preserved(self):
        bridge, item = Bridge(), ready()
        owner = Owner(bridge)
        for clock in (1000, 1600):
            NATIVE["time"].monotonic_ns = lambda: clock
            dispatch = owner._prefix_ready_read_decision(item)
            self.assertEqual(dispatch.action, "defer")
            work, snapshot, observed = bridge.seen[-1]
            self.assertIs(type(work), WorkDescriptor)
            self.assertEqual(work.created_ns, 250)
            self.assertEqual(work.child_id, "preload:31:" + HASH.hex())
            self.assertEqual(work.parent_id, 0)
            self.assertEqual(work.nbytes, IO_SIZE)
            self.assertEqual(work.minimum_unit_bytes, IO_SIZE)
            self.assertEqual(snapshot.native_state.free_staging_bytes, 9 * IO_SIZE)
            self.assertEqual(observed, clock)
        self.assertEqual(bridge.failures, [])

    def test_unqualified_or_malformed_unbound_preloads_keep_original_path(self):
        cases = [ready(preload_info=None), ready(file_index=0), ready(open_start_ns=0),
            ready(open_start_ns=-1), ready(open_start_ns=1001), ready(preload_hash=None),
            ready(preload_hash=b"x" * 31),
            ready(preload_info=Info(HASH, "p", "r", "t", 0, 8)),
            ready(preload_info=Info(HASH, "p", "r", "t", 1, 1)),
            ready(preload_info=Info(HASH, "", "r", "t", 0, 1)),
            ready(preload_info=Info(HASH, "p", "", "t", 0, 1)),
            ready(preload_info=Info(b"y" * 32, "p", "r", "t", 0, 1))]
        for item in cases:
            with self.subTest(item=item):
                bridge = Bridge()
                self.assertIsNone(Owner(bridge)._prefix_ready_read_decision(item))
                self.assertEqual(bridge.seen, [])
                self.assertEqual(bridge.failures, [])
        bridge = Bridge(single_file=False)
        self.assertIsNone(Owner(bridge)._prefix_ready_read_decision(ready()))
        self.assertEqual(bridge.seen, [])

    def test_live_step_expiry_at_native_decision_returns_original_path(self):
        bridge = Bridge()
        bridge.record_single_file_deferral = Mock(return_value=False)
        self.assertIsNone(Owner(bridge)._prefix_ready_read_decision(ready()))
        self.assertEqual(len(bridge.seen), 1)
        bridge.record_single_file_deferral.assert_called_once()
        self.assertEqual(bridge.failures, [])

    def test_source_direct_nonpositive_sequence_does_not_create_binding(self):
        for sequence in (0, -1):
            bridge, item = Bridge(), ready(sequence=sequence)
            self.assertIsNone(Owner(bridge)._prefix_stage_decide("ssd_read", IO_SIZE,
                ("ssd_read", sequence), ready=item, reserved_bytes=IO_SIZE))
            self.assertEqual(bridge.seen, [])

    def test_original_sequence_assignment_stable_across_deferred_retries(self):
        bridge, item = Bridge(), ready(sequence=0)
        owner = Owner(bridge)
        owner._prefix_ready_read_decision(item)
        owner._prefix_ready_read_decision(item)
        self.assertEqual(item.sequence, 31)
        self.assertEqual([x[0].child_id for x in bridge.seen], ["preload:31:" + HASH.hex()] * 2)

    def test_original_progress_bypasses_additional_deferral(self):
        for progress in ("mandatory", "continuation", "shutdown", "mandatory_support"):
            with self.subTest(progress=progress):
                bridge, item = Bridge(), ready()
                owner = Owner(bridge)
                job = None
                kwargs = {}
                if progress in ("mandatory", "continuation"):
                    job = SimpleNamespace(failed=None, future="mandatory" if progress ==
                        "mandatory" else "ordinary", accepted_parent_sequence=2,
                        profile=SimpleNamespace(start_ns=100))
                    kwargs["continuation"] = progress == "continuation"
                elif progress == "shutdown":
                    owner._stop = True
                else:
                    kwargs["support"] = True
                self.assertIsNone(owner._prefix_stage_decide("ssd_read", IO_SIZE,
                    ("ssd_read", item.sequence), job, ready=item, **kwargs))
                self.assertEqual(bridge.seen[-1][0].progress, progress)
                self.assertEqual(bridge.failures, [])

    def test_original_policy_age_override_precedes_single_file_condition(self):
        bridge, owner = Bridge(), None
        owner = Owner(bridge)
        owner._prefix_ready_read_decision(ready())
        work, snapshot, now = bridge.seen[0]
        policy = object.__new__(P4Policy)
        policy.run_id = bridge.run_id
        policy.config = P4Config(mode="interference", max_wait_ns=500,
            sample_max_age_ns=10000, internal_step_budget_ns=1)
        policy.single_file = object()  # Access after the age guard would fail.
        result = policy.issue_preview(work, snapshot, now_ns=now, expected_epoch=1)
        self.assertEqual((result.action, result.reason), ("issue", "native_progress_override"))

    def test_deferred_original_queue_retains_fd_and_releases_reserved_slot(self):
        for has_waiter in (False, True):
            with self.subTest(has_waiter=has_waiter):
                owner, item = Owner(Bridge()), ready()
                owner._ready_fds_preload = deque((item,))
                owner._has_preload_waiter = lambda value: has_waiter
                owner._has_real_load_pressure = lambda: False
                owner._can_issue_speculative_preload = lambda: True
                owner._data_inflight, owner.iodepth = 0, 4
                owner._reserve_foreground_slot = lambda: SimpleNamespace(index=3)
                owner._reserve_preload_slot = lambda: SimpleNamespace(index=3)
                owner.staging_pool = SimpleNamespace(release=Mock())
                owner._submit_read_from_ready = Mock(side_effect=AssertionError("unexpected read"))
                self.assertFalse(owner._drain_ready_preload_fds())
                self.assertIs(owner._ready_fds_preload[0], item)
                self.assertEqual(len(owner._ready_fds_preload), 1)
                self.assertEqual(item.fd, 7)
                owner.staging_pool.release.assert_called_once_with(3)
                owner._submit_read_from_ready.assert_not_called()

    def test_original_native_queue_implementation_is_unchanged(self):
        def body(path):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            reactor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IoReactor")
            return ast.dump(next(n for n in reactor.body if isinstance(n, ast.FunctionDef)
                and n.name == "_drain_ready_preload_fds"), include_attributes=False)
        self.assertEqual(body(CANDIDATE), body(ORIGINAL))


if __name__ == "__main__":
    unittest.main(verbosity=2)
