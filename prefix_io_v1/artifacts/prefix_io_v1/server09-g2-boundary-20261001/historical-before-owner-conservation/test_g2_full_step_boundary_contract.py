"""Fake event and actual locked NativeWindowJournal CPU contracts only."""
import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types
import unittest
import weakref

HERE = Path(__file__).resolve().parent
CPU_ROOT = HERE.parents[1] / "prefix_io_v1_server08_p4_cpu_extended_20261001" / "verified-v3" / "source" / "third_party" / "work" / "prefix-io-p4-02-cpu" / "src" / "prefix_io_control"
SOURCE_ROOT = HERE.parent / "g2_source_readonly"
RUNTIME_SOURCE = HERE.parent / "runtime_connector" / "p4_runtime_scalar_connector.py"
COLLECTOR_SOURCE = HERE.parent / "collector" / "p4_full_step_frame_adapter.py"
remaining_arguments = []
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-source-root", type=Path, default=CPU_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--runtime-source", type=Path, default=RUNTIME_SOURCE)
    parser.add_argument("--collector-source", type=Path, default=COLLECTOR_SOURCE)
    arguments, remaining_arguments = parser.parse_known_args()
    CPU_ROOT, SOURCE_ROOT = arguments.cpu_source_root.resolve(), arguments.source_root.resolve()
    RUNTIME_SOURCE, COLLECTOR_SOURCE = arguments.runtime_source.resolve(), arguments.collector_source.resolve()


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
B = load_file("g2_boundary_candidate", HERE / "g2_full_step_boundary_contract.py")
E = load_file("g2_source_preflight_candidate", HERE / "run_g2_normal_model_candidate.py")
runtime_bytes = RUNTIME_SOURCE.read_bytes()
if len(runtime_bytes) != 14795 or hashlib.sha256(runtime_bytes).hexdigest() != "347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb":
    raise ValueError("frozen runtime connector bytes/SHA differs")
C = load_file("g2_frozen_runtime_connector", RUNTIME_SOURCE)
F = C.load_frozen_adapter(COLLECTOR_SOURCE)
J = None
NATIVE_SHA = "2b0ac1e8e68f82adb42ade03b74b49843eabe55361a00f0ac1f7a738ac423e18"


def load_locked_journal(root):
    refs = (
        ("stage_accounting.py", 3580, "11a736586a9278e2c373d4fc1bfe0faf9cd166b81e9d9c3f658500a51ab954aa"),
        ("dispatch_budget.py", 11021, "412c43761cb2f59636240e15c884b25d2a1d0ee708684d496259baf2f34d667c"),
        ("p4_native_window_journal.py", 9647, "3a9ded39846cccf88ee9aceed2e16b86d8a7bbe953adfea3ba1a302b2c5aa8c3"))
    for name, nbytes, sha in refs:
        C.SourceRef(str((root / name).resolve()), nbytes, sha).checked_path()
    package = types.ModuleType("_g2_locked_control")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    for name, _, _ in refs:
        module = load_file(package.__name__ + "." + name[:-3], root / name)
    return module


class Poison:
    def __getattribute__(self, name): raise AssertionError("off event read")


class FakeEvent:
    def __init__(self, at, ready=True):
        self.at, self.ready, self.records = at, ready, 0
    def record(self): self.records += 1
    def query(self): return self.ready
    def elapsed_time(self, other): return (other.at - self.at) / 1_000_000
    def synchronize(self): raise AssertionError("contract must never synchronize")


class FakeFactory:
    def __init__(self, *events): self.events, self.calls = list(events), 0
    def __call__(self, *, enable_timing):
        if enable_timing is not True: raise ValueError("timing required")
        self.calls += 1
        return self.events.pop(0)


class BoundaryTests(unittest.TestCase):
    def frame(self, ordinal=0):
        prepared = F.PreparedFrame(ordinal, (F.PreparedRow("r", 128, 128, 1),),
            1, 1, 0, 128, "decode", (129,))
        return F.ClosedFrame(ordinal, 10, 20, prepared, (("r", (7,)),))
    def probe(self, **kwargs):
        return B.FullStepEventContract("cpu", NATIVE_SHA, frame_type=F.ClosedFrame,
            enabled=True, **kwargs)
    def window(self, **kwargs):
        return B.CompleteStepEventWindow("cpu", 0, 100, 100, 200, True, NATIVE_SHA, **kwargs)
    def journal(self, times):
        iterator = iter(times)
        return J.NativeWindowJournal("cpu", NATIVE_SHA, enabled=True, clock=lambda: next(iterator))

    def test_off_before_event_or_frame_reads(self):
        probe = B.FullStepEventContract("cpu", NATIVE_SHA, frame_type=Poison(), enabled=False)
        probe.before_execute(Poison(), Poison())
        probe.after_sample(Poison())
        self.assertEqual(probe.resolve_ready(Poison()), ())

    def test_off_does_not_retain_supplied_frame_owner(self):
        class Owner: pass
        owner = Owner(); ref = weakref.ref(owner)
        probe = B.FullStepEventContract("cpu", NATIVE_SHA, frame_type=owner, enabled=False)
        del owner; gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(probe.frame_type)

    def test_execute_boundary_waits_for_real_sample_closed_frame(self):
        start, end = FakeEvent(100), FakeEvent(200)
        probe = self.probe()
        probe.before_execute(0, FakeFactory(start, end))
        self.assertEqual(start.records, 1)
        self.assertEqual(end.records, 0)
        self.assertEqual(probe.pending, [])
        self.assertIsNotNone(probe.active)
        probe.after_sample(self.frame())
        self.assertEqual(end.records, 1)
        rows = probe.resolve_ready(B.CPUClockReference(FakeEvent(0), 1, "cpu", True))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].event_elapsed_ns, 100)
        self.assertEqual(rows[0].mapped_start_ns, 101)
        self.assertEqual(rows[0].mapped_end_ns, 201)
        self.assertIsNone(rows[0].gpu_elapsed_ns)
        self.assertFalse(rows[0].gpu_verified)
        self.assertEqual(rows[0].scope, "full_decode_step")
        self.assertEqual(probe.runtime_hook_status, "not_installed")

    def test_query_unready_never_synchronizes_or_invents_elapsed(self):
        start, end = FakeEvent(100), FakeEvent(200, ready=False)
        probe = self.probe()
        probe.before_execute(0, FakeFactory(start, end)); probe.after_sample(self.frame())
        reference = B.CPUClockReference(FakeEvent(0), 1, "cpu", True)
        self.assertEqual(probe.resolve_ready(reference), ())
        self.assertEqual(len(probe.pending), 1)
        end.ready = True
        self.assertEqual(len(probe.resolve_ready(reference)), 1)
        self.assertEqual(probe.pending, [])

    def test_unknown_cross_clock_or_fallback_never_fills_mapped_IO(self):
        for reference in (B.CPUClockReference(FakeEvent(0), 1, "cpu"),
            B.CPUClockReference(FakeEvent(0), 1, "cpu", True, True)):
            probe = self.probe(); probe.before_execute(0, FakeFactory(FakeEvent(100), FakeEvent(200)))
            probe.after_sample(self.frame())
            row = probe.resolve_ready(reference)[0]
            self.assertIsNone(row.mapped_start_ns)
            self.assertIsNone(row.mapped_end_ns)
            attribution = B.attribute_published_full_step(row, Poison(), run_id="cpu", native_source_sha256=NATIVE_SHA)
            self.assertEqual(attribution.status, "UNKNOWN")
            self.assertIsNone(attribution.new_io)

    def test_event_errors_missing_sample_ordinals_and_capacity_fail_closed(self):
        probe = self.probe(); probe.before_execute(0, lambda **k: (_ for _ in ()).throw(RuntimeError()))
        self.assertFalse(probe.valid)
        probe = self.probe(); probe.before_execute(0, FakeFactory(FakeEvent(1), FakeEvent(2)))
        probe.before_execute(1, Poison())
        self.assertFalse(probe.valid)
        probe = self.probe(); probe.before_execute(0, FakeFactory(FakeEvent(1), FakeEvent(2)))
        probe.after_sample(self.frame(1))
        self.assertFalse(probe.valid)
        probe = self.probe(max_pending=1); probe.before_execute(0, FakeFactory(FakeEvent(1), FakeEvent(2)))
        probe.after_sample(self.frame()); probe.before_execute(1, Poison())
        self.assertFalse(probe.valid)
        self.assertEqual(probe.pending, [])

    def test_event_factory_not_retained_after_begin(self):
        factory = FakeFactory(FakeEvent(100), FakeEvent(200))
        ref = weakref.ref(factory)
        probe = self.probe(); probe.before_execute(0, factory)
        del factory; gc.collect()
        self.assertIsNone(ref())
        probe.invalidate("CPU cleanup")
        self.assertIsNone(probe.active)

    def test_cpu_event_origin_cannot_claim_GPU_elapsed_or_unknown_mapping(self):
        with self.assertRaises(ValueError): self.window(gpu_elapsed_ns=100)
        with self.assertRaises(ValueError): self.window(origin="native_gpu_diagnostic")
        with self.assertRaises(ValueError): replace(self.window(), mapping_known_in_fixture=False)
        with self.assertRaises(TypeError): self.window(production_qualified=True)
        with self.assertRaises(TypeError): self.window(gpu_verified=True)

    def test_actual_locked_journal_existing_and_all_four_new_stages(self):
        journal = self.journal([80, 90, 120, 121, 122, 123, 210, 220, 221, 222, 223, 230])
        journal.publish_owner_frame()
        journal.accepted("ssd_read", "existing", 512)
        for i, stage in enumerate(B.STAGES): journal.accepted(stage, ("new", i), 1024)
        journal.completed("ssd_read", "existing", 512)
        for i, stage in enumerate(B.STAGES): journal.completed(stage, ("new", i), 1024)
        journal.publish_owner_frame()
        published = journal.published()
        result = B.attribute_published_full_step(self.window(), published,
            run_id="cpu", native_source_sha256=NATIVE_SHA)
        self.assertEqual(result.status, "CPU_ATTRIBUTION_CONTRACT_ONLY", result.reason)
        self.assertEqual(result.existing_io[0], B.PhysicalAmount(1, 512))
        self.assertEqual(result.new_io, (B.PhysicalAmount(1, 1024),) * 4)
        self.assertEqual(result.completed_new_io, result.new_io)
        self.assertFalse(result.gpu_verified)
        self.assertFalse(result.production_qualified)
        ref = weakref.ref(journal); del journal; gc.collect()
        self.assertIsNone(ref())  # published owner scalars retain no reactor/journal.

    def test_short_IO_unfinished_IO_and_boundary_ties_are_unknown(self):
        for mode in ("short", "unfinished", "boundary"):
            journal = self.journal([80, 100 if mode == "boundary" else 120, 220, 230])
            journal.publish_owner_frame(); journal.accepted("h2d", "new", 1024)
            if mode != "unfinished": journal.completed("h2d", "new", 512 if mode == "short" else 1024)
            journal.publish_owner_frame()
            result = B.attribute_published_full_step(self.window(), journal.published(),
                run_id="cpu", native_source_sha256=NATIVE_SHA)
            self.assertEqual(result.status, "UNKNOWN")
            self.assertIsNone(result.new_io)

    def test_bad_scope_source_run_missing_owner_or_overflow_rejected(self):
        journal = self.journal([80, 230]); journal.publish_owner_frame(); journal.publish_owner_frame()
        published = journal.published()
        windows = (replace(self.window(), scope="model_forward"),
            replace(self.window(), run_id="other"), replace(self.window(), native_source_sha256="0" * 64))
        for window in windows:
            self.assertEqual(B.attribute_published_full_step(window, published,
                run_id="cpu", native_source_sha256=NATIVE_SHA).status, "UNKNOWN")
        for bad in ((published[0], (), True), (published[0], published[1], False)):
            self.assertEqual(B.attribute_published_full_step(self.window(), bad,
                run_id="cpu", native_source_sha256=NATIVE_SHA).status, "UNKNOWN")

    def test_corrupt_event_sequence_and_completion_identity_rejected(self):
        journal = self.journal([80, 120, 220, 230]); journal.publish_owner_frame()
        journal.accepted("h2d", "new", 1024); journal.completed("h2d", "new", 1024); journal.publish_owner_frame()
        events, frames, valid = journal.published()
        for event in (replace(events[0], sequence=2), replace(events[0], operation_sequence=9)):
            result = B.attribute_published_full_step(self.window(), ((event,) + events[1:], frames, valid),
                run_id="cpu", native_source_sha256=NATIVE_SHA)
            self.assertEqual(result.status, "UNKNOWN")

    def test_frozen_drain_requires_actual_transferred_bytes_not_requested_completion(self):
        totals = (F.StageTotal(1, 1024),) + (F.StageTotal(0, 0),) * 3
        drain = F.DrainEvidence("cpu", NATIVE_SHA, totals, totals,
            (512, 0, 0, 0), (1, 0, 0, 0), True, 0, True, True, True, 0, 0, 0, 0, 0)
        with self.assertRaises(ValueError): drain.validate("cpu", NATIVE_SHA)
        replace(drain, transferred_bytes=(1024, 0, 0, 0), failed_ops=(0, 0, 0, 0)).validate("cpu", NATIVE_SHA)

    def test_source_only_preflight_reads_six_pins_without_importing_runtime(self):
        result = E.cpu_preflight(SOURCE_ROOT)
        self.assertEqual(len(result["checked_sources"]), 6)
        self.assertFalse(result["gpu_initialized"])
        self.assertFalse(result["source_imported"])
        self.assertFalse(result["executable_native_G2"])
        self.assertFalse(result["model_bytes_verified"])
        self.assertEqual(result["runtime_hook_status"], "not_installed")

    def test_native_run_is_rejected_before_source_or_GPU_import(self):
        result = subprocess.run([sys.executable, "-B", "-I", "-S",
            str(HERE / "run_g2_normal_model_candidate.py"), "--run",
            "--source-root", "/nonexistent/CPU-negative"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["gpu_initialized"])
        self.assertEqual(payload["status"], "BLOCKED_NATIVE_G2_NOT_IMPLEMENTED_OR_QUALIFIED")


if __name__ == "__main__":
    J = load_locked_journal(CPU_ROOT)
    unittest.main(argv=[sys.argv[0]] + remaining_arguments, verbosity=2)
