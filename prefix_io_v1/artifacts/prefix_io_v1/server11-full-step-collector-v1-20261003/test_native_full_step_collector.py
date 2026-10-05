"""CPU integration with the unchanged G2 scalar/event observer; never CUDA."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parents[1] / "prefix_io_v1_server09_candidates"
parser = argparse.ArgumentParser()
parser.add_argument("--g2-source", type=Path, default=DEFAULT / "g2_normal_worker_site_cache_v4_final/g2_worker_observation.py")
parser.add_argument("--scalar-source", type=Path, default=DEFAULT / "runtime_connector/p4_runtime_scalar_connector.py")
parser.add_argument("--frame-source", type=Path, default=DEFAULT / "collector/p4_full_step_frame_adapter.py")
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("server11_fullstep_test", HERE / "native_full_step_collector.py")
G = load("server11_original_g2_worker", args.g2_source)
C = G.load_pinned(args.scalar_source, *G.FROZEN["scalar"])
F = C.load_frozen_adapter(args.frame_source)
TICK = 1000


def clock():
    global TICK
    TICK += 10
    return TICK


class FakeEvent:
    ready = True
    query_result = None
    instances = []
    def __init__(self, *, enable_timing):
        assert enable_timing is True
        self.record_calls = self.query_calls = self.elapsed_calls = 0
        self.synchronize_calls = 0
        type(self).instances.append(self)
    def record(self):
        self.record_calls += 1
    def query(self):
        self.query_calls += 1
        return self.ready if self.query_result is None else self.query_result
    def elapsed_time(self, other):
        self.elapsed_calls += 1
        return 0.000001
    def synchronize(self):
        self.synchronize_calls += 1
        raise AssertionError("collector must never synchronize")


class FakeRunner:
    def __init__(self):
        self._profile_step = 37
        self.use_async_scheduling = False
        self.speculative_config = None
        self.is_pooling_model = False
        self.parallel_config = NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1)
        self.input_batch = NS(num_reqs=1, req_ids=["native-cpu"],
            num_computed_tokens_cpu=[0], num_prompt_tokens=[128])
        self.calls = dict(execute=0, prepare=0, sample=0)
        self.error = None
        self.last_output = None
    def execute_model(self, scheduler):
        self.calls["execute"] += 1
        if self.error is not None:
            raise self.error
        self._profile_step += 1
        self._prepare_inputs(scheduler)
        return None
    def _prepare_inputs(self, scheduler):
        self.calls["prepare"] += 1
        self.scheduled = scheduler.num_scheduled_tokens["native-cpu"]
        return self.input_batch
    def sample_tokens(self, ignored=None):
        self.calls["sample"] += 1
        self.last_output = NS(req_ids=["native-cpu"], req_id_to_index={"native-cpu": 0},
                              sampled_token_ids=[[self.calls["sample"]]])
        self.input_batch.num_computed_tokens_cpu[0] += self.scheduled
        return self.last_output


def build(action=None, selected=(16,)):
    runner = FakeRunner()
    path = Path(__file__).resolve()
    binding = C.MethodBinding((C.SourceRef(str(path), path.stat().st_size,
        hashlib.sha256(path.read_bytes()).hexdigest()),),
        tuple("FakeRunner." + name for name in C.METHODS), "cpu_fixture")
    capture = M.FullStepCapture(run_id="cpu-128", origin="cpu_fixture",
        selected_offsets=selected, action=action, clock=clock)
    capture.event_source = M.event_class_source(FakeEvent, "cpu_fixture")
    capture.factory = M.BoundedEventFactory(FakeEvent, clock=clock)
    capture.observer = G.connect_worker_observation(NS(model_runner=runner),
        scalar_source=args.scalar_source, frame_source=args.frame_source,
        binding=binding, run_id="cpu-128", native_source_sha256="b" * 64,
        event_factory=capture.factory, enabled=True, max_steps=128, max_pending=128,
        clock=clock)
    assert capture.observer.enabled, capture.observer.reason
    capture.attach_prepare(runner)
    return runner, capture


def step(runner, offset):
    tokens = 128 if offset == 0 else 1
    schedule = NS(num_scheduled_tokens={"native-cpu": tokens},
                  total_num_scheduled_tokens=tokens, scheduled_spec_decode_tokens={})
    assert runner.execute_model(schedule) is None
    assert runner.sample_tokens() is runner.last_output


class CollectorTests(unittest.TestCase):
    def setUp(self):
        FakeEvent.ready, FakeEvent.query_result = True, None
        FakeEvent.instances = []

    def test_real_frozen_g2_observer_full128_and_original_once(self):
        runner, capture = build()
        for offset in range(128):
            step(runner, offset)
        result = capture.export()
        self.assertTrue(result["valid"], result["failures"])
        self.assertEqual(runner.calls, dict(execute=128, prepare=128, sample=128))
        self.assertEqual(result["frames"][0]["prepared"]["context_length"], 0)
        self.assertEqual(len(result["frames"]), 128)
        self.assertEqual(len(result["event_witnesses"]), 128)
        self.assertEqual(result["actions"][0]["step_offset"], 16)
        self.assertFalse(result["actions"][0]["action_enabled"])
        self.assertTrue(all(event.record_calls == 1 and event.synchronize_calls == 0
                            for event in FakeEvent.instances))
        self.assertFalse(result["production_qualified"])
        capture.detach()
        self.assertNotIn("execute_model", vars(runner))
        self.assertNotIn("_prepare_inputs", vars(runner))
        self.assertNotIn("sample_tokens", vars(runner))

    def test_native_action_after_actual_start_query_before_event_end(self):
        calls = []
        def action(offset, ordinal):
            calls.append((offset, ordinal, clock()))
            return {"accepted": True, "hashes": ["ab" * 32]}
        runner, capture = build(action)
        for offset in range(128):
            step(runner, offset)
        result = capture.export()
        self.assertTrue(result["valid"], result["failures"])
        self.assertEqual(len(calls), 1)
        witness, action_row = result["event_witnesses"][16], result["actions"][0]
        self.assertLess(witness["start_completed_query_ns"], calls[0][2])
        self.assertLess(calls[0][2], witness["end_record_before_ns"])
        self.assertLess(action_row["trigger_before_ns"], calls[0][2])

    def test_unready_start_does_not_wait_or_inject_io(self):
        calls = []
        runner, capture = build(lambda *items: calls.append(items))
        for offset in range(16):
            step(runner, offset)
        FakeEvent.ready = False
        step(runner, 16)
        self.assertFalse(capture.valid)
        self.assertEqual(calls, [])
        self.assertEqual(runner.calls["sample"], 17)
        self.assertTrue(all(event.synchronize_calls == 0 for event in FakeEvent.instances))

    def test_unready_tail_is_invalid_not_missing_success(self):
        runner, capture = build()
        for offset in range(128):
            step(runner, offset)
        FakeEvent.ready = False
        result = capture.export()
        self.assertFalse(result["valid"])
        self.assertEqual(result["event_witnesses"], [])

    def test_observer_action_failure_preserves_original_model_progress(self):
        def fail(*items):
            raise RuntimeError("native preload refused")
        runner, capture = build(fail)
        for offset in range(128):
            step(runner, offset)
        self.assertEqual(runner.calls["sample"], 128)
        result = capture.export()
        self.assertFalse(result["valid"])
        self.assertIn("native preload refused", " ".join(result["failures"]))

    def test_control_signal_is_not_hidden(self):
        def stop(*items):
            raise KeyboardInterrupt()
        runner, capture = build(stop)
        for offset in range(16):
            step(runner, offset)
        with self.assertRaises(KeyboardInterrupt):
            step(runner, 16)
        self.assertFalse(capture.valid)

    def test_original_exception_identity_no_retry(self):
        runner, capture = build()
        error = RuntimeError("original model")
        runner.error = error
        with self.assertRaises(RuntimeError) as caught:
            step(runner, 0)
        self.assertIs(caught.exception, error)
        self.assertEqual(runner.calls["execute"], 1)
        self.assertEqual(runner.calls["sample"], 0)

    def test_metadata_refuses_owner_and_still_original_progresses(self):
        runner, capture = build(lambda *items: {"owner": object()})
        for offset in range(17):
            step(runner, offset)
        self.assertFalse(capture.valid)
        self.assertEqual(runner.calls["sample"], 17)

    def test_incomplete_run_cannot_export_full_trace(self):
        runner, capture = build()
        for offset in range(17):
            step(runner, offset)
        self.assertFalse(capture.export()["valid"])

    def test_duplicate_event_record_rejected(self):
        proxy = M.EventProxy(FakeEvent(enable_timing=True), clock)
        proxy.record()
        with self.assertRaises(ValueError):
            proxy.record()
        self.assertEqual(proxy.raw.record_calls, 1)

    def test_no_elapsed_before_actual_both_completed(self):
        one, two = [M.EventProxy(FakeEvent(enable_timing=True), clock) for _ in range(2)]
        one.record(); two.record()
        with self.assertRaises(ValueError):
            one.elapsed_time(two)
        one.query()
        with self.assertRaises(ValueError):
            one.elapsed_time(two)
        two.query()
        self.assertEqual(one.elapsed_time(two), 0.000001)

    def test_nonbool_query_rejected(self):
        proxy = M.EventProxy(FakeEvent(enable_timing=True), clock)
        proxy.record()
        FakeEvent.query_result = 1
        with self.assertRaises(ValueError):
            proxy.query()

    def test_foreign_prepare_override_is_preserved(self):
        runner, capture = build()
        foreign = lambda *items: None
        runner._prepare_inputs = foreign
        capture.detach()
        self.assertIs(runner._prepare_inputs, foreign)

    def test_native_origin_refuses_fake_event_and_clock(self):
        with self.assertRaises(ValueError):
            M.event_class_source(FakeEvent, "native_gpu_recording")
        with self.assertRaises(ValueError):
            M.FullStepCapture(run_id="native", origin="native_gpu_recording",
                selected_offsets=(16,), action=None, clock=clock)

    def test_selected_prefill_and_duplicate_offsets_rejected(self):
        for offsets in ((0,), (16, 16), (128,), (True,)):
            with self.assertRaises(ValueError):
                M.FullStepCapture(run_id="cpu", origin="cpu_fixture",
                    selected_offsets=offsets, action=None)

    def test_bounded_factory_refuses_more_than_original128_pairs(self):
        factory = M.BoundedEventFactory(FakeEvent, clock=clock, max_steps=1)
        factory(enable_timing=True); factory(enable_timing=True)
        with self.assertRaises(ValueError):
            factory(enable_timing=True)


if __name__ == "__main__":
    unittest.main()
