"""Stdlib CPU replay tests; no native runner, CUDA, GPU timing or I/O claims."""
import argparse
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
import weakref

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p4_runtime_scalar_connector_candidate",
    HERE / "p4_runtime_scalar_connector.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
ADAPTER_PATH = HERE.parent / "collector" / "p4_full_step_frame_adapter.py"
F = None
CLOCK_VALUE = 100
CLOCK_FAULT = False


def cpu_clock():
    global CLOCK_VALUE
    if CLOCK_FAULT:
        raise RuntimeError("CPU clock failed")
    CLOCK_VALUE += 10
    return CLOCK_VALUE


def observer_fault(*args, **kwargs):
    raise RuntimeError("CPU observer failure")


def observer_control_signal(*args, **kwargs):
    raise KeyboardInterrupt("CPU observer cancellation")


class Poison:
    def __getattribute__(self, name):
        raise AssertionError("off observer read poison input")


class Output:
    def __init__(self, ids=("r",), tokens=((7,),)):
        self.req_ids = list(ids)
        self.sampled_token_ids = [list(row) for row in tokens]
        self.req_id_to_index = {rid: i for i, rid in enumerate(ids)}


class Scheduler:
    def __init__(self, ids=("r",), scheduled=1):
        self.num_scheduled_tokens = {rid: scheduled for rid in ids}
        self.total_num_scheduled_tokens = scheduled * len(ids)
        self.scheduled_spec_decode_tokens = {}


class FakeOriginalRunner:
    """Original CPU methods call each other in the audited synchronous order."""
    def __init__(self, *, context=0, prompt=128, ids=("r",)):
        self.use_async_scheduling = False
        self.speculative_config = None
        self.is_pooling_model = False
        self.parallel_config = NS(tensor_parallel_size=1,
            pipeline_parallel_size=1, data_parallel_size=1)
        self.input_batch = NS(num_reqs=len(ids), req_ids=list(ids),
            num_computed_tokens_cpu=[context] * len(ids),
            num_prompt_tokens=[prompt] * len(ids))
        self._profile_step = 0
        self.calls = dict(execute=0, prepare=0, sample=0)
        self.argument_ids = None
        self.sample_argument_ids = None
        self.execute_error = None
        self.prepare_error = None
        self.sample_error = None
        self.direct_return = None
        self.prepare_result = object()
        self.output = None
        self.sample_cpu_clock_tick = False

    def execute_model(self, scheduler_output, marker=None, *, extra=None):
        self.calls["execute"] += 1
        self.argument_ids = (id(scheduler_output), id(marker), id(extra))
        if self.execute_error is not None:
            raise self.execute_error
        self._profile_step += 1
        self._prepare_inputs(scheduler_output, tuple(scheduler_output.num_scheduled_tokens.values()))
        return self.direct_return

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        self.calls["prepare"] += 1
        if self.prepare_error is not None:
            raise self.prepare_error
        return self.prepare_result

    def sample_tokens(self, marker=None, *, extra=None):
        self.calls["sample"] += 1
        self.sample_argument_ids = (id(marker), id(extra))
        if self.sample_error is not None:
            raise self.sample_error
        if self.sample_cpu_clock_tick:
            # Windows 3.12 monotonic resolution can be coarse for a tiny fake;
            # original fixture CPU work waits for one real host clock tick.
            started = C.time.monotonic_ns()
            while C.time.monotonic_ns() == started:
                pass
        output = self.output
        self.output = None
        return output


def source_ref(path):
    data = Path(path).read_bytes()
    return C.SourceRef(str(Path(path).resolve()), len(data), hashlib.sha256(data).hexdigest())


def fixture_binding():
    return C.MethodBinding((source_ref(__file__),), tuple(
        "FakeOriginalRunner." + name for name in C.METHODS), "cpu_fixture")


class ConnectorTests(unittest.TestCase):
    def setUp(self):
        global CLOCK_VALUE, CLOCK_FAULT
        CLOCK_VALUE, CLOCK_FAULT = 100, False

    def make(self, *, runner=None, max_steps=4096, binding=None, clock=cpu_clock):
        runner = runner or FakeOriginalRunner()
        adapter = F.FullStepFrameAdapter("cpu-replay", source_ref(__file__).sha256,
            "c" * 64, enabled=True, max_steps=max_steps)
        connection = C.connect_runtime_scalar_observer(runner, adapter=adapter,
            adapter_source=ADAPTER_PATH, binding=binding or fixture_binding(),
            enabled=True, clock=clock)
        return runner, adapter, connection

    def perform(self, runner, context, scheduled, tokens=(7,), *, prompt=128):
        runner.input_batch.num_computed_tokens_cpu = [context]
        runner.input_batch.num_prompt_tokens = [prompt]
        scheduler = Scheduler(scheduled=scheduled)
        output = Output(tokens=(tokens,))
        runner.output = output
        self.assertIsNone(runner.execute_model(scheduler))
        self.assertIs(runner.sample_tokens(), output)
        return output

    def drain(self, adapter, *, incomplete=False):
        totals = tuple(F.StageTotal(0, 0) for _ in range(4))
        return F.DrainEvidence(adapter.run_id, adapter.native_source_sha256,
            totals, totals, (0, 0, 0, 0), (0, 0, 0, 0), True,
            1 if incomplete else 0, True, True, True, 0, 0, 0, 0, 0)

    def test_off_reads_no_objects_or_clock_and_installs_nothing(self):
        poison = Poison()
        connection = C.connect_runtime_scalar_observer(poison, adapter=poison,
            adapter_source=poison, binding=poison, clock=poison, enabled=False)
        self.assertEqual(connection.status, "off")
        self.assertFalse(connection.enabled)
        sentinel = object()
        calls = []
        result = connection.observe_shutdown(lambda: calls.append(1) or sentinel,
            snapshot_factory=poison, expected_outputs=poison)
        self.assertIs(result, sentinel)
        self.assertEqual(calls, [1])

    def test_original_arguments_once_and_return_identity(self):
        runner, adapter, connection = self.make()
        self.assertTrue(connection.enabled)
        scheduler = Scheduler(scheduled=128)
        marker, extra = object(), object()
        self.assertIsNone(runner.execute_model(scheduler, marker, extra=extra))
        self.assertEqual(runner.argument_ids, (id(scheduler), id(marker), id(extra)))
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=0))
        self.assertEqual(adapter.frames, ())
        self.assertEqual(adapter._pending["phase"], "awaiting_sample")
        output = Output()
        runner.output = output
        self.assertIs(runner.sample_tokens(marker, extra=extra), output)
        self.assertEqual(runner.sample_argument_ids, (id(marker), id(extra)))
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))
        self.assertEqual(len(adapter.frames), 1)
        self.assertEqual(adapter.frames[0].prepared.context_length, 0)
        self.assertEqual(adapter.frames[0].prepared.input_seq_lens_from_cpu_inputs, (128,))

    def test_prepare_keyword_preserves_original_result(self):
        runner, adapter, connection = self.make()
        adapter.begin(0, cpu_clock())
        runner._profile_step = 1
        scheduler = Scheduler(scheduled=128)
        self.assertIs(runner._prepare_inputs(scheduler_output=scheduler,
            num_scheduled_tokens=(128,)), runner.prepare_result)
        self.assertEqual(runner.calls["prepare"], 1)
        self.assertTrue(connection.enabled)
        self.assertEqual(adapter._pending["phase"], "prepared")

    def test_complete_cold_prefill_and_127_decode_is_128_outputs(self):
        runner, adapter, connection = self.make()
        expected = [1000]
        self.perform(runner, 0, 128, (1000,))
        for i in range(127):
            expected.append(1001 + i)
            self.perform(runner, 128 + i, 1, (1001 + i,))
        self.assertEqual(len(adapter.frames), 128)
        self.assertEqual(runner.calls, dict(execute=128, prepare=128, sample=128))
        events = []
        sentinel = object()
        def shutdown():
            events.append("original")
            return sentinel
        def snapshot():
            events.append("provider")
            return self.drain(adapter)
        self.assertIs(connection.observe_shutdown(shutdown, snapshot_factory=snapshot,
            expected_outputs={"r": expected}), sentinel)
        self.assertEqual(events, ["original", "provider"])
        self.assertIsNotNone(connection.closed_run)
        self.assertEqual(connection.closed_run.outputs, (("r", tuple(expected)),))
        self.assertEqual(connection.runtime_hook_status, "not_installed")
        self.assertFalse(connection.GPU_collector_verified)
        self.assertFalse(connection.production_qualified)
        for frame in adapter.frames:
            self.assertIsNone(frame.gpu_elapsed_ns)
            self.assertIsNone(frame.existing_io)
            self.assertIsNone(frame.new_io)

    def test_partial_prefill_empty_then_complete_prefill_output(self):
        runner, adapter, connection = self.make()
        self.perform(runner, 0, 64, ())
        self.perform(runner, 64, 64, (8,))
        self.assertTrue(connection.enabled)
        self.assertEqual(adapter.frames[0].outputs, (("r", ()),))
        self.assertEqual(adapter.frames[1].outputs, (("r", (8,)),))

    def test_partial_prefill_emits_token_invalidates_only_observation(self):
        runner, adapter, connection = self.make()
        output = self.perform(runner, 0, 64, (8,))
        self.assertIsNotNone(output)
        self.assertFalse(connection.enabled)
        self.assertFalse(adapter.valid)
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))

    def test_original_execute_prepare_sample_exceptions_are_same_object(self):
        for field in ("execute_error", "prepare_error", "sample_error"):
            with self.subTest(field=field):
                runner, adapter, connection = self.make()
                error = RuntimeError(field)
                setattr(runner, field, error)
                scheduler = Scheduler(scheduled=128)
                try:
                    runner.execute_model(scheduler)
                    runner.sample_tokens()
                except RuntimeError as observed:
                    self.assertIs(observed, error)
                else:
                    self.fail("original error lost")
                self.assertFalse(connection.enabled)
                self.assertFalse(adapter.valid)
                self.assertEqual(runner.calls["execute"], 1)
                self.assertLessEqual(runner.calls["prepare"], 1)
                self.assertLessEqual(runner.calls["sample"], 1)
                self.assertNotIn("last_error", vars(connection))

    def test_original_error_wins_after_observer_clock_failure(self):
        global CLOCK_FAULT
        runner, adapter, connection = self.make()
        CLOCK_FAULT = True
        error = ValueError("original priority")
        runner.execute_error = error
        with self.assertRaises(ValueError) as caught:
            runner.execute_model(Scheduler(scheduled=128))
        self.assertIs(caught.exception, error)
        self.assertEqual(runner.calls["execute"], 1)

    def test_original_error_wins_over_invalidation_control_signal(self):
        runner, adapter, connection = self.make()
        adapter.invalidate = observer_control_signal
        error = RuntimeError("original priority")
        runner.execute_error = error
        with self.assertRaises(RuntimeError) as caught:
            runner.execute_model(Scheduler(scheduled=128))
        self.assertIs(caught.exception, error)
        self.assertEqual(runner.calls["execute"], 1)
        self.assertFalse(connection.enabled)

    def test_observer_control_signal_is_transparent_when_no_original_error(self):
        runner, adapter, connection = self.make()
        adapter.begin = observer_control_signal
        with self.assertRaises(KeyboardInterrupt):
            runner.execute_model(Scheduler(scheduled=128))
        self.assertEqual(runner.calls["execute"], 0)

    def test_begin_clock_failure_does_not_skip_original(self):
        global CLOCK_FAULT
        runner, adapter, connection = self.make()
        CLOCK_FAULT = True
        self.perform(runner, 0, 128)
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))
        self.assertFalse(connection.enabled)

    def test_end_clock_failure_preserves_sample_identity(self):
        global CLOCK_FAULT
        runner, adapter, connection = self.make()
        runner.execute_model(Scheduler(scheduled=128))
        output = Output()
        runner.output = output
        CLOCK_FAULT = True
        self.assertIs(runner.sample_tokens(), output)
        self.assertEqual(runner.calls["sample"], 1)
        self.assertFalse(connection.enabled)
        self.assertEqual(adapter.frames, ())

    def test_adapter_faults_before_and_after_original_keep_once(self):
        for method in ("begin", "prepared", "executed", "sampled"):
            with self.subTest(method=method):
                runner, adapter, connection = self.make()
                setattr(adapter, method, observer_fault)
                self.perform(runner, 0, 128)
                self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))
                self.assertFalse(connection.enabled)

    def test_observer_invalidate_failure_cannot_replace_original(self):
        runner, adapter, connection = self.make()
        adapter.begin = observer_fault
        adapter.invalidate = observer_fault
        self.perform(runner, 0, 128)
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))
        self.assertFalse(connection.enabled)

    def test_early_execute_return_preserves_identity_and_no_sample_added(self):
        runner, adapter, connection = self.make()
        sentinel = object()
        runner.direct_return = sentinel
        self.assertIs(runner.execute_model(Scheduler(scheduled=128)), sentinel)
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=0))
        self.assertFalse(connection.enabled)
        self.assertEqual(adapter.frames, ())

    def test_async_spec_pooling_parallel_preflight_refuses_without_hooks(self):
        cases = ("async", "spec", "pooling", "parallel")
        for case in cases:
            with self.subTest(case=case):
                runner = FakeOriginalRunner()
                if case == "async": runner.use_async_scheduling = True
                if case == "spec": runner.speculative_config = object()
                if case == "pooling": runner.is_pooling_model = True
                if case == "parallel": runner.parallel_config.tensor_parallel_size = 2
                runner, adapter, connection = self.make(runner=runner)
                self.assertFalse(connection.enabled)
                self.assertFalse(any(name in vars(runner) for name in C.METHODS))
                self.assertIsNone(runner.execute_model(Scheduler(scheduled=128)))
                self.assertEqual(runner.calls["execute"], 1)

    def test_async_changes_after_preflight_invalidates_observer_not_original(self):
        runner, adapter, connection = self.make()
        runner.use_async_scheduling = True
        self.perform(runner, 0, 128)
        self.assertFalse(connection.enabled)
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))

    def test_mixed_context_rejects_observation_without_reexecution(self):
        runner = FakeOriginalRunner(ids=("a", "b"))
        runner, adapter, connection = self.make(runner=runner)
        runner.input_batch.num_computed_tokens_cpu = [0, 1]
        runner.output = Output(("a", "b"), ((1,), (2,)))
        runner.execute_model(Scheduler(("a", "b"), scheduled=127))
        output = runner.output
        self.assertIs(runner.sample_tokens(), output)
        self.assertFalse(connection.enabled)
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))

    def test_contiguous_ordinal_missing_sample_and_duplicate_sample_fail_closed(self):
        for violation in ("ordinal", "missing_sample", "duplicate_sample"):
            with self.subTest(violation=violation):
                runner, adapter, connection = self.make()
                self.perform(runner, 0, 128)
                if violation == "ordinal":
                    runner._profile_step += 1
                    self.perform(runner, 128, 1)
                elif violation == "missing_sample":
                    runner.input_batch.num_computed_tokens_cpu = [128]
                    scheduler = Scheduler()
                    runner.execute_model(scheduler)
                    runner.execute_model(scheduler)
                else:
                    output = Output()
                    runner.output = output
                    self.assertIs(runner.sample_tokens(), output)
                self.assertFalse(connection.enabled)
                self.assertFalse(adapter.valid)

    def test_step_capacity_overflow_retains_original_path(self):
        runner, adapter, connection = self.make(max_steps=1)
        self.perform(runner, 0, 128)
        self.perform(runner, 128, 1)
        self.assertFalse(connection.enabled)
        self.assertEqual(len(adapter.frames), 1)
        self.assertEqual(runner.calls, dict(execute=2, prepare=2, sample=2))

    def test_detach_restores_class_methods_and_foreign_override_is_preserved(self):
        runner, adapter, connection = self.make()
        connection.detach()
        self.assertFalse(any(name in vars(runner) for name in C.METHODS))
        self.assertIs(runner.execute_model.__func__, FakeOriginalRunner.execute_model)
        poison = Poison()
        # With detached originals, observer inputs/clock are no longer consulted.
        connection._adapter = poison
        connection._clock = poison
        self.perform(runner, 0, 128)
        self.assertEqual(adapter.frames, ())
        runner, adapter, connection = self.make()
        foreign = lambda *a, **k: "foreign"
        runner.sample_tokens = foreign
        connection.detach()
        self.assertIs(runner.sample_tokens, foreign)
        self.assertNotIn("execute_model", vars(runner))

    def test_disable_installed_observation_reads_no_adapter_or_clock(self):
        runner, adapter, connection = self.make()
        connection.enabled = False
        adapter.begin = observer_fault
        adapter.prepared = observer_fault
        adapter.executed = observer_fault
        adapter.sampled = observer_fault
        global CLOCK_FAULT
        CLOCK_FAULT = True
        self.perform(runner, 0, 128)
        self.assertEqual(adapter.frames, ())
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))

    def test_weak_refs_do_not_hold_runner_scheduler_output_or_owner(self):
        class Owner: pass
        runner, adapter, connection = self.make()
        owner = Owner()
        runner.owner = owner
        scheduler = Scheduler(scheduled=128)
        output = Output()
        runner.output = output
        refs = [weakref.ref(v) for v in (runner, scheduler, output, owner)]
        runner.execute_model(scheduler)
        runner.sample_tokens()
        del runner, scheduler, output, owner
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(len(adapter.frames), 1)
        self.assertIsNone(connection._runner())

    def test_observer_failure_retains_no_exception_owner(self):
        runner, adapter, connection = self.make()
        adapter.prepared = observer_fault
        scheduler = Scheduler(scheduled=128)
        output = Output()
        runner.output = output
        refs = [weakref.ref(v) for v in (runner, scheduler, output)]
        runner.execute_model(scheduler)
        runner.sample_tokens()
        del runner, scheduler, output
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(connection.last_reason, "observer:RuntimeError")

    def test_original_exception_cycle_does_not_become_connection_owner(self):
        runner, adapter, connection = self.make()
        runner.execute_error = RuntimeError("original error with traceback")
        ref = weakref.ref(runner)
        try:
            runner.execute_model(Scheduler(scheduled=128))
        except RuntimeError:
            pass
        del runner
        gc.collect()
        self.assertIsNone(ref())
        self.assertFalse(connection.enabled)
        self.assertEqual(connection.last_reason, "original_execute_failed")

    def test_source_hash_byte_qualname_origin_and_existing_override_rejected(self):
        ref = source_ref(__file__)
        base = fixture_binding()
        bad = [C.MethodBinding((C.SourceRef(ref.path, ref.nbytes, "0" * 64),), base.qualnames, base.origin),
            C.MethodBinding((C.SourceRef(ref.path, ref.nbytes + 1, ref.sha256),), base.qualnames, base.origin),
            C.MethodBinding(base.refs, ("bad",) + base.qualnames[1:], base.origin),
            C.MethodBinding(base.refs, base.qualnames, "native_candidate"),
            C.MethodBinding((), base.qualnames, base.origin)]
        for binding in bad:
            with self.subTest(binding=binding):
                runner, adapter, connection = self.make(binding=binding)
                self.assertFalse(connection.enabled)
                self.assertFalse(any(name in vars(runner) for name in C.METHODS))
                runner.execute_model(Scheduler(scheduled=128))
                self.assertEqual(runner.calls["execute"], 1)
        runner = FakeOriginalRunner()
        foreign = lambda *a, **k: "original override"
        runner.sample_tokens = foreign
        runner, adapter, connection = self.make(runner=runner)
        self.assertFalse(connection.enabled)
        self.assertIs(runner.sample_tokens, foreign)

    def test_unpinned_code_file_and_adapter_source_binding_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "other.py"
            path.write_text("# CPU unrelated source\n", encoding="utf-8")
            base = fixture_binding()
            binding = C.MethodBinding((source_ref(path),), base.qualnames, base.origin)
            runner, adapter, connection = self.make(binding=binding)
            self.assertFalse(connection.enabled)
        runner = FakeOriginalRunner()
        adapter = F.FullStepFrameAdapter("cpu", "a" * 64, "c" * 64, enabled=True)
        connection = C.connect_runtime_scalar_observer(runner, adapter=adapter,
            adapter_source=ADAPTER_PATH, binding=fixture_binding(), enabled=True)
        self.assertFalse(connection.enabled)
        self.assertFalse(any(name in vars(runner) for name in C.METHODS))

    def test_decorator_file_must_be_pinned_separately(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cpu_decorator.py"
            path.write_text("import functools\ndef decorate(f):\n"
                "    @functools.wraps(f)\n    def wrapper(*a, **k):\n"
                "        return f(*a, **k)\n    return wrapper\n", encoding="utf-8")
            decorator_spec = importlib.util.spec_from_file_location("cpu_temp_decorator", path)
            module = importlib.util.module_from_spec(decorator_spec)
            decorator_spec.loader.exec_module(module)
            class DecoratedRunner(FakeOriginalRunner):
                sample_tokens = module.decorate(FakeOriginalRunner.sample_tokens)
            runner, adapter, connection = self.make(runner=DecoratedRunner())
            self.assertFalse(connection.enabled)
            self.assertFalse(any(name in vars(runner) for name in C.METHODS))
            base = fixture_binding()
            binding = C.MethodBinding(base.refs + (source_ref(path),), base.qualnames, base.origin)
            runner, adapter, connection = self.make(runner=DecoratedRunner(), binding=binding)
            self.assertTrue(connection.enabled, connection.last_reason)
            self.perform(runner, 0, 128)
            self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))

    def test_frozen_adapter_source_drift_rejected_before_import(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.py"
            path.write_bytes(ADAPTER_PATH.read_bytes() + b"\n# drift\n")
            runner = FakeOriginalRunner()
            adapter = F.FullStepFrameAdapter("cpu", source_ref(__file__).sha256, "c" * 64, enabled=True)
            connection = C.connect_runtime_scalar_observer(runner, adapter=adapter,
                adapter_source=path, binding=fixture_binding(), enabled=True)
            self.assertFalse(connection.enabled)
            self.assertFalse(any(name in vars(runner) for name in C.METHODS))
            with self.assertRaises(ValueError):
                C.load_frozen_adapter(path)

    def test_clock_bound_owner_and_closure_rejected(self):
        class Owner:
            def clock(self): return 100
        owner = Owner()
        runner, adapter, connection = self.make(clock=owner.clock)
        self.assertFalse(connection.enabled)
        ref = weakref.ref(owner)
        del owner
        gc.collect()
        self.assertIsNone(ref())
        value = 5
        runner, adapter, connection = self.make(clock=lambda: value)
        self.assertFalse(connection.enabled)

    def test_clock_function_defaults_owner_is_not_retained(self):
        class Owner: pass
        owner = Owner()
        def clock(owner=owner): return 500
        owner_ref, clock_ref = weakref.ref(owner), weakref.ref(clock)
        runner, adapter, connection = self.make(clock=clock)
        self.assertTrue(connection.enabled)
        del clock, owner
        gc.collect()
        self.assertIsNone(clock_ref())
        self.assertIsNone(owner_ref())
        # Expired test clock fails observation; original work still progresses.
        self.perform(runner, 0, 128)
        self.assertFalse(connection.enabled)
        self.assertEqual(runner.calls, dict(execute=1, prepare=1, sample=1))

    def test_native_candidate_custom_clock_rejected_before_runner_read(self):
        adapter = F.FullStepFrameAdapter("cpu-negative", source_ref(__file__).sha256,
            "c" * 64, enabled=True)
        binding = C.MethodBinding((source_ref(__file__),), tuple(
            "GPUModelRunner." + name for name in C.METHODS), "native_candidate")
        connection = C.connect_runtime_scalar_observer(Poison(), adapter=adapter,
            adapter_source=ADAPTER_PATH, binding=binding, enabled=True, clock=cpu_clock)
        self.assertFalse(connection.enabled)
        self.assertEqual(connection.last_reason, "preflight:ValueError")
        # Without the native builtin guard this would touch Poison and produce
        # AssertionError; it cannot silently treat a fixture clock as native.

    def test_builtin_monotonic_clock_records_only_host_ordering(self):
        runner, adapter, connection = self.make(clock=C.time.monotonic_ns)
        self.assertTrue(connection.enabled, connection.last_reason)
        runner.sample_cpu_clock_tick = True
        self.perform(runner, 0, 128)
        self.assertTrue(connection.enabled, connection.last_reason)
        frame = adapter.frames[0]
        self.assertGreater(frame.end_ns, frame.start_ns)
        self.assertIsNone(frame.gpu_elapsed_ns)

    def test_source_read_limit_rejects_before_allocating_file_bytes(self):
        ref = source_ref(__file__)
        for nbytes in (True, 0, C.MAX_SOURCE_BYTES + 1):
            with self.subTest(nbytes=nbytes):
                with self.assertRaises(ValueError):
                    C.SourceRef(ref.path, nbytes, ref.sha256).checked_path()

    def test_shutdown_original_exception_priority_and_provider_not_read(self):
        runner, adapter, connection = self.make()
        error = RuntimeError("original shutdown")
        def shutdown(): raise error
        with self.assertRaises(RuntimeError) as caught:
            connection.observe_shutdown(shutdown, snapshot_factory=Poison(), expected_outputs=Poison())
        self.assertIs(caught.exception, error)
        self.assertFalse(connection.enabled)

    def test_shutdown_provider_failure_returns_original_identity(self):
        runner, adapter, connection = self.make()
        self.perform(runner, 0, 128)
        sentinel = object()
        self.assertIs(connection.observe_shutdown(lambda: sentinel,
            snapshot_factory=observer_fault, expected_outputs={"r": [7]}), sentinel)
        self.assertFalse(connection.enabled)
        self.assertIsNone(connection.closed_run)

    def test_shutdown_pending_or_partial_drain_cannot_close(self):
        for pending in (True, False):
            with self.subTest(pending=pending):
                runner, adapter, connection = self.make()
                if pending:
                    runner.execute_model(Scheduler(scheduled=128))
                else:
                    self.perform(runner, 0, 128)
                sentinel = object()
                result = connection.observe_shutdown(lambda: sentinel,
                    snapshot_factory=lambda: self.drain(adapter, incomplete=True),
                    expected_outputs={"r": [7]})
                self.assertIs(result, sentinel)
                self.assertFalse(connection.enabled)
                self.assertIsNone(connection.closed_run)

    def test_no_output_truncation_at_shutdown(self):
        runner, adapter, connection = self.make()
        self.perform(runner, 0, 128, (7,))
        self.perform(runner, 128, 1, (8,))
        sentinel = object()
        self.assertIs(connection.observe_shutdown(lambda: sentinel,
            snapshot_factory=lambda: self.drain(adapter), expected_outputs={"r": [7]}), sentinel)
        self.assertFalse(connection.enabled)
        self.assertIsNone(connection.closed_run)
        self.assertEqual(len(adapter.frames), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector-source", type=Path, default=ADAPTER_PATH)
    parsed, remaining = parser.parse_known_args()
    ADAPTER_PATH = parsed.collector_source.resolve()
    F = C.load_frozen_adapter(ADAPTER_PATH)
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
