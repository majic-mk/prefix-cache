"""CPU replay of original pinned UniProc RPC methods, not a second executor."""
import argparse
import ast
from concurrent.futures import Future
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import weakref

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, default=Path(__file__).parent.parent / "g2_source_readonly")
parser.add_argument("--scalar-source", type=Path, default=Path(__file__).parent.parent / "runtime_connector/p4_runtime_scalar_connector.py")
parser.add_argument("--frame-source", type=Path, default=Path(__file__).parent.parent / "collector/p4_full_step_frame_adapter.py")
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
source = Path(__file__).with_name("g2_worker_observation.py")
spec = importlib.util.spec_from_file_location("g2_worker_under_test", source)
G = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = G
spec.loader.exec_module(G)
C = G.load_pinned(args.scalar_source, *G.FROZEN["scalar"])
F = C.load_frozen_adapter(args.frame_source)
TEST_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RUN = "g2-cpu-original-rpc"
NATIVE = "b" * 64
fixture_clock_tick = 100


def fixture_clock():
    # Windows monotonic resolution can yield equal immediate fake calls.
    global fixture_clock_tick
    fixture_clock_tick += 100
    return fixture_clock_tick


class FakeRunner:
    def __init__(self):
        self._profile_step = 0
        self.use_async_scheduling = False
        self.speculative_config = None
        self.is_pooling_model = False
        self.parallel_config = NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1)
        self.input_batch = NS(num_reqs=1, req_ids=["native-0"], num_computed_tokens_cpu=[0], num_prompt_tokens=[128])
        self.execute_calls = self.prepare_calls = self.sample_calls = 0
        self.execute_error = self.sample_error = None
        self.output = None
        self.seen_arguments = None

    def execute_model(self, scheduler_output, *extra, **kwargs):
        self.execute_calls += 1
        self.seen_arguments = (scheduler_output, extra, kwargs)
        if self.execute_error is not None:
            raise self.execute_error
        self._profile_step += 1
        self._prepare_inputs(scheduler_output)
        return None

    def _prepare_inputs(self, scheduler_output):
        self.prepare_calls += 1
        self.scheduled = scheduler_output.num_scheduled_tokens["native-0"]
        return self.input_batch

    def sample_tokens(self, grammar_output=None):
        self.sample_calls += 1
        if self.sample_error is not None:
            raise self.sample_error
        self.output = NS(req_ids=["native-0"], req_id_to_index={"native-0": 0},
                         sampled_token_ids=[[self.sample_calls - 1]])
        self.input_batch.num_computed_tokens_cpu[0] += self.scheduled
        return self.output


class FakeWrapper:
    def __init__(self, runner): self.model_runner = runner
    def execute_model(self, *a, **kw): return self.model_runner.execute_model(*a, **kw)
    def sample_tokens(self, *a, **kw): return self.model_runner.sample_tokens(*a, **kw)


class FakeAsyncOutput: pass


def original_uni_class():
    """Compile only exact original methods in a test module, no Torch import.

    Product code never calls this. Only the test-global run_method dispatcher
    dependency is a fixture; the RPC/execute/sample AST body is unmodified.
    """
    path = args.source_root / "third_party/work/vllm-author-p4-02-cpu/vllm/v1/executor/uniproc_executor.py"
    raw = path.read_bytes()
    assert len(raw) == 7420
    assert hashlib.sha256(raw).hexdigest() == "93fac0cbecffb9800b1a90746fd179bfdc7c61bf22c6c1337ebe0d917157f247"
    tree = ast.parse(raw)
    author = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UniProcExecutor")
    wanted = {"collective_rpc", "execute_model", "sample_tokens"}
    methods = [n for n in author.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(methods) == 3
    fixture = ast.ClassDef(name="UniProcExecutor", bases=[], keywords=[], body=methods,
                           decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fixture], type_ignores=[])
    ast.fix_missing_locations(module)
    def run_method(worker, method, positional, keywords):
        return getattr(worker, method)(*positional, **keywords) if type(method) is str else method(worker, *positional, **keywords)
    namespace = {"Future": Future, "Any": object, "AsyncModelRunnerOutput": FakeAsyncOutput,
                 "run_method": run_method}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["UniProcExecutor"]


Uni = original_uni_class()


class FakeEvent:
    records = []
    next_ready = True
    query_value = True
    factory_error = None
    record_error = None
    def __init__(self, *, enable_timing):
        if type(self).factory_error is not None: raise type(self).factory_error
        assert enable_timing is True
        self.ready = type(self).next_ready
        self.record_calls = self.query_calls = self.elapsed_calls = 0
        type(self).records.append(self)
    def record(self):
        self.record_calls += 1
        if type(self).record_error is not None: raise type(self).record_error
    def query(self):
        self.query_calls += 1
        return False if not self.ready else type(self).query_value
    def elapsed_time(self, other): self.elapsed_calls += 1; return 1.25
    def synchronize(self): raise AssertionError("observer must not synchronize")


def binding():
    return C.MethodBinding((C.SourceRef(str(Path(__file__).absolute()), Path(__file__).stat().st_size, TEST_SHA),),
        tuple("FakeRunner." + name for name in C.METHODS), "cpu_fixture")


def connect(worker, **overrides):
    kw = dict(scalar_source=args.scalar_source, frame_source=args.frame_source,
              binding=binding(), run_id=RUN, native_source_sha256=NATIVE,
              event_factory=FakeEvent, enabled=True, clock=fixture_clock)
    kw.update(overrides)
    return G.connect_worker_observation(worker, **kw)


def scheduler(n=1):
    return NS(num_scheduled_tokens={"native-0": n}, total_num_scheduled_tokens=n,
              scheduled_spec_decode_tokens={})


class Poison:
    def __getattribute__(self, name): raise AssertionError("off read input " + name)


class WorkerContracts(unittest.TestCase):
    def setUp(self):
        FakeEvent.records = []
        FakeEvent.next_ready = FakeEvent.query_value = True
        FakeEvent.factory_error = FakeEvent.record_error = None

    def make(self, **kw):
        runner = FakeRunner()
        worker = FakeWrapper(runner)
        conn = connect(worker, **kw)
        self.assertTrue(conn.enabled, conn.reason)
        return runner, worker, conn

    def test_off_reads_none_of_worker_paths_binding_factory_or_clock(self):
        p = Poison()
        conn = G.connect_worker_observation(p, scalar_source=p, frame_source=p,
            binding=p, run_id=p, native_source_sha256=p, event_factory=p, clock=p, enabled=False)
        self.assertEqual(conn.status, "off")
        self.assertIsNone(conn.scalar)

    def test_original_uniproc_future_none_does_not_close_before_original_sample(self):
        r, w, c = self.make()
        executor = Uni(); executor.driver_worker = w
        before = scheduler(128)
        future = executor.execute_model(before, non_block=True)
        self.assertIs(type(future), Future)
        self.assertIsNone(future.result())
        self.assertEqual(c.frames, ())
        self.assertIsNotNone(c.events.active)
        self.assertEqual([e.record_calls for e in FakeEvent.records], [1, 0])
        sampled = executor.sample_tokens(None, non_block=True).result()
        self.assertIs(sampled, r.output)
        self.assertEqual((r.execute_calls, r.prepare_calls, r.sample_calls), (1, 1, 1))
        self.assertIsNone(c.events.active)
        self.assertEqual(c.frames[0].prepared.context_length, 0)
        self.assertEqual(c.frames[0].outputs, (("native-0", (0,)),))
        self.assertEqual([e.record_calls for e in FakeEvent.records], [1, 1])

    def test_full_actual_cold_prefill_and_127_decode_outputs_are_not_truncated(self):
        r, w, c = self.make()
        ex = Uni(); ex.driver_worker = w
        observed = []
        for i in range(128):
            self.assertIsNone(ex.execute_model(scheduler(128 if i == 0 else 1), non_block=True).result())
            output = ex.sample_tokens(None, non_block=True).result()
            self.assertIs(output, r.output)
            observed.extend(output.sampled_token_ids[0])
            c.resolve_ready()
        self.assertEqual(observed, list(range(128)))
        self.assertEqual((r.execute_calls, r.prepare_calls, r.sample_calls), (128, 128, 128))
        self.assertEqual(len(c.frames), 128)
        self.assertEqual(c.frames[0].prepared.context_length, 0)
        self.assertEqual(sum(x.prepared.step_kind == "decode" for x in c.frames), 127)
        self.assertEqual(len(c.events.diagnostics), 128)
        self.assertTrue(all(x.gpu_elapsed_ns is None and x.mapped_start_ns is None and
                            not x.production_qualified for x in c.events.diagnostics))

    def test_return_and_arguments_preserve_identity(self):
        r, w, c = self.make()
        s, marker = scheduler(128), object()
        self.assertIsNone(w.execute_model(s, marker, payload=marker))
        seen = r.seen_arguments
        self.assertIs(seen[0], s); self.assertIs(seen[1][0], marker); self.assertIs(seen[2]["payload"], marker)
        self.assertIs(w.sample_tokens(marker), r.output)

    def test_original_execute_exception_is_same_object_no_retry(self):
        r, w, c = self.make()
        error = RuntimeError("original")
        r.execute_error = error
        with self.assertRaises(RuntimeError) as caught: w.execute_model(scheduler(128))
        self.assertIs(caught.exception, error)
        self.assertEqual(r.execute_calls, 1)
        self.assertFalse(c.enabled); self.assertIsNone(c.events.active)

    def test_original_sample_exception_is_same_object_no_retry(self):
        r, w, c = self.make()
        w.execute_model(scheduler(128)); error = ValueError("original sample")
        r.sample_error = error
        with self.assertRaises(ValueError) as caught: w.sample_tokens()
        self.assertIs(caught.exception, error)
        self.assertEqual(r.sample_calls, 1); self.assertEqual(c.frames, ())

    def test_original_control_signal_not_swallowed(self):
        r, w, c = self.make()
        signal = KeyboardInterrupt()
        r.execute_error = signal
        with self.assertRaises(KeyboardInterrupt) as caught: w.execute_model(scheduler(128))
        self.assertIs(caught.exception, signal); self.assertEqual(r.execute_calls, 1)

    def test_event_factory_exception_falls_back_original_execute_sample_once(self):
        r, w, c = self.make()
        FakeEvent.factory_error = ValueError("factory")
        self.assertIsNone(w.execute_model(scheduler(128)))
        self.assertIs(w.sample_tokens(), r.output)
        self.assertEqual((r.execute_calls, r.prepare_calls, r.sample_calls), (1, 1, 1))
        self.assertFalse(c.enabled); self.assertEqual(c.frames, ())

    def test_event_end_exception_does_not_change_successful_original_output(self):
        r, w, c = self.make(); w.execute_model(scheduler(128))
        FakeEvent.record_error = ValueError("end")
        self.assertIs(w.sample_tokens(), r.output)
        self.assertEqual(r.sample_calls, 1); self.assertFalse(c.enabled)

    def test_query_unready_and_nonbool_fail_closed_without_synchronize(self):
        r, w, c = self.make(); FakeEvent.next_ready = False
        w.execute_model(scheduler(128)); w.sample_tokens()
        self.assertEqual(c.resolve_ready(), ()); self.assertEqual(len(c.events.pending), 1)
        for e in FakeEvent.records: e.ready = True
        FakeEvent.query_value = 1
        self.assertEqual(c.resolve_ready(), ()); self.assertFalse(c.enabled)

    def test_disable_installed_observer_restores_original_progress_without_events(self):
        r, w, c = self.make(); c.disable()
        self.assertIsNone(w.execute_model(scheduler(128)))
        self.assertIs(w.sample_tokens(), r.output)
        self.assertEqual(FakeEvent.records, [])
        self.assertEqual((r.execute_calls, r.prepare_calls, r.sample_calls), (1, 1, 1))

    def test_detach_restores_class_methods_and_preserves_later_foreign_override(self):
        r, w, c = self.make(); c.detach()
        self.assertNotIn("execute_model", vars(r)); self.assertNotIn("_prepare_inputs", vars(r))
        self.assertNotIn("sample_tokens", vars(r))
        r, w, c = self.make(); foreign = lambda: 17
        r.sample_tokens = foreign; c.detach()
        self.assertIs(r.sample_tokens, foreign)

    def test_async_or_speculative_domain_rejected_without_original_execution(self):
        for field, value in (("use_async_scheduling", True), ("speculative_config", object())):
            r = FakeRunner(); setattr(r, field, value)
            c = connect(FakeWrapper(r))
            self.assertFalse(c.enabled)
            self.assertNotIn("execute_model", vars(r))
            self.assertEqual(r.execute_calls, 0)

    def test_source_drift_rejected_without_installation(self):
        r = FakeRunner()
        bad = C.MethodBinding((C.SourceRef(str(Path(__file__).absolute()), Path(__file__).stat().st_size,
                                          "0" * 64),), binding().qualnames, "cpu_fixture")
        c = connect(FakeWrapper(r), binding=bad)
        self.assertFalse(c.enabled); self.assertNotIn("execute_model", vars(r))

    def test_pending_bound_only_stops_observation_original_remains_once(self):
        r, w, c = self.make(max_pending=1)
        w.execute_model(scheduler(128)); w.sample_tokens()
        w.execute_model(scheduler()); self.assertIs(w.sample_tokens(), r.output)
        self.assertFalse(c.enabled)
        self.assertEqual((r.execute_calls, r.sample_calls), (2, 2))

    def test_runner_wrapper_scheduler_output_and_factory_owner_not_retained(self):
        r, w, c = self.make()
        s = scheduler(128); w.execute_model(s); result = w.sample_tokens(); c.resolve_ready()
        refs = [weakref.ref(x) for x in (r, w)]
        # NS is not weakrefable; use replacement ordinary objects for these edges.
        class Holder: pass
        owner = Holder()
        def factory(*, enable_timing, held=owner): return FakeEvent(enable_timing=enable_timing)
        r2 = FakeRunner(); w2 = FakeWrapper(r2); c2 = connect(w2, event_factory=factory)
        owner_ref = weakref.ref(owner)
        del owner, factory, w2, r2
        del r, w, s, result
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertIsNone(owner_ref()); self.assertIsNone(c2._factory())

    def test_factory_gone_invalidates_only_observation_and_original_still_runs(self):
        r = FakeRunner(); w = FakeWrapper(r)
        def temporary(*, enable_timing): return FakeEvent(enable_timing=enable_timing)
        c = connect(w, event_factory=temporary); del temporary; gc.collect()
        self.assertIsNone(w.execute_model(scheduler(128)))
        self.assertIs(w.sample_tokens(), r.output); self.assertFalse(c.enabled)

    def test_unqualified_native_event_has_no_host_mapping_or_gpu_cost_field(self):
        events = G.QueryOnlyEventObserver(RUN, "native_candidate")
        events.begin(3, FakeEvent)
        events.finish(NS(native_step_ordinal=3, gpu_elapsed_ns=None,
                         prepared=NS(context_basis="pre_computed_tokens")))
        row = events.resolve_ready()[0]
        self.assertEqual(row.raw_event_elapsed_ns, 1250000)
        self.assertEqual(row.origin, "native_candidate")
        self.assertIsNone(row.gpu_elapsed_ns); self.assertIsNone(row.mapped_start_ns)
        self.assertFalse(row.cross_clock_mapping_verified); self.assertFalse(row.GPU_collector_verified)

    def test_diagnostic_constructor_cannot_assert_gpu_or_mapping_qualification(self):
        for keyword in ("gpu_elapsed_ns", "mapped_start_ns", "production_qualified", "GPU_collector_verified"):
            with self.assertRaises(TypeError):
                G.StepEventDiagnostic(RUN, 0, "cpu_fixture", 100, **{keyword: True})
        with self.assertRaises(ValueError): G.StepEventDiagnostic(RUN, 0, "native_candidate", True)

    def test_origin_and_scope_reject_owner_objects_and_string_subclasses(self):
        class Owner: pass
        class EqualityOrigin:
            def __init__(self, owner): self.owner = owner
            def __eq__(self, other): return True
        class OriginSubclass(str): pass
        owner = Owner(); owner_ref = weakref.ref(owner)
        origin = EqualityOrigin(owner)
        def reject_all(value):
            with self.assertRaises(ValueError): G.QueryOnlyEventObserver(RUN, value)
            with self.assertRaises(ValueError): G.StepEventDiagnostic(RUN, 0, value, 1)
            with self.assertRaises(ValueError): G.StepEventDiagnostic(RUN, 0, "cpu_fixture", 1, scope=value)
            bad = C.MethodBinding(binding().refs, binding().qualnames, value)
            r = FakeRunner(); c = connect(FakeWrapper(r), binding=bad)
            self.assertFalse(c.enabled); self.assertIsNone(c.scalar)
        reject_all(origin)
        reject_all(OriginSubclass("cpu_fixture"))
        del origin, owner
        gc.collect()
        self.assertIsNone(owner_ref())

    def test_runner_source_ref_is_found_by_code_file_not_last_ref(self):
        own = Path(G.__file__).resolve()
        reversed_binding = C.MethodBinding(binding().refs + (
            C.SourceRef(str(own), own.stat().st_size, hashlib.sha256(own.read_bytes()).hexdigest()),),
            binding().qualnames, "cpu_fixture")
        r, w, c = self.make(binding=reversed_binding)
        w.execute_model(scheduler(128)); self.assertIs(w.sample_tokens(), r.output)
        self.assertEqual(c.frames[0].prepared.context_length, 0)

    def test_observer_control_signals_are_transparent_cancellation_not_retry(self):
        # Same policy as frozen scalar connector: ordinary Exception fails closed;
        # KeyboardInterrupt/SystemExit are external control cancellation signals.
        r, w, c = self.make(); signal = KeyboardInterrupt()
        FakeEvent.factory_error = signal
        with self.assertRaises(KeyboardInterrupt) as caught: w.execute_model(scheduler(128))
        self.assertIs(caught.exception, signal); self.assertEqual(r.execute_calls, 0)
        FakeEvent.factory_error = None
        r, w, c = self.make(); w.execute_model(scheduler(128)); signal = SystemExit(7)
        FakeEvent.record_error = signal
        with self.assertRaises(SystemExit) as caught: w.sample_tokens()
        self.assertIs(caught.exception, signal); self.assertEqual(r.sample_calls, 1)
        self.assertEqual(c.frames[0].outputs, (("native-0", (0,)),))


if __name__ == "__main__": unittest.main(verbosity=2)
