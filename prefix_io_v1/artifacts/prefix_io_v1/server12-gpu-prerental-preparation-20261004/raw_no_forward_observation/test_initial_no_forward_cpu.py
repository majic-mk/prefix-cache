"""CPU-only replay of frozen author AST and original observer ordering."""
import argparse
import ast
import contextlib
import copy
from dataclasses import dataclass
from functools import cached_property
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
import weakref

HERE = Path(__file__).resolve().parent
PREP = HERE.parent
ROOT = PREP.parents[2]
def actual_source(local_name, remote_name, sha):
    for name in (local_name, remote_name):
        path = ROOT / name
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, "actual source differs: " + name
            return path
    raise FileNotFoundError("neither explicit actual source layout exists: " + remote_name)


ADAPTER = actual_source("artifacts/prefix_io_v1_server09_candidates/collector/p4_full_step_frame_adapter.py",
    "artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py",
    "bea850fd60831010c2ac0bf88c1f5f121f3ffca50d63ab56759319ed6e512582")
SCALAR = actual_source("artifacts/prefix_io_v1_server09_candidates/runtime_connector/p4_runtime_scalar_connector.py",
    "artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py",
    "347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb")
WORKER = actual_source("artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/g2_worker_observation.py",
    "artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/g2_worker_observation.py",
    "096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b")
AUTHOR = PREP / "source_inputs/gpu_model_runner.py"
META = actual_source("artifacts/prefix_io_v1_server12_candidates/strong_gpu_system_verification_20261005/SOURCE_NO_FORWARD_scheduler_output.py",
    "artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/SOURCE_NO_FORWARD_scheduler_output.py",
    "52ccd683f1709dd49b79a5837c3db174ae6135320ac1f7a49ef42c4d6041bad7")
PINNED = {ADAPTER: "bea850fd60831010c2ac0bf88c1f5f121f3ffca50d63ab56759319ed6e512582",
          SCALAR: "347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb",
          AUTHOR: "61828af13a5f523c9683df560374d75bc1df93254a9ac4a449c95447100e9075"}


def reference(path):
    raw = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def load(path, name):
    if path in PINNED:
        assert reference(path)["sha256"] == PINNED[path]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load(PREP / "runner/bounded_native_full_step_collector_v2.py", "_initial_v2_cpu")
S = load(SCALAR, "_initial_original_scalar_cpu")
W = load(WORKER, "_initial_original_worker_cpu")
F = S.load_frozen_adapter(ADAPTER)
OLD = load(PREP / "runner/bounded_native_full_step_collector.py", "_initial_frozen_capture_cpu")
assert reference(PREP / "runner/bounded_native_full_step_collector.py")["sha256"] == V2.ORIGINAL_COLLECTOR_SHA


def compile_definition(node, path, namespace, *, strip_signature=False):
    node = copy.deepcopy(node)
    node.decorator_list = []
    if strip_signature:
        node.returns = None
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.annotation = None
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec", dont_inherit=True), namespace)
    return namespace[node.name]


meta_module = types.ModuleType("_initial_actual_metadata_AST_cpu")
meta_module.__file__ = str(META)
sys.modules[meta_module.__name__] = meta_module
vars(meta_module).update(dataclass=dataclass, cached_property=cached_property)
for node in ast.parse(META.read_bytes()).body:
    if isinstance(node, ast.ClassDef) and node.name in ("CachedRequestData", "SchedulerOutput"):
        # Retain actual dataclass defaults/fields/methods; only the decorator
        # is reapplied explicitly since compile_definition removes decorators.
        cls = compile_definition(node, META, vars(meta_module))
        vars(meta_module)[node.name] = dataclass(cls)
TYPES = meta_module.SchedulerOutput, meta_module.CachedRequestData


class StopPositiveCPUProbe(RuntimeError):
    pass


class CPUEvents:
    def __init__(self, trace): self.trace = trace
    def record(self): self.trace.append("cpu_event_record")


class CPUArray(list):
    def max(self): return max(self)


def fixture(*, native_exception=None):
    trace = []
    sentinel = object()
    namespace = dict(has_kv_transfer_group=lambda: True,
        get_kv_transfer_group=lambda: types.SimpleNamespace(handle_preemptions=lambda meta: trace.append("original_preemptions")),
        has_ec_transfer=lambda: False, record_function_or_nullcontext=lambda *a: contextlib.nullcontext(),
        np=types.SimpleNamespace(array=lambda values, dtype: CPUArray(values), int32=object()))
    author = next(n for n in ast.parse(AUTHOR.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
    execute = next(n for n in author.body if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    original_ast = compile_definition(execute, AUTHOR, namespace, strip_signature=True)
    class Model:
        def __init__(self):
            self._profile_step = 71
            self.execute_model_state = None
            self.routed_experts_initialized = False
            self.speculative_config = None
            self.parallel_config = types.SimpleNamespace(distributed_executor_backend="uni", data_parallel_size=1)
            self.cache_config = types.SimpleNamespace(kv_sharing_fast_prefill=False)
            self.input_batch = types.SimpleNamespace(num_reqs=1, req_ids=["cpu-request"])
            self.vllm_config = object()
        def synchronize_input_prep(self): return contextlib.nullcontext()
        def _update_states(self, value): trace.append("original_update_states")
        def kv_connector_no_forward(self, value, config):
            trace.append("original_no_forward")
            if native_exception is not None: raise native_exception
            return sentinel
        def _prepare_inputs(self, *args):
            trace.append("original_positive_prepare")
            raise StopPositiveCPUProbe("CPU sentinel after actual author positive branch reaches prepare")
    Model.execute_model = original_ast
    model = Model()
    original = model.execute_model
    adapter = F.FullStepFrameAdapter("cpu-fixture", reference(AUTHOR)["sha256"], "a" * 64, enabled=True)
    scalar = S.RuntimeScalarConnection()
    scalar.enabled = True
    scalar._adapter = adapter
    originals = (weakref.WeakMethod(original),)
    scalar_tree = ast.parse(SCALAR.read_bytes())
    scalar_connect = next(n for n in scalar_tree.body if isinstance(n, ast.FunctionDef) and n.name == "connect_runtime_scalar_observer")
    scalar_execute = next(n for n in ast.walk(scalar_connect) if isinstance(n, ast.FunctionDef) and n.name == "execute")
    scalar_namespace = dict(vars(S), originals=originals, connection=scalar, adapter=adapter, clock_ref=None)
    scalar_execute = compile_definition(scalar_execute, SCALAR, scalar_namespace)
    observer = W.WorkerObservationConnection()
    observer.enabled = True
    observer.scalar = scalar
    observer._runner = weakref.ref(model)
    observer.events = W.QueryOnlyEventObserver("cpu-fixture", "cpu_fixture")
    def factory(**kwargs): return CPUEvents(trace)
    worker_connect = next(n for n in ast.parse(WORKER.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == "connect_worker_observation")
    worker_execute = next(n for n in ast.walk(worker_connect) if isinstance(n, ast.FunctionDef) and n.name == "execute")
    worker_namespace = dict(vars(W), connection=observer, inner=scalar, execute_inner=scalar_execute,
        factory_ref=weakref.ref(factory))
    observed = compile_definition(worker_execute, WORKER, worker_namespace)
    model.execute_model = observed
    scalar._originals = originals
    class Inner:
        valid = True
        def __init__(self): self.observer = observer
        def export(self): return dict(valid=observer.enabled and adapter.valid, frames=[dataclasses_asdict(f) for f in adapter.frames], event_witnesses=[], production_qualified=False)
        def detach(self): trace.append("inner_detach")
    inner = Inner()
    capture = V2.PassiveNoForwardCapture(inner, model, metadata_types=TYPES,
        original_collector_ref=dict(sha256=V2.ORIGINAL_COLLECTOR_SHA), metadata_source_ref=reference(META))
    # Keep the explicit CPU Event factory alive for the frozen weak closure.
    return types.SimpleNamespace(model=model, original=original, capture=capture, observed=observed,
        adapter=adapter, scalar=scalar, observer=observer, trace=trace, sentinel=sentinel, factory=factory)


def dataclasses_asdict(value):
    from dataclasses import asdict
    return asdict(value)


def zero():
    value = TYPES[0].make_empty()
    value.kv_connector_metadata = object()
    value.finished_req_ids = {"finished-cpu-notification"}
    return value


class NoForwardTests(unittest.TestCase):
    def test_author_zero_AST_original_call_and_return(self):
        f = fixture()
        self.assertIs(f.model.execute_model(zero()), f.sentinel)
        self.assertEqual(f.trace, ["original_preemptions", "original_update_states", "original_no_forward"])
        self.assertEqual(f.model._profile_step, 72)
        self.assertEqual(f.adapter.frames, ())
        self.assertIsNone(f.adapter._first_ordinal)
        self.assertIsNone(f.observer.events.first_ordinal)
        self.assertTrue(f.observer.enabled)
        self.assertEqual(f.capture._no_forward[0]["original_native_ordinal"], 71)

    def test_original_observer_zero_disables_without_prepare_or_sample(self):
        f = fixture()
        self.assertIs(f.observed(zero()), f.sentinel)
        self.assertFalse(f.scalar.enabled)
        self.assertFalse(f.adapter.valid)
        self.assertIn("cpu_event_record", f.trace)
        self.assertEqual(f.trace.count("original_no_forward"), 1)
        self.assertEqual(f.adapter.frames, ())

    def test_original_zero_exception_preserved_once(self):
        error = RuntimeError("original CPU native callback exception")
        f = fixture(native_exception=error)
        with self.assertRaises(RuntimeError) as caught: f.model.execute_model(zero())
        self.assertIs(caught.exception, error)
        self.assertEqual(f.trace.count("original_no_forward"), 1)
        self.assertEqual(f.capture._no_forward[0]["original_exception_type"], "RuntimeError")

    def test_author_positive_AST_keeps_original_observer_and_prepare(self):
        f = fixture(); value = zero()
        value.total_num_scheduled_tokens = 1; value.num_scheduled_tokens = {"cpu-request": 1}
        with self.assertRaises(StopPositiveCPUProbe): f.model.execute_model(value)
        self.assertEqual(f.trace.count("original_positive_prepare"), 1)
        self.assertNotIn("original_no_forward", f.trace)
        self.assertEqual(f.trace.count("cpu_event_record"), 1)
        self.assertEqual(f.capture._no_forward, [])
        self.assertEqual(f.model._profile_step, 72)

    def test_missing_work_metadata_does_not_bypass(self):
        f = fixture(); value = zero(); del value.scheduled_encoder_inputs
        self.assertIs(f.model.execute_model(value), f.sentinel)
        self.assertEqual(f.capture._no_forward, [])
        self.assertIn("cpu_event_record", f.trace)

    def test_work_structure_despite_zero_does_not_bypass(self):
        f = fixture(); value = zero(); value.scheduled_cached_reqs.req_ids = ["cpu-request"]
        self.assertIs(f.model.execute_model(value), f.sentinel)
        self.assertEqual(f.capture._no_forward, [])

    def test_zero_after_first_model_call_not_bypassed(self):
        f = fixture(); f.capture._forward_started = True
        self.assertIs(f.model.execute_model(zero()), f.sentinel)
        self.assertIn("cpu_event_record", f.trace)
        self.assertEqual(f.capture._no_forward, [])

    def test_pending_scalar_or_event_not_bypassed(self):
        for kind in ("scalar", "event"):
            f = fixture()
            if kind == "scalar": f.adapter._pending = dict(phase="begun")
            else: f.observer.events.active = (71, object(), object())
            self.assertIs(f.model.execute_model(zero()), f.sentinel)
            self.assertEqual(f.capture._no_forward, [])

    def test_initial_ordinals_not_rewritten_or_frames_fabricated(self):
        f = fixture()
        for _ in range(3): self.assertIs(f.model.execute_model(zero()), f.sentinel)
        self.assertEqual([r["original_native_ordinal"] for r in f.capture._no_forward], [71,72,73])
        self.assertEqual(f.model._profile_step, 74)
        self.assertEqual(f.adapter.frames, ())
        self.assertEqual(f.observer.events.closed_steps, 0)
        self.assertFalse(f.capture.export()["initial_no_forward_observation"]["production_qualified"])

    def test_first_call_diagnostics_copied_only_counts(self):
        f = fixture(); value = zero(); value.new_block_ids_to_zero = [9]
        f.model.execute_model(value)
        row = f.capture.export()["initial_no_forward_observation"]["first_call_metadata_summary"]
        self.assertTrue(row["KV_metadata_present"])
        self.assertFalse(row["routed_initial_no_forward"])
        self.assertEqual(row["work_fields"]["new_block_ids_to_zero"]["length"], 1)
        json.dumps(row, allow_nan=False)
        self.assertNotIn("finished-cpu-notification", json.dumps(row))

    def test_metadata_bool_zero_refused(self):
        value = zero(); value.total_num_scheduled_tokens = False
        self.assertFalse(V2._strict_no_work_metadata(value, TYPES))

    def test_actual_keyword_zero_return_preserved_once(self):
        f = fixture()
        self.assertIs(f.model.execute_model(scheduler_output=zero()), f.sentinel)
        self.assertEqual(f.trace.count("original_no_forward"), 1)
        self.assertEqual(f.model._profile_step, 72)
        self.assertNotIn("cpu_event_record", f.trace)

    def test_nonempty_or_missing_actual_work_fields_refused(self):
        cases = (("scheduled_new_reqs", [object()]), ("scheduled_spec_decode_tokens", {"request": []}),
                 ("scheduled_encoder_inputs", {"request": []}), ("num_scheduled_tokens", {"request": 0}),
                 ("has_structured_output_requests", True), ("pending_structured_output_tokens", True),
                 ("ec_connector_metadata", object()), ("num_invalid_spec_tokens", {"request": 1}),
                 ("new_block_ids_to_zero", [1]))
        for name, value in cases:
            with self.subTest(field=name):
                candidate = zero(); setattr(candidate, name, value)
                self.assertFalse(V2._strict_no_work_metadata(candidate, TYPES))
        for name in ("req_ids", "resumed_req_ids", "new_token_ids", "all_token_ids", "new_block_ids", "num_computed_tokens", "num_output_tokens"):
            with self.subTest(cached_missing=name):
                candidate = zero(); delattr(candidate.scheduled_cached_reqs, name)
                self.assertFalse(V2._strict_no_work_metadata(candidate, TYPES))

    def test_no_forward_zero_frames_fails_original_exact128_export(self):
        f = fixture()
        for _ in range(3): f.model.execute_model(zero())
        original_capture = OLD.FullStepCapture(run_id="zero-CPU-rejection", origin="cpu_fixture",
            selected_offsets=(16,), action=None, max_steps=128)
        original_capture.observer = f.observer
        f.capture._inner = original_capture
        result = f.capture.export()
        self.assertFalse(result["valid"])
        self.assertEqual(result["frames"], [])
        self.assertEqual(result["event_witnesses"], [])
        self.assertTrue(any("exactly128" in reason for reason in result["failures"]))
        self.assertFalse(result["production_qualified"])
        self.assertFalse(result["initial_no_forward_observation"]["production_qualified"])

    def test_detach_restores_only_own_wrapper(self):
        f = fixture(); f.capture.detach()
        self.assertIs(f.model.execute_model, f.observed)
        self.assertEqual(f.trace, ["inner_detach"])
        self.assertTrue(f.capture._restored)

    def test_detach_preserves_foreign_override(self):
        f = fixture(); foreign = lambda value: None; f.model.execute_model = foreign
        f.capture.detach(); self.assertIs(f.model.execute_model, foreign)
        self.assertFalse(f.capture._restored)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    files = [ADAPTER, SCALAR, WORKER, AUTHOR, META, PREP / "runner/bounded_native_full_step_collector.py",
             PREP / "runner/bounded_native_full_step_collector_v2.py"]
    before = [reference(p) for p in files]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NoForwardTests))
    assert before == [reference(p) for p in files]
    assert not any(n == "torch" or n.startswith("torch.") or n == "vllm" or n.startswith("vllm.") or n == "numpy" for n in sys.modules)
    doc = dict(status="PASS_ACTUAL_SOURCE_AST_CPU_NO_FORWARD_ROUTING" if result.wasSuccessful() else "FAILED_CPU_ROUTING",
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        GPU_operations=0, RPC_calls=0, actual_CUDA_events_created=0, source_mutations=0,
        real_author_execute_AST_replayed=True, original_scalar_and_worker_wrapper_AST_replayed=True,
        actual_failed_CAL03_frame_recovered=False, native_failure_branch_proven=False,
        CPU_positive_probe_stops_at_prepare=True, complete128_GPU_qualification=False,
        source_refs=before, sources_unchanged=True, interpreter=sys.executable, version=sys.version)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(doc, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__": raise SystemExit(main())
