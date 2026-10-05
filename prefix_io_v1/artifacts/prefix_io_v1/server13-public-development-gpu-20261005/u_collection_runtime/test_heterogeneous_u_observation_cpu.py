"""Six bounded CPU reproductions. All geometry/events/timings are fixtures.

Actual pinned source bytes are used, but no framework, model, shared object,
GPU, RPC, source qualification receipt or historical success is produced.
"""
import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent


def load(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path.resolve())
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path.resolve()), "exec", dont_inherit=True), vars(module))
    return module


def actual(canonical, local):
    for root in (Path.cwd(), *HERE.parents):
        for rel in (canonical, local):
            p = root / rel
            if p.is_file():
                return p
    raise RuntimeError("UNBOUND actual pinned source: " + canonical)


F = load(HERE / "heterogeneous_full_step_frame_adapter.py", "_fixture_heterogeneous_frame")
C = load(HERE / "bounded_native_full_step_collector_v3.py", "_fixture_heterogeneous_collector")
OLD_FRAME = actual("artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py",
    "artifacts/prefix_io_v1_server09_migration_20261001/collector-candidate/p4_full_step_frame_adapter.py")
WORKER = actual("artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/g2_worker_observation.py",
    "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/g2_worker_observation.py")
SCALAR = actual("artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py",
    "artifacts/prefix_io_v1_server09_candidates/runtime_connector/p4_runtime_scalar_connector.py")
PREP = actual("artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/activation/control_observation/reserve_join.py",
    "artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/activation/control_observation/reserve_join.py").parents[2]


def geometry(contexts, prompts, scheduled, ordinal=0):
    ids = ["request" + str(i) for i in range(len(contexts))]
    runner = types.SimpleNamespace(speculative_config=None, use_async_scheduling=False,
        is_pooling_model=False, parallel_config=types.SimpleNamespace(pipeline_parallel_size=1,
        data_parallel_size=1, tensor_parallel_size=1), _profile_step=ordinal+1,
        input_batch=types.SimpleNamespace(num_reqs=len(ids), req_ids=ids,
            num_computed_tokens_cpu=contexts, num_prompt_tokens=prompts))
    scheduler = types.SimpleNamespace(num_scheduled_tokens=dict(zip(ids, scheduled)),
        total_num_scheduled_tokens=sum(scheduled), scheduled_spec_decode_tokens={})
    return runner, scheduler


class ObservationTests(unittest.TestCase):
    def test_homogeneous_exact_functions_preserved_and_replayed(self):
        original = ast.parse(OLD_FRAME.read_bytes())
        candidate = ast.parse((HERE / "heterogeneous_full_step_frame_adapter.py").read_bytes())
        for name in ("capture_prepared_frame", "capture_sampled_outputs"):
            a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == "_capture_homogeneous_" + name.removeprefix("capture_"))
            b.name = name
            self.assertEqual(ast.dump(a, include_attributes=False), ast.dump(b, include_attributes=False))
        runner, scheduler = geometry([145], [141], [1])
        frame = F.capture_prepared_frame(runner, scheduler, 0)
        self.assertEqual((frame.context_length, frame.step_kind, frame.exact_cell_eligible), (145, "decode", True))

    def test_actual_mixed_transition_reproduced_without_exact_cell(self):
        runner, scheduler = geometry([262, 128], [141, 142], [1, 14], 122)
        value = F.capture_prepared_frame(runner, scheduler, 122)
        self.assertEqual((value.step_kind, value.context_length, value.exact_cell_eligible), ("mixed", None, False))
        self.assertEqual((value.active_decode, value.prefill_tokens), (1, 14))
        output = types.SimpleNamespace(req_ids=["request0", "request1"], sampled_token_ids=[[7], [8]],
            req_id_to_index={"request0": 0, "request1": 1})
        self.assertEqual(F.capture_sampled_outputs(output, value), (("request0", (7,)), ("request1", (8,))))
        doc = json.loads(json.dumps(asdict(value)))
        F.validate_prepared_document(doc, 122)
        doc["context_length"] = 262
        with self.assertRaises(ValueError): F.validate_prepared_document(doc, 122)
        raw_frame = asdict(F.ClosedFrame(122, 100, 200, value, (("request0", (7,)), ("request1", (8,)))))
        owner = types.SimpleNamespace(export=lambda: {"frames": [raw_frame]})
        exported = C.HeterogeneousCapture(owner, {}).export()
        self.assertIs(type(exported["frames"][0]["prepared"]["rows"]), list)
        self.assertIs(type(exported["frames"][0]["outputs"]), list)
        F.validate_prepared_document(exported["frames"][0]["prepared"], 122)

    def test_partial_prefill_and_bad_geometry_fail_closed(self):
        runner, scheduler = geometry([262, 128], [141, 142], [1, 5])
        value = F.capture_prepared_frame(runner, scheduler, 0)
        output = types.SimpleNamespace(req_ids=["request0", "request1"], sampled_token_ids=[[7], []],
            req_id_to_index={"request0": 0, "request1": 1})
        F.capture_sampled_outputs(output, value)
        output.sampled_token_ids[1] = [9]
        with self.assertRaises(ValueError): F.capture_sampled_outputs(output, value)
        for contexts, prompts, scheduled in (([262, 128], [141, 142], [1, 15]), ([262, 128], [141, 142], [2, 14])):
            runner, scheduler = geometry(contexts, prompts, scheduled)
            with self.assertRaises(ValueError): F.capture_prepared_frame(runner, scheduler, 0)

    def test_query_only_midstream_retirement_preserves_136_ordinals(self):
        worker = load(WORKER, "_fixture_original_query_observer")
        class Event:
            def record(self): pass
            def query(self): return True
            def elapsed_time(self, other): return 0.01
            def synchronize(self): raise AssertionError("forbidden extra synchronization")
        observer = worker.QueryOnlyEventObserver("synthetic-fixture", "cpu_fixture", max_pending=128, max_steps=4096)
        owner = types.SimpleNamespace(observer=types.SimpleNamespace(resolve_ready=observer.resolve_ready))
        handle = C.HeterogeneousCapture(owner, {})
        for ordinal in range(136):
            observer.begin(ordinal, lambda **kwargs: Event())
            observer.finish(types.SimpleNamespace(native_step_ordinal=ordinal, gpu_elapsed_ns=None,
                prepared=types.SimpleNamespace(context_basis="pre_computed_tokens")))
            handle.resolve_after_original_step()
        self.assertEqual(observer.pending, [])
        self.assertEqual([r.ordinal for r in observer.diagnostics], list(range(136)))
        self.assertEqual((observer.max_pending, observer.max_steps), (128, 4096))

    def test_exact_private_loader_derivation_preserves_authentication(self):
        scalar = load(SCALAR, "_fixture_original_scalar")
        worker = load(WORKER, "_fixture_original_worker")
        old = load(PREP / "runner/bounded_native_full_step_collector.py", "_fixture_original_collector")
        scalar._collection_frame = F
        C._derive(scalar, SCALAR.read_bytes(), "connect_runtime_scalar_observer",
            [("load_frozen_adapter(adapter_source)", "_collection_frame")])
        worker._collection_scalar, worker._collection_frame = scalar, F
        C._derive(worker, WORKER.read_bytes(), "connect_worker_observation", [
            ('load_pinned(scalar_source, *FROZEN["scalar"])', "_collection_scalar"),
            ("scalar.load_frozen_adapter(frame_source)", "_collection_frame")])
        old._collection_worker, old._collection_scalar, old._collection_frame_path = worker, scalar, HERE / "heterogeneous_full_step_frame_adapter.py"
        C._derive(old, (PREP / "runner/bounded_native_full_step_collector.py").read_bytes(), "install", [
            ("common.load_ref(root, refs[common.WORKER])", "_collection_worker"),
            ('worker_module.load_pinned(common.safe(root, common.SCALAR), *worker_module.FROZEN["scalar"])', "_collection_scalar"),
            ("common.safe(root, common.FRAME)", "_collection_frame_path")])
        self.assertIn("_check_original_methods", scalar.connect_runtime_scalar_observer.__code__.co_names)
        self.assertIn("_check_original_methods", worker.connect_worker_observation.__code__.co_names)
        with self.assertRaises(ValueError): C._derive(scalar, SCALAR.read_bytes(), "connect_runtime_scalar_observer", [("not_a_load_site()", "_collection_frame")])

    def test_complete_mixed_capture_replay_and_failed_cuda_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root, refs = Path(temp), {}
            def put(relative, path):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                raw = target.read_bytes()
                row = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                refs[relative] = row
                return target, row
            collector_path, source_ref = put("runner/bounded_native_full_step_collector_v3.py", HERE / "bounded_native_full_step_collector_v3.py")
            _, frame_ref = put("runner/heterogeneous_full_step_frame_adapter.py", HERE / "heterogeneous_full_step_frame_adapter.py")
            declared = {"heterogeneous_full_step_frame_adapter.py": frame_ref}
            for name, path, relative in (("bounded_native_full_step_collector.py", PREP / "runner/bounded_native_full_step_collector.py", "runner/bounded_native_full_step_collector.py"),
                ("bounded_native_full_step_collector_v2.py", PREP / "runner/bounded_native_full_step_collector_v2.py", "runner/bounded_native_full_step_collector_v2.py"),
                ("worker", WORKER, "support/worker.py"), ("scalar", SCALAR, "support/scalar.py"), ("frame", OLD_FRAME, "support/frame.py")):
                _, declared[name] = put(relative, path)
            host_path, _ = put("support/host_control_observer.py", PREP / "activation/control_observation/host_control_observer.py")
            join_path, _ = put("support/reserve_join.py", PREP / "activation/control_observation/reserve_join.py")
            previous = sys.modules.get("host_control_observer")
            try:
                load(host_path, "host_control_observer")
                join = load(join_path, "_fixture_original_capture_join")
            finally:
                if previous is None: sys.modules.pop("host_control_observer", None)
                else: sys.modules["host_control_observer"] = previous
            module = load(collector_path, "_fixture_closed_collection_consumer")
            frames, witnesses = [], []
            for ordinal in range(128):
                runner, scheduler = geometry([141+ordinal, 142+ordinal], [141, 142], [1, 1], ordinal)
                prepared = json.loads(json.dumps(asdict(F.capture_prepared_frame(runner, scheduler, ordinal))))
                base = 10**9 + ordinal * 10000
                frames.append(dict(native_step_ordinal=ordinal, start_ns=base+3, end_ns=base+100,
                    prepared=prepared, outputs=[["request0", [ordinal]], ["request1", [ordinal+1]]],
                    gpu_elapsed_ns=None, existing_io=None, new_io=None, intended_timing_scope="full_decode_step"))
                witnesses.append(dict(native_step_ordinal=ordinal, start_record_before_ns=base,
                    start_record_after_ns=base+1, start_completed_query_ns=base+2, end_record_before_ns=base+101,
                    end_record_after_ns=base+102, end_completed_query_ns=base+103,
                    event_elapsed_source="torch.cuda.Event.elapsed_time", gpu_elapsed_ns=50))
            capture = dict(run_id="synthetic-fixture", origin="native_gpu_recording", valid=True,
                scope="bounded_original_full_step_stream_v1", failures=[], pending_event_pairs=0,
                open_event_pair=False, no_added_synchronization=True, cross_clock_absolute_mapping=False,
                frames=frames, event_witnesses=witnesses, selected_offsets=[], actions=[], heterogeneous_observation=dict(
                    schema="source_bound_heterogeneous_original_U_observation_v1", source_refs=declared,
                    heterogeneous_frame_count=128, mixed_frames_preserved=True, exact_finite_cost_cells_qualified=False,
                    pending_event_bound=128, full_frame_bound=4096,
                    progress_retirement="original_query_only_resolve_ready_after_original_step",
                    no_added_synchronization=True, table_issued=False, ordinary_I_authorized=False))
            rows = []
            for index in range(2):
                times = [10**9 + i*10000+200 for i in range(128)]
                rows.append(dict(native_request_id="request"+str(index), state="COMPLETED",
                    output_token_ids=[i+index for i in range(128)], token_return_ns=times, itl_ns=[10000]*127))
            frontend = dict(status="PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS", actual_request_stream_completed=True,
                failure_or_unsubmitted_requests=0, rows=rows, successful_requests=2, planned_requests=2)
            def verify(doc):
                return module.validate_collection_capture(root, refs, doc, frontend, run_id="synthetic-fixture",
                    expected_ordinals=list(range(128)), source_ref=source_ref, original_join=join)
            result = verify(capture)
            self.assertEqual((result["full_native_steps"], result["complete128_requests"]), (128, 2))
            self.assertIs(result["exact_finite_cost_cells_qualified"], False)
            for change in (lambda doc: doc.update(valid=False),
                lambda doc: doc["event_witnesses"][4].update(gpu_elapsed_ns=100000),
                lambda doc: doc["frames"][5]["prepared"].update(exact_cell_eligible=True),
                lambda doc: doc["frames"][6]["outputs"][0][1].clear()):
                bad = deepcopy(capture)
                change(bad)
                with self.assertRaises(ValueError): verify(bad)


if __name__ == "__main__":
    unittest.main()
