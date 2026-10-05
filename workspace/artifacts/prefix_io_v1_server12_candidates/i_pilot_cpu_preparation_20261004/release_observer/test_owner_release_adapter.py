"""CPU contracts and exact original scheduler method replays; no GPU imports."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import __future__

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
SOURCES = HERE.parent / "source_inputs"
BASE_SHA = "b24902ca76aa998c564218738ec02cfc3c96f5b12523a2df80765a489c04ebda"
BASE_PATHS = (
    SOURCES / "native_flush_probe.py",
    PROJECT / "artifacts/prefix_io_v1_server11_candidates/notification_v5_gpu_entry_path_revision/common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/native_flush_probe.py",
    PROJECT / "artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/native_flush_probe.py",
)
BASE = next((p for p in BASE_PATHS if p.is_file()), None)
if BASE is None or hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
    raise RuntimeError("the exact original NativeFlushProbe source is unavailable or mismatched")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adapter = load_module("release_adapter_cpu", HERE / "owner_release_adapter.py")
original = load_module("native_flush_original_cpu", BASE)


def state(*, generation=1, refs=0, pending=(10,), linked=True):
    return dict(block=1, owner_known=True, generation=generation, source_generation=1,
                same_generation=generation == 1, active_refs=refs, is_null=False,
                free_queue_linked=linked, protecting_jobs=list(pending),
                protecting_jobs_count=len(pending), parents_truncated=False)


def flush(parent_ids=(10,), **state_values):
    return dict(reason="restore_destination", parents_truncated=False,
                parents=[dict(job_id=j, is_store=True, pending_worker_acks=2,
                              source_sample_known=True, source_blocks=[state(pending=parent_ids, **state_values)])
                         for j in parent_ids],
                conflict_blocks=[dict(block=1, parent_ids=list(parent_ids), parents_truncated=False)])


def retired(jid=10, **state_values):
    return dict(kind="native_parent_retired", job_id=jid, source_blocks=[state(**state_values)])


def latest(observer):
    return observer.export()["resources"][-1]


class AdapterContracts(unittest.TestCase):
    def make(self, verified=True):
        return adapter.BoundedReleaseAdapter("cpu-fixture-not-GPU-evidence", 4096,
                                             worker_completion_verified=verified)

    def test_no_native_parent_cannot_create_release(self):
        observer = self.make()
        observer.observe_flush(flush(()))
        self.assertEqual(observer.export()["resources"], [])

    def test_all_original_parent_retirements_and_free_list_required(self):
        observer = self.make()
        observer.observe_flush(flush((10, 11)))
        observer.observe_native_retirement(retired(10, pending=(11,)))
        self.assertFalse(latest(observer)["native_reusable"])
        observer.observe_native_retirement(retired(11, pending=()))
        self.assertTrue(latest(observer)["native_reusable"])
        self.assertEqual(latest(observer)["reusable_now_bytes"], 4096)
        self.assertFalse(observer.export()["production_release_capability"])

    def test_retirement_without_verified_worker_semantics_stays_unknown(self):
        observer = self.make(False)
        observer.observe_flush(flush())
        observer.observe_native_retirement(retired(pending=()))
        self.assertIsNone(latest(observer)["native_reusable"])
        self.assertEqual(latest(observer)["reusable_now_bytes"], 0)

    def test_active_reference_does_not_earn_release(self):
        observer = self.make()
        observer.observe_flush(flush())
        observer.observe_native_retirement(retired(pending=(), refs=1))
        self.assertEqual(latest(observer)["reason"], "active_native_reference")
        self.assertEqual(latest(observer)["reusable_now_bytes"], 0)

    def test_free_queue_links_are_needed(self):
        observer = self.make()
        observer.observe_flush(flush())
        observer.observe_native_retirement(retired(pending=(), linked=False))
        self.assertEqual(latest(observer)["reason"], "not_on_native_free_queue")

    def test_generation_change_never_credits_old_source(self):
        observer = self.make()
        observer.observe_flush(flush())
        observer.observe_native_retirement(retired(pending=(), generation=2))
        self.assertEqual(latest(observer)["reason"], "generation_changed")
        self.assertIsNone(latest(observer)["native_reusable"])

    def test_unobserved_protector_remains_unknown(self):
        observer = self.make()
        observer.observe_flush(flush())
        observer.observe_native_retirement(retired(pending=(99,)))
        self.assertEqual(latest(observer)["reason"], "unobserved_native_protector")

    def test_omitted_source_sample_cannot_prove_membership(self):
        event = flush()
        event["parents"][0]["source_blocks"] = []
        observer = self.make()
        observer.observe_flush(event)
        observer.observe_native_retirement(retired(pending=()))
        self.assertIsNone(latest(observer)["native_reusable"])

    def test_truncated_parent_closure_is_not_observed(self):
        event = flush()
        event["parents_truncated"] = True
        observer = self.make()
        observer.observe_flush(event)
        self.assertEqual(observer.export()["resources"], [])

    def test_non_resource_flush_is_not_natural_resource_wait(self):
        observer = self.make()
        event = flush()
        event["reason"] = "all_requests_finished"
        observer.observe_flush(event)
        self.assertEqual(observer.export()["resources"], [])

    def test_cpu_future_is_not_a_native_retirement(self):
        observer = self.make()
        observer.observe_flush(flush())
        with self.assertRaises(ValueError):
            observer.observe_native_retirement(dict(kind="future_done", job_id=10))
        self.assertEqual(latest(observer)["reusable_now_bytes"], 0)

    def test_invalid_boolean_identity_rejected(self):
        observer = self.make()
        event = flush()
        event["conflict_blocks"][0]["block"] = True
        with self.assertRaises(ValueError):
            observer.observe_flush(event)

    def test_candidate_bound_omits_observation_without_consuming_native_work(self):
        observer = self.make()
        for bid in range(1, 10):
            event = flush()
            event["conflict_blocks"][0]["block"] = bid
            event["parents"][0]["source_blocks"][0]["block"] = bid
            observer.observe_flush(event)
        self.assertEqual(len(observer.export()["resources"]), 8)
        self.assertEqual(observer.export()["omitted_resources"], 1)

    def test_snapshots_are_read_only(self):
        observer = self.make()
        event = flush((10, 11))
        completion = retired(10, pending=(11,))
        prior = json.dumps([event, completion], sort_keys=True)
        observer.observe_flush(event)
        observer.observe_native_retirement(completion)
        self.assertEqual(json.dumps([event, completion], sort_keys=True), prior)

    def test_inconsistent_parent_generations_stay_unknown(self):
        event = flush((10, 11))
        event["parents"][1]["source_blocks"][0]["source_generation"] = 2
        observer = self.make()
        observer.observe_flush(event)
        self.assertIsNone(latest(observer)["native_reusable"])


class RealSourceBindings(unittest.TestCase):
    def test_downloaded_actual_server_source_hashes(self):
        entries = json.loads((SOURCES / "SOURCE_INPUTS.json").read_text(encoding="utf-8"))
        extra = json.loads((SOURCES / "KV_OFFLOAD_SOURCE_INPUT.json").read_text(encoding="utf-8"))
        if isinstance(extra, dict):
            extra = [extra]
        for entry in entries + extra:
            filename = entry["local_path"].replace("\\", "/").rsplit("/", 1)[-1]
            payload = (SOURCES / filename).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
            self.assertEqual(len(payload), entry["bytes"])

    def test_actual_owner_fields_and_native_ack_contract(self):
        sources = {n: ast.parse((SOURCES / (n + ".py")).read_text(encoding="utf-8"))
                   for n in ("block_pool", "kv_cache_utils", "scheduler", "worker", "kv_offload_worker")}
        block = next(x for x in sources["kv_cache_utils"].body if isinstance(x, ast.ClassDef) and x.name == "KVCacheBlock")
        fields = {x.target.id for x in block.body if isinstance(x, ast.AnnAssign) and isinstance(x.target, ast.Name)}
        self.assertTrue({"block_id", "ref_cnt", "is_null", "prev_free_block", "next_free_block"} <= fields)
        sched = next(x for x in sources["scheduler"].body if isinstance(x, ast.ClassDef) and x.name == "OffloadingConnectorScheduler")
        methods = {x.name: x for x in sched.body if isinstance(x, ast.FunctionDef)}
        update = ast.unparse(methods["update_connector_output"])
        self.assertIn("job_status.pending_count -= count", update)
        self.assertIn("assert job_status.pending_count == 0", update)
        self.assertLess(update.index("complete_store("), update.index("del self._jobs[job_id]"))
        self.assertIn("_remove_pending_job", update)
        self.assertIn("_prefix_flush_probe", ast.unparse(methods["_observe_prefix_flush"]))
        self.assertIn("self.worker.wait(kv_connector_metadata.jobs_to_flush)", ast.unparse(sources["worker"]))
        self.assertIn("assert transfer_result.success", ast.unparse(sources["worker"]))
        self.assertIn("self._connector_worker_meta.mark_completed(job_id)", ast.unparse(sources["worker"]))
        self.assertIn("finished.extend(handler.get_finished())", ast.unparse(sources["kv_offload_worker"]))

    def test_original_scheduler_partial_and_full_ack_replay(self):
        tree = ast.parse((SOURCES / "scheduler.py").read_text(encoding="utf-8"))
        source_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OffloadingConnectorScheduler")
        functions = [n for n in source_cls.body if isinstance(n, ast.FunctionDef) and
                     n.name in ("update_connector_output", "_remove_pending_job")]
        class Meta:
            def __init__(self, count):
                self.completed_jobs = {10: count}
                self.declined_req_ids = set()
        env = {"OffloadingWorkerMetadata": Meta}
        # Exact original AST methods with deferred annotations; no rewriting.
        exec(compile(ast.Module(body=functions, type_ignores=[]), "actual_scheduler_exact_methods",
                     "exec", flags=__future__.annotations.compiler_flag), env)
        class Scheduler:
            update_connector_output = env["update_connector_output"]
            _remove_pending_job = env["_remove_pending_job"]
            def _clear_preload_state(self, req):
                self.cleared.append(req)
        scheduler = Scheduler()
        store_calls = []
        scheduler.manager = SimpleNamespace(complete_store=lambda *args: store_calls.append(args))
        scheduler._declined_reqs = set()
        scheduler._stale_job_threshold = 0
        scheduler._block_id_to_pending_jobs = {1: {10}}
        scheduler._jobs = {10: SimpleNamespace(pending_count=2, is_store=True, keys={"source"},
                                             req_id="request", sliding_window_block_ids=None,
                                             non_sliding_window_block_ids=[1])}
        scheduler._req_status = {"request": SimpleNamespace(req=SimpleNamespace(is_finished=lambda: True),
                                                           req_context={}, transfer_jobs={10})}
        scheduler.cleared = []
        scheduler.update_connector_output(SimpleNamespace(kv_connector_worker_meta=Meta(1)))
        self.assertEqual(scheduler._jobs[10].pending_count, 1)
        self.assertEqual(scheduler._block_id_to_pending_jobs, {1: {10}})
        self.assertEqual(store_calls, [])
        scheduler.update_connector_output(SimpleNamespace(kv_connector_worker_meta=Meta(1)))
        self.assertEqual(scheduler._jobs, {})
        self.assertEqual(scheduler._block_id_to_pending_jobs, {})
        self.assertEqual(len(store_calls), 1)
        self.assertEqual(scheduler.cleared, ["request"])

    def test_extension_reuses_original_install_and_uninstall(self):
        subclass = adapter.make_observer_class(original.NativeFlushProbe)
        self.assertIs(subclass.install, original.NativeFlushProbe.install)
        self.assertIs(subclass.uninstall, original.NativeFlushProbe.uninstall)
        self.assertIs(subclass.safe, original.NativeFlushProbe.safe)
        observer = subclass("CPU-test", 4096)
        self.assertEqual(observer.export()["immediately_reusable_bytes"], None)
        self.assertFalse(observer.export()["release_observation"]["production_release_capability"])

    def test_no_backend_imported(self):
        self.assertFalse(any(n == "torch" or n.startswith("torch.") or n == "vllm" or n.startswith("vllm.")
                             or n == "py_kvcache" or n.startswith("py_kvcache.") for n in sys.modules))

    def test_real_probe_callbacks_follow_exact_owner_ack_retirement(self):
        tree = ast.parse((SOURCES / "scheduler.py").read_text(encoding="utf-8"))
        source_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OffloadingConnectorScheduler")
        functions = [n for n in source_cls.body if isinstance(n, ast.FunctionDef) and
                     n.name in ("update_connector_output", "_remove_pending_job")]
        class Meta:
            def __init__(self, count):
                self.completed_jobs = {10: count}
                self.declined_req_ids = set()
        env = {"OffloadingWorkerMetadata": Meta}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "actual_scheduler_exact_methods",
                     "exec", flags=__future__.annotations.compiler_flag), env)
        class Scheduler:
            update_connector_output = env["update_connector_output"]
            _remove_pending_job = env["_remove_pending_job"]
            def _clear_preload_state(self, req):
                pass
        class Pool:
            pass
        pool = Pool()
        block = SimpleNamespace(block_id=1, ref_cnt=0, is_null=False)
        previous, following = SimpleNamespace(), SimpleNamespace()
        block.prev_free_block, block.next_free_block = previous, following
        previous.next_free_block, following.prev_free_block = block, block
        pool.blocks = [SimpleNamespace(is_null=True), block]
        scheduler = Scheduler()
        scheduler._declined_reqs, scheduler._stale_job_threshold = set(), 0
        scheduler._block_id_to_pending_jobs = {1: {10}}
        scheduler._jobs = {10: SimpleNamespace(pending_count=2, is_store=True, keys={"source"},
                                             req_id="request", sliding_window_block_ids=None,
                                             non_sliding_window_block_ids=[1])}
        scheduler._req_status = {"request": SimpleNamespace(req=SimpleNamespace(is_finished=lambda: True),
                                                           req_context={}, transfer_jobs={10})}
        calls = []
        scheduler.manager = SimpleNamespace(complete_store=lambda *a: calls.append(a))
        subclass = adapter.make_observer_class(original.NativeFlushProbe)
        observer = subclass("CPU-callback-fixture-not-GPU-evidence", 4096, worker_completion_verified=True)
        observer.safe(observer.allocated, pool, [block])
        observer.safe(observer.stores, scheduler, {10: object()})
        observer.safe(observer.flush, scheduler, "restore_destination", "request", [1], None)
        self.assertEqual(observer.export()["release_observation"]["resources"][0]["potential_bytes"], 4096)
        for ordinal in (1, 2):
            output = SimpleNamespace(kv_connector_worker_meta=Meta(1))
            scheduler.update_connector_output(output)  # exact original method first
            observer.safe(observer.completed, scheduler, output)
            self.assertEqual(observer.retired_jobs, 0 if ordinal == 1 else 1)
        exported = observer.export()
        self.assertEqual(exported["errors"], 0)
        self.assertTrue(exported["release_observation"]["resources"][0]["native_reusable"])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(exported["immediately_reusable_bytes"])
        self.assertFalse(exported["release_observation"]["production_release_capability"])

    def test_original_failure_containment_preserves_optional_observation(self):
        subclass = adapter.make_observer_class(original.NativeFlushProbe)
        observer = subclass("CPU-failure-fixture", 4096)
        observer.safe(lambda: (_ for _ in ()).throw(ValueError("CPU observer fixture")))
        self.assertTrue(observer.faulted)
        self.assertEqual(observer.errors, 1)
        self.assertIsNone(observer.safe(lambda: 7))


if __name__ == "__main__":
    unittest.main()
