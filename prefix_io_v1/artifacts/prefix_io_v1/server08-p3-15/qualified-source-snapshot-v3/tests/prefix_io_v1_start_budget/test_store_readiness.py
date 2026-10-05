import threading
from concurrent.futures import Future
from types import SimpleNamespace as NS
import pytest
from prefix_io_control.store_readiness import StoreReadinessProbe
from tests.prefix_io_v1_start_budget._fixture import rig,eventually

def native():
    r=NS(_worker=threading.current_thread(),_prefix_progress=NS(run_id="r"),
        _active=[],_pending_copies=[],_inflight={},_data_inflight=1,iodepth=1,
        staging_pool=NS(free_count=0),is_mandatory=lambda f:True)
    return r

# Weak references need a normal instance (SimpleNamespace is not weak-referenceable).
class Reactor: pass

def fake():
    r=Reactor();r.__dict__.update(native().__dict__);return r

def job(jid=7):
    class NeverQuery:
        def query(self):raise AssertionError("observer must not query CUDA")
    return NS(is_store=True,failed=None,future_set=False,job_id=jid,total_files=3,
        next_file_index=1,inflight_files=1,done_files=0,future=Future(),compute_event=NeverQuery())

def test_bounded_samples_and_no_false_causal_or_release_claims():
    r=fake();j=job();r._active=[j]
    r._pending_copies=[NS(is_store=True,job=j,end_event=j.compute_event)]
    p=StoreReadinessProbe(run_id="r",max_samples=2,interval_ns=1,clock=iter(range(5)).__next__)
    for _ in range(5):p(r)
    assert not p.faulted and p.samples==5 and p.overwritten==0
    row=p.records[-1]["state"]["stores"][0]
    assert row["queued_files"]==2 and row["mandatory"]
    assert row["d2h_pending_observed"]==1
    assert row["compute_event_state"]=="not_queried"
    assert row["immediate_reusable_gpu_bytes"] is None
    assert p.counts["device_capacity_full"]==5
    with pytest.raises(RuntimeError):p.export()

def test_truncation_is_visible_and_not_unbounded_scan():
    r=fake();r._active=[job(i) for i in range(80)]
    r._inflight={i:NS(op_kind="write",job=r._active[-1]) for i in range(100)}
    p=StoreReadinessProbe(run_id="r",clock=lambda:0);p(r)
    assert not p.faulted and p.truncated_samples==1
    assert p.records[0]["state"]["parents_observed"]==32
    assert len(p.records[0]["state"]["stores"])==32
    assert all(x["pending_counts_are_lower_bounds"] for x in p.records[0]["state"]["stores"])

def test_sampling_interval_and_clock_regression_invalidate():
    now=[10];r=fake()
    p=StoreReadinessProbe(run_id="r",interval_ns=10,clock=lambda:now[0])
    p(r);now[0]=15;p(r);assert p.samples==1
    now[0]=9;p(r);assert p.faulted

def test_bad_owner_run_and_reuse_disable_only_observation():
    r=fake();p=StoreReadinessProbe(run_id="wrong");p(r);assert p.faulted
    p=StoreReadinessProbe(run_id="r",clock=lambda:0);p(r);p(fake());assert p.faulted
    p=StoreReadinessProbe(run_id="r");r._worker=NS(ident=-1);p(r);assert p.faulted

def test_native_mock_pipeline_reports_pending_then_drains_without_policy(monkeypatch):
    with rig(monkeypatch,enabled=False,slots=2,depth=1) as x:
        p=StoreReadinessProbe(run_id="cpu-run",interval_ns=1)
        x.r._observation_sink=p
        x.copy_ready.clear()
        f=x.submit(files=3)
        x.start()
        eventually(lambda:p.counts["queued_stores"]>0)
        assert not f.done() and x.writes==0
        waiter=x.waiter({7})
        eventually(lambda:p.counts["mandatory_stores"]>0)
        x.copy_ready.set();waiter.join(2)
        assert not waiter.is_alive() and f.result()==3*4096
        x.stop()
        report=p.export()
        assert not report["faulted"] and report["native_scheduling_modified"] is False
        assert report["inferred_gpu_release_bytes"] is None
        assert len(x.r.file_store.finished)==3

def test_probe_failure_does_not_fail_native_completion(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        p=StoreReadinessProbe(run_id="wrong")
        x.r._observation_sink=p
        f=x.submit(files=1);x.start()
        assert f.result(timeout=2)==4096
        x.stop()
        assert p.faulted and x.r.file_store.finished

def test_changed_states_remain_bounded_and_keep_observed_time_ranges():
    r=fake();j=job();r._active=[j]
    p=StoreReadinessProbe(run_id="r",max_samples=2,interval_ns=1,clock=iter(range(6)).__next__)
    p(r);p(r)
    assert p.records[0]["first_time_ns"]==0 and p.records[0]["last_time_ns"]==1
    assert p.records[0]["samples"]==2
    j.next_file_index=2;p(r);p(r)
    j.next_file_index=3;p(r);p(r)
    assert len(p.records)==2 and p.overwritten==1
    assert p.records[0]["first_time_ns"]==2 and p.records[1]["last_time_ns"]==5

def test_mandatory_filter_keeps_only_required_store_windows():
    r=fake();r._active=[job()]
    r.is_mandatory=lambda f:False
    p=StoreReadinessProbe(run_id="r",mandatory_only=True,interval_ns=1,clock=iter(range(3)).__next__)
    p(r);assert p.samples==0
    r.is_mandatory=lambda f:True
    p(r);assert p.samples==1 and p.records[0]["first_time_ns"]==1

def test_worker_composition_preserves_sink_and_native_shutdown(monkeypatch):
    import importlib
    import store_readiness_worker_probe as module
    native_state=importlib.import_module("vllm.distributed.kv_transfer.kv_transfer_state")
    with rig(monkeypatch,enabled=False,depth=1) as x:
        calls=[];actions=[]
        old=lambda r:calls.append(1)
        x.r._observation_sink=old
        cw=NS(worker=NS(handlers=[x.handler]))
        monkeypatch.setattr(native_state,"get_kv_transfer_group",lambda:NS(connector_worker=cw))
        monkeypatch.setattr(module,"original",lambda w,a,limits=None:actions.append(a) or {})
        worker=NS()
        module.worker_probe(worker,"install")
        module.worker_probe(worker,"start")
        with pytest.raises(RuntimeError):module.worker_probe(worker,"start")
        x.copy_ready.clear();f=x.submit(files=3);x.start();waiter=x.waiter({7})
        eventually(lambda:worker._readiness_probes[0][-1].samples>0)
        x.copy_ready.set();waiter.join(2)
        assert not waiter.is_alive() and f.result()==12288
        result=module.worker_probe(worker,"finish")
        assert calls and actions==["install","start","finish"]
        assert x.r._observation_sink is old and not x.r._worker.is_alive()
        assert len(result["store_readiness"])==1
        assert not result["store_readiness"][0]["faulted"]

def test_experiment_driver_preserved_outside_opt_in_observer_flag():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    prior=root/"artifacts/prefix_io_v1/server07-p3-11/before-model-observer/experiments/prefix_io_v1/scripts/run_concurrent_pilot.py"
    current=root/"experiments/prefix_io_v1/scripts/run_concurrent_pilot.py"
    class RemoveProbe(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and any(isinstance(a,ast.Constant) and a.value in ("--store-readiness-probe","--storage-registration") for a in node.value.args):
                return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="source_registration":
                return None
            return self.generic_visit(node)
        def visit_If(self,node):
            # P314 optional preparation is independently compared against its
            # frozen pre-change complete driver AST; retain the original path.
            if (isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Attribute)
                    and node.test.left.attr=="storage_registration"):
                return self.visit(node.body[0])
            if (isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name)
                    and node.test.left.id=="source_registration"):
                return None
            if isinstance(node.test,ast.Attribute) and node.test.attr=="store_readiness_probe":
                return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            node.keywords=[k for k in node.keywords if k.arg!="store_readiness_probe"]
            return self.generic_visit(node)
    from tests.prefix_io_v1_simple_stage_model.default_ast import strip_simple
    assert ast.dump(RemoveProbe().visit(strip_simple(ast.parse(current.read_text()))))==ast.dump(ast.parse(prior.read_text()))
