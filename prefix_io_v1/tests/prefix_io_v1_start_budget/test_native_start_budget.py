import ast
from pathlib import Path
import threading
from unittest.mock import Mock
import pytest
from py_kvcache.reactor import IoReactor
from prefix_io_control.start_budget import StartBudget,StartBudgetConfig
from ._fixture import rig,eventually

def joined(thread):
    thread.join(timeout=2);assert not thread.is_alive()

def test_exhausted_shared_budget_mandatory_wait_needs_no_new_epoch(monkeypatch):
    with rig(monkeypatch) as x:
        future=x.submit(files=3)
        x.start();eventually(lambda:x.budget.denied_quota>0)
        assert x.budget.ordinary_starts==1 and not future.done()
        joined(x.waiter({7}))
        assert future.result()==12288 and x.writes==3
        x.stop();assert x.budget.epoch_refreshes==1
        assert x.budget.reasons["mandatory"]==2 and not x.budget.faulted

def test_mandatory_does_not_override_physical_slot_or_compute_completion(monkeypatch):
    with rig(monkeypatch,slots=1,depth=8) as x:
        x.compute_ready.clear();x.copy_ready.clear()
        f=x.submit(files=3);x.start();w=x.waiter({7})
        assert x.copy_issued.wait(2) and x.copy_polled.wait(2)
        assert x.issued==1 and x.writes==0 and not f.done()
        x.compute_ready.set();assert not f.done()
        x.copy_ready.set();joined(w)
        assert f.result()==12288 and len(x.r.staging_pool.releases)==3
        x.stop()

def test_accepted_d2h_to_write_continues_when_quota_exhausted(monkeypatch):
    with rig(monkeypatch) as x:
        x.r.ring.auto=False
        f=x.submit(files=1);x.start()
        assert x.write_queued.wait(2)
        assert x.budget.used==1 and not f.done()
        x.r.ring.permits.release()
        assert f.result(timeout=2)==4096
        x.stop();assert x.budget.starts["store"]==1

def test_joint_load_store_native_pump_cannot_spend_separate_budgets(monkeypatch):
    with rig(monkeypatch) as x:
        load=x.load();store=x.submit(files=1)
        x.start();assert load.result(timeout=2)==4096
        eventually(lambda:x.budget.denied_quota>0)
        assert x.reads==1 and x.writes==0 and not store.done()
        x.clock[0]=100
        assert store.result(timeout=2)==4096
        x.stop()
        assert x.budget.starts==dict(load=1,store=1,preload=0)
        assert x.budget.ordinary_starts==2 and not x.budget.faulted

def test_pressure_reserve_eventually_allows_old_store_without_scheduler_epoch(monkeypatch):
    with rig(monkeypatch,mode="pressure",slots=1,depth=1,reserve=1,max_age=100) as x:
        f=x.submit(files=2);x.start()
        eventually(lambda:x.budget.denied_pressure>0)
        assert x.issued==0
        x.clock[0]=100
        eventually(lambda:x.issued==1)
        assert not f.done()  # Each successful start resets its own wait age.
        x.clock[0]=200
        assert f.result(timeout=2)==8192
        x.stop();assert x.budget.reasons["age"]==2

def test_shutdown_drains_accepted_stores_at_fixed_exhausted_allowance(monkeypatch):
    with rig(monkeypatch) as x:
        f=x.submit(files=3);x.start()
        eventually(lambda:x.budget.denied_quota>0)
        t=threading.Thread(target=x.r.shutdown,daemon=True);t.start();joined(t)
        assert f.result()==12288 and x.budget.reasons["shutdown"]==2
        assert not x.budget.waiting and not x.r._active

def test_short_write_keeps_native_failure_and_drains_accepted_chain(monkeypatch):
    with rig(monkeypatch,units=2) as x:
        x.r.ring.auto=False;x.r.ring.results[1]=4095
        f=x.submit(files=2);x.start()
        eventually(lambda:x.writes==2)
        x.r.ring.permits.release()
        with pytest.raises(IOError):f.result(timeout=2)
        x.r.ring.auto=True;x.stop()
        assert len(x.r.file_store.cleaned)==1 and len(x.r.file_store.finished)==1
        assert len(x.r.staging_pool.releases)==2

def test_duplicate_cqe_does_not_double_complete_or_double_charge(monkeypatch):
    with rig(monkeypatch,units=2) as x:
        x.r.ring.duplicate=True
        f=x.submit(files=2);calls=[];f.add_done_callback(lambda f:calls.append(f))
        x.start();assert f.result(timeout=2)==8192
        x.stop()
        assert len(calls)==1 and len(x.r.staging_pool.releases)==2
        assert x.budget.starts["store"]==2

def test_inflight_shared_preload_continues_and_fuses_two_consumers(monkeypatch):
    with rig(monkeypatch,share=True) as x:
        x.r.ring.auto=False;x.preload([b"shared"]);x.start()
        eventually(lambda:x.opens==1)
        a=x.load(job_id=20,hashes=[b"shared"])
        b=x.load(job_id=21,hashes=[b"shared"])
        eventually(lambda:len(x.r._preload_waiters.get(b"shared",()))==2)
        assert x.budget.used==1  # only original preload issued ordinary work
        x.r.ring.auto=True
        assert a.result(timeout=2)==b.result(timeout=2)==4096
        x.stop()
        assert x.reads==1 and x.h2d_launches==[dict(entries=2,bytes=8192)]
        assert x.budget.reasons["continuation"]==2
        assert not x.budget.faulted and x.budget.starts["load"]==2

def test_cached_preload_new_h2d_start_waits_for_next_shared_allowance(monkeypatch):
    with rig(monkeypatch,share=True) as x:
        x.preload([b"cached"]);x.start()
        eventually(lambda:b"cached" in x.r._shared_cached)
        f=x.load(hashes=[b"cached"])
        eventually(lambda:x.budget.denied_quota>0)
        assert not f.done() and x.reads==1
        x.clock[0]=100
        assert f.result(timeout=2)==4096
        x.stop();assert x.reads==1 and len(x.h2d_launches)==1
        assert x.budget.ordinary_starts==2

def test_budget_error_returns_to_native_progress_without_success_fabrication(monkeypatch):
    with rig(monkeypatch) as x:
        x.budget.allow=Mock(side_effect=ValueError("test injected"))
        f=x.submit(files=3);x.start()
        assert f.result(timeout=2)==12288
        x.stop();assert x.budget.faulted and x.budget.errors>0
        assert x.writes==3

def test_off_runs_original_pipeline_without_budget_clock_or_ticket(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        monkeypatch.setattr("py_kvcache.reactor.time.monotonic_ns",Mock(side_effect=AssertionError("off policy clock")))
        f=x.submit(files=3);x.start()
        assert f.result(timeout=2)==12288
        x.stop();assert x.r._prefix_start_budget is None
        assert x.writes==3 and x.budget.ordinary_starts==0

@pytest.mark.parametrize("bad",["object","no_progress"])
def test_invalid_constructor_options_rejected_before_hardware_allocation(monkeypatch,bad):
    ring=Mock(side_effect=AssertionError("no resources"))
    monkeypatch.setattr("py_kvcache.reactor.LiburingRing",ring)
    value=object() if bad=="object" else StartBudget(StartBudgetConfig("fixed",100,1,100))
    with pytest.raises((ValueError,TypeError)):
        IoReactor(config=None,file_mapper=None,layout=None,start_budget=value,
                  progress_run_id="test" if bad=="object" else None)
    ring.assert_not_called()

def test_strip_only_start_bridge_recovers_entire_original_native_ast():
    root=Path(__file__).resolve().parents[2]
    before=ast.parse((root/"artifacts/prefix_io_v1/server07-p3-08/before/reactor.py").read_text())
    after=ast.parse((root/"third_party/work/py-kvcache-p3-quota-cpu/py_kvcache/reactor.py").read_text())
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self,n):
            if n.name.startswith("_prefix_"):return None
            old=list(zip(n.args.kwonlyargs,n.args.kw_defaults))
            selected=[(a,d) for a,d in old if a.arg not in ("start_budget","stage_accounting")]
            n.args.kwonlyargs=[a for a,d in selected];n.args.kw_defaults=[d for a,d in selected]
            return self.generic_visit(n)
        def visit_Assign(self,n):
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id in ("budget","ticket"):return None
                if isinstance(t,ast.Attribute) and t.attr in ("_prefix_start_budget","_prefix_stage_accounting"):return None
            return self.generic_visit(n)
        def visit_If(self,n):
            if any(isinstance(v,ast.Name) and v.id in ("start_budget","stage_accounting","budget","ticket") for v in ast.walk(n.test)):
                return None
            n=self.generic_visit(n)
            return n if n.body else None
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr in ("_prefix_commit_start","_prefix_stage","_prefix_stage_copy"):
                return None
            return self.generic_visit(n)
        def visit_Call(self,n):
            n.keywords=[k for k in n.keywords if k.arg not in ("start_budget","stage_accounting")]
            return self.generic_visit(n)
    assert ast.dump(Strip().visit(after),include_attributes=False)==ast.dump(before,include_attributes=False)

def test_mandatory_load_supports_native_stores_without_dependency_guessing(monkeypatch):
    with rig(monkeypatch,slots=4,depth=2) as x:
        x.copy_ready.clear()
        load=x.load();store=x.submit(files=3)
        x.start()
        eventually(lambda:len(x.h2d_launches)==1 and x.budget.denied_quota>0)
        assert x.issued==0
        w=x.waiter({20})
        eventually(lambda:x.budget.reasons["mandatory_support"]>0)
        assert not load.done() and not store.done() and x.writes==0
        x.copy_ready.set();joined(w)
        assert load.result()==4096
        x.stop();assert store.result()==12288
        assert not x.budget.faulted

def test_full_physical_slot_does_not_consume_unused_start_allowance(monkeypatch):
    with rig(monkeypatch,units=2,slots=1,depth=8) as x:
        x.copy_ready.clear()
        f=x.submit(files=2);x.start()
        assert x.copy_issued.wait(2) and x.copy_polled.wait(2)
        assert x.budget.used==1 and x.issued==1
        x.copy_ready.set()
        assert f.result(timeout=2)==8192
        x.stop();assert x.budget.ordinary_starts==2

def test_misowned_optional_gate_falls_back_through_native_retirement(monkeypatch):
    with rig(monkeypatch) as x:
        x.budget.owner=-1
        f=x.submit(files=3);x.start()
        assert f.result(timeout=2)==12288
        x.stop()
        assert x.budget.faulted and x.budget.errors==1
        assert x.writes==3 and not x.budget.waiting

def test_single_budget_cannot_be_reused_by_second_reactor(monkeypatch):
    b=StartBudget(StartBudgetConfig("fixed",100,1,100))
    b.bind()
    ring=Mock(side_effect=AssertionError("no resources"))
    monkeypatch.setattr("py_kvcache.reactor.LiburingRing",ring)
    with pytest.raises(ValueError,match="another reactor"):
        IoReactor(config=None,file_mapper=None,layout=None,start_budget=b,progress_run_id="test")
    ring.assert_not_called()
