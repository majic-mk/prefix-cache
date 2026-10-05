import ast,threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from py_kvcache.reactor import IoReactor
from prefix_io_control.store_order import MandatoryStoreOrder
from prefix_io_control.order_options import KEY,parse_order_options,build_order_kwargs
from tests.prefix_io_v1_start_budget._fixture import rig,eventually

def attach(x):
    order=MandatoryStoreOrder();order.bind();x.r._prefix_store_order=order
    issued=[];native=x.r._schedule_store_copy
    def record(job,index,slot):
        issued.append((job.job_id,index));return native(job,index,slot)
    x.r._schedule_store_copy=record
    return order,issued

@pytest.mark.parametrize("enabled",[False,True])
def test_native_parent_order_and_one_completion_each(monkeypatch,enabled):
    with rig(monkeypatch,enabled=False,slots=2,depth=1) as x:
        order,issued=attach(x)
        if not enabled:x.r._prefix_store_order=None
        x.r.ring.auto=False
        large=x.submit(files=4,job_id=7);small=x.submit(files=1,job_id=8)
        assert x.coordinator.request_mandatory([small])
        done=[];large.add_done_callback(lambda f:done.append(7));small.add_done_callback(lambda f:done.append(8))
        x.start();eventually(lambda:x.writes==1)
        assert issued==[(8 if enabled else 7,0)] and not large.done() and not small.done()
        x.r.ring.permits.release()
        if enabled:
            eventually(small.done);assert not large.done()
        x.r.ring.auto=True;x.stop()
        assert large.result()==16384 and small.result()==4096
        assert sorted(done)==[7,8] and len(x.r.staging_pool.releases)==5
        assert len(issued)==len(set(issued))==5
        if enabled:assert order.reordered_passes>0 and not order.faulted

def test_existing_copy_and_write_continue_before_priority_issue(monkeypatch):
    with rig(monkeypatch,enabled=False,slots=2,depth=1) as x:
        order,issued=attach(x);x.copy_ready.clear();x.r.ring.auto=False
        large=x.submit(files=3,job_id=7);x.start();eventually(lambda:len(issued)==1)
        small=x.submit(files=1,job_id=8)
        x.coordinator.request_mandatory([small]);eventually(lambda:small in x.r._prefix_progress.required)
        assert issued==[(7,0)] and x.writes==0
        x.copy_ready.set();eventually(lambda:x.writes==1)
        assert not small.done() and not large.done()
        x.r.ring.permits.release();eventually(lambda:x.writes==2)
        assert issued[:2]==[(7,0),(8,0)] and not large.done()
        x.r.ring.permits.release();eventually(small.done)
        assert not large.done()
        x.r.ring.auto=True;x.stop();assert large.result()==12288
        assert len(x.r.staging_pool.releases)==4

def test_order_cannot_bypass_staging_or_compute_event(monkeypatch):
    with rig(monkeypatch,enabled=False,slots=1,depth=1) as x:
        attach(x);held=x.r.staging_pool.try_reserve();x.compute_ready.clear()
        f=x.submit(files=1);x.coordinator.request_mandatory([f]);x.start()
        eventually(lambda:f in x.r._prefix_progress.required)
        assert x.issued==0 and x.writes==0 and not f.done()
        x.r.staging_pool.release(held.index);eventually(lambda:x.issued==1)
        assert x.writes==0 and not f.done()
        x.compute_ready.set();assert f.result(timeout=2)==4096;x.stop()

def test_outside_window_remains_owned_and_eventually_finishes(monkeypatch):
    with rig(monkeypatch,enabled=False,slots=2,depth=1) as x:
        order,issued=attach(x);x.r.ring.auto=False
        futures=[x.submit(files=1,job_id=i) for i in range(34)]
        x.coordinator.request_mandatory([futures[-1]]);x.start();eventually(lambda:x.writes==1)
        assert issued==[(0,0)] and order.truncated_windows>0
        x.r.ring.auto=True;x.stop()
        assert all(f.result()==4096 for f in futures)
        assert len(issued)==len(set(issued))==34 and {j for j,i in issued}==set(range(34))

def test_shared_preload_read_and_fused_consumers_preserved(monkeypatch):
    with rig(monkeypatch,enabled=False,share=True) as x:
        attach(x);x.r.ring.auto=False;x.preload([b"shared"]);x.start()
        eventually(lambda:x.opens==1)
        a=x.load(job_id=20,hashes=[b"shared"]);b=x.load(job_id=21,hashes=[b"shared"])
        eventually(lambda:len(x.r._preload_waiters.get(b"shared",()))==2)
        x.coordinator.request_mandatory([a,b]);x.r.ring.auto=True
        assert a.result(timeout=2)==b.result(timeout=2)==4096;x.stop()
        assert x.reads==1 and x.h2d_launches==[dict(entries=2,bytes=8192)]

def test_short_write_keeps_native_error_and_releases_once(monkeypatch):
    with rig(monkeypatch,enabled=False,slots=2,depth=1) as x:
        attach(x);x.r.ring.results[1]=4095
        f=x.submit(files=1);x.coordinator.request_mandatory([f]);x.start()
        with pytest.raises(IOError):f.result(timeout=2)
        x.stop();assert len(x.r.file_store.cleaned)==1 and not x.r.file_store.finished
        assert len(x.r.staging_pool.releases)==1

def test_duplicate_cqe_and_shutdown_preserve_all_accepted_parents(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        attach(x);x.r.ring.duplicate=True
        a=x.submit(files=2,job_id=7);b=x.submit(files=2,job_id=8)
        x.coordinator.request_mandatory([b]);x.start();x.stop()
        assert a.result()==b.result()==8192 and len(x.r.staging_pool.releases)==4

@pytest.mark.parametrize("fault",["owner","exception"])
def test_optional_order_fault_falls_back_without_false_success(monkeypatch,fault):
    with rig(monkeypatch,enabled=False,depth=1) as x:
        order,issued=attach(x)
        if fault=="owner":order.owner=-1
        else:order.ordered=Mock(side_effect=ValueError("injected"))
        a=x.submit(files=2,job_id=7);b=x.submit(files=1,job_id=8)
        x.coordinator.request_mandatory([b]);x.start()
        assert a.result(timeout=2)==8192 and b.result(timeout=2)==4096;x.stop()
        assert order.faulted and issued[0]==(7,0)

def test_off_does_not_call_order_or_clock(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        monkeypatch.setattr(MandatoryStoreOrder,"ordered",Mock(side_effect=AssertionError("off")))
        monkeypatch.setattr("py_kvcache.reactor.time.monotonic_ns",Mock(side_effect=AssertionError("off clock")))
        f=x.submit(files=2);x.start();assert f.result(timeout=2)==8192;x.stop()

@pytest.mark.parametrize("bad",["type","no_progress","reuse","quota"])
def test_constructor_rejects_before_native_resources(monkeypatch,bad):
    from prefix_io_control.start_budget import StartBudget,StartBudgetConfig
    obj=object() if bad=="type" else MandatoryStoreOrder()
    if bad=="reuse":obj.bind()
    ring=Mock(side_effect=AssertionError("resources allocated"))
    monkeypatch.setattr("py_kvcache.reactor.LiburingRing",ring)
    kw=dict(store_order=obj,progress_run_id=None if bad=="no_progress" else "r")
    if bad=="quota":kw["start_budget"]=StartBudget(StartBudgetConfig("fixed",100,1,100))
    with pytest.raises((ValueError,TypeError)):
        IoReactor(config=None,file_mapper=None,layout=None,**kw)
    ring.assert_not_called()

def raw(**changes):
    return {KEY:dict(schema_version=1,mode="pressure",run_id="r",candidate_parents=32,**changes)}

def test_strict_adapter_off_has_no_new_runtime_state():
    from prefix_io_control.start_options import parse_start_options,build_start_kwargs
    assert build_order_kwargs(parse_order_options({}))==build_start_kwargs(parse_start_options({}))=={}
    assert build_order_kwargs(parse_order_options({KEY:{"mode":"off"}}))=={}
    x=build_order_kwargs(parse_order_options(raw()))
    assert x["progress_run_id"]=="r" and isinstance(x["store_order"],MandatoryStoreOrder)
    assert "start_budget" not in x and "stage_accounting" not in x

@pytest.mark.parametrize("field,value",[
    ("candidate_parents",True),("candidate_parents",64),("schema_version",True),
    ("schema_version",2),("mode","joint"),("run_id",""),("run_id",4),("run_id","x"*129)])
def test_adapter_invalid_fields(field,value):
    data=raw();data[KEY][field]=value
    with pytest.raises(ValueError):parse_order_options(data)

def test_adapter_unknown_keys_identity_and_mixed_policies():
    data=raw();data[KEY]["unexpected"]=1
    with pytest.raises(ValueError):parse_order_options(data)
    with pytest.raises(ValueError):parse_order_options({KEY:{"mode":"off","run_id":"r"}})
    data=raw();data.update(prefix_io_observation_mode="shadow",prefix_io_observation_run_id="other")
    with pytest.raises(ValueError):parse_order_options(data)
    data=raw();data["prefix_io_start_budget"]=dict(schema_version=1,mode="shadow",run_id="r")
    with pytest.raises(ValueError):parse_order_options(data)

def test_native_ast_preserved_outside_order_bridge():
    root=Path(__file__).resolve().parents[2]
    before=ast.parse((root/"artifacts/prefix_io_v1/server07-p3-12/before/py_kvcache/reactor.py").read_text())
    after=ast.parse((root/"third_party/work/py-kvcache-p3-order-cpu/py_kvcache/reactor.py").read_text())
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self,n):
            if n.name=="_prefix_ordered_stores":return None
            pairs=[(a,d) for a,d in zip(n.args.kwonlyargs,n.args.kw_defaults) if a.arg!="store_order"]
            n.args.kwonlyargs=[a for a,d in pairs];n.args.kw_defaults=[d for a,d in pairs]
            return self.generic_visit(n)
        def visit_If(self,n):
            if any(isinstance(v,ast.Name) and v.id=="store_order" for v in ast.walk(n.test)):return None
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Attribute) and t.attr=="_prefix_store_order" for t in n.targets):return None
            return self.generic_visit(n)
        def visit_Call(self,n):
            if isinstance(n.func,ast.Attribute) and n.func.attr=="_prefix_ordered_stores":
                return ast.Attribute(value=ast.Name(id="self",ctx=ast.Load()),attr="_active",ctx=ast.Load())
            n.keywords=[k for k in n.keywords if k.arg!="store_order"]
            return self.generic_visit(n)
    assert ast.dump(Strip().visit(after))==ast.dump(before)

def test_vllm_only_adapter_imports_changed():
    root=Path(__file__).resolve().parents[2]
    old=(root/"artifacts/prefix_io_v1/server07-p3-12/before/py_kvcache/vllm.py").read_text()
    new=(root/"third_party/work/py-kvcache-p3-order-cpu/py_kvcache/vllm.py").read_text()
    new=new.replace("prefix_io_control.order_options","prefix_io_control.start_options")
    new=new.replace("parse_order_options","parse_start_options").replace("build_order_kwargs","build_start_kwargs")
    assert new==old
